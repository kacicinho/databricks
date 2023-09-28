from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType, StructField, StructType, FloatType
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from databricks_jobs.db_common import BANNED_KIND_IDS, build_keep_person_id, \
    build_vod_df_with_episode_infos, build_episode_df, build_broadcast_df_with_episode_info
from databricks_jobs.jobs.utils.popular_reco_utils import format_for_reco_output
from databricks_jobs.jobs.utils.utils import load_snowflake_table, \
    write_df_to_snowflake, get_snowflake_options


def compute_log10(col_name):
    return F.log10(1 + F.col(col_name))


def compute_log10_normalization_score(user_df, col_name, group_by=None):
    """Add a new column to the user_df with a min max normalization computation.
    The output column is named as the col_name but with a suffix "_norm".
    """
    minmax_scale = F.round(((compute_log10(col_name) - F.col('min')) / (F.col('max') - F.col('min'))), 2)
    minmax_expr = [(F.min(compute_log10(col_name))).alias('min'), F.max(compute_log10(col_name)).alias('max')]

    if group_by:
        minmax_df = user_df.groupBy(group_by).agg(*minmax_expr)
        user_df = user_df.join(minmax_df, [group_by])
    else:
        minmax_df = user_df.agg(*minmax_expr)
        user_df = user_df.join(minmax_df)

    return user_df. \
        withColumn(col_name + "_norm", F.when(F.col('max') == F.col('min'), 0.5).otherwise(minmax_scale)). \
        drop('min'). \
        drop('max')


class PopularPersonalityJob(Job):
    TOP_ORDER = 10
    MIN_DURATION_IN_MIN = 10
    FOLLOWS_LOOKBACK = timedelta(days=7)
    TOP_PERSON = 200
    USER_RECO_LIMIT = 50
    REQUIRED_NB_H_PER_AFF = 2
    DAILY_RECO_TABLE = "RECO_USER_PERSON_LATEST"

    def __init__(self, *args, **kwargs):
        super(PopularPersonalityJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=7)
        self.vod_delta = timedelta(days=3)

        self.options = get_snowflake_options(self.conf, 'PROD', 'DW', **{"sfWarehouse": "PROD_WH"})
        self.write_options = get_snowflake_options(self.conf, 'PROD', 'ML', **{"sfWarehouse": "PROD_WH"})

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        # 1 - Prepare available programs
        freely_available_programs_df = self.prepare_freely_available_programs_df()

        # 2 - Prepare available person
        personality_available_df = self.personality_available(freely_available_programs_df)

        # 3 - Build non personalized reco
        future_personality_with_infos_df = self.join_info_to_person(freely_available_programs_df,
                                                                    personality_available_df)
        general_reco_df = self.build_non_personalized_reco(future_personality_with_infos_df)

        # 4 - Get best affinities per user
        user_watch_history = PopularRecoJob.build_user_watch_history(self.spark, self.options, self.now)
        user_affinity_preference = PopularRecoJob.compute_user_affinity_preference(self.spark, self.options,
                                                                                   self.REQUIRED_NB_H_PER_AFF,
                                                                                   user_watch_history)
        user_df = self.prepare_user_df(user_affinity_preference)

        # 5 - Add penalization score for person that are too present
        user_penalized_df = self.build_user_reco_history()

        # 6 - Build user_id to person_id affinity score (& add user 0 reco)
        all_users_person_reco_df = self.build_user_reco(user_affinity_preference, general_reco_df, user_df, user_penalized_df)

        # 7 - Write daily recos to snowflake
        self.write_recos_to_snowflake(all_users_person_reco_df, self.DAILY_RECO_TABLE, write_mode="append")

    def prepare_user_df(self, user_affinity_preference):
        """Prepare users df to recommend. Add user 0 to the users.
        """

        user0 = self.spark.createDataFrame([0], IntegerType())
        user_df = user_affinity_preference. \
            select("USER_ID"). \
            dropDuplicates(). \
            union(user0)

        return user_df

    def prepare_freely_available_programs_df(self):
        """
        We get all:
         - the broadcast programs that match the timeframe [now, now + 7 days]
         - vod programs that are available from now and until minimum 3 days

        A little bit of filtering is applied to avoid programs not worthy.
        Filter on free bundles, eg.
        25=Free + Mango, 90=MLT+, 50=26=Extended, 60=36=OCS, 77=AdultSwim, 31=23=Ciné+, 98=StarZ
        """
        # 0 - Load tables
        prog_affinity_df = load_snowflake_table(self.spark, self.options, "external_sources.DAILY_PROG_AFFINITY")
        episode_info_df = build_episode_df(self.spark, self.options)
        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df,
                                                 self.now, self.vod_delta, min_duration_in_mins=15)
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df, self.now,
                                                            self.now + self.delta, min_duration_in_mins=10,
                                                            free_bundle=False, filter_kind=True)

        # 1 - Union VOD & Broadcast, filter on wanted bundles 
        broadcast_vod_available_programs_df = vod_df. \
            union(broadcast_df). \
            where(~F.col("REF_PROGRAM_KIND_ID").isin(BANNED_KIND_IDS)). \
            where("TVBUNDLE_ID in (25, 90, 26)"). \
            where(F.col("PRODUCTION_YEAR").isNull() | (F.col("PRODUCTION_YEAR") == 0) | (F.col("PRODUCTION_YEAR") > 1970))

        # 2 - Add affinity info
        prog_affinity_df_fixed = prog_affinity_df.dropDuplicates()
        broadcast_vod_available_programs_df = broadcast_vod_available_programs_df. \
            join(prog_affinity_df_fixed,
                 prog_affinity_df_fixed.PROGRAM_ID == broadcast_vod_available_programs_df.PROGRAM_ID, "left"). \
            drop(prog_affinity_df_fixed.PROGRAM_ID). \
            fillna(value="unknown", subset=["AFFINITY"]). \
            filter(F.col("AFFINITY") != "Enfants"). \
            select("PROGRAM_ID", "EPISODE_ID", "CHANNEL_ID", "DURATION", "REF_PROGRAM_KIND_ID",
                   "REF_PROGRAM_CATEGORY_ID", "TVBUNDLE_ID", "AFFINITY").distinct()

        return broadcast_vod_available_programs_df

    def personality_available(self, freely_available_programs_df):
        """
        From the availables programs, compute the personality available and rank their best affinities
        """
        person_df = load_snowflake_table(self.spark, self.options, "backend.person")
        episode_person_df = load_snowflake_table(self.spark, self.options, "backend.rel_episode_person")
        person_function = load_snowflake_table(self.spark, self.options, "backend.ref_person_function")

        # List of function to keep
        # Réalisateur(2), Acteur(4), Chef cuisinier(34), Chroniqueur(37), Sportif(93), Sujet(99),
        # Personnage d'animation(170), Invité (18, 129, 132), Performer (217)
        keep_functions = [2, 4, 34, 37, 93, 99, 170, 18, 129, 132, 217]

        # List of person_id to keep
        keep_person_id = build_keep_person_id(self.spark, self.options)

        # Filtering on personality:
        # - keep top order per function
        # - keep personalities in keep function
        base_personality_available_df = freely_available_programs_df. \
            join(episode_person_df, freely_available_programs_df.EPISODE_ID == episode_person_df.EPISODE_ID). \
            drop(episode_person_df.EPISODE_ID). \
            join(person_function, episode_person_df.REF_PERSON_FUNCTION_ID == person_function.ID). \
            drop(person_function.ID). \
            where(F.col("REF_PERSON_FUNCTION_ID").isin(*keep_functions)). \
            withColumn("NEW_ORDER", F.row_number().
                       over(Window.partitionBy("PROGRAM_ID", "EPISODE_ID", "REF_PERSON_FUNCTION_ID").orderBy("ORDER"))). \
            where(F.col("NEW_ORDER") <= self.TOP_ORDER). \
            join(person_df, episode_person_df.PERSON_ID == person_df.ID). \
            drop(person_df.ID). \
            join(keep_person_id, keep_person_id.ID == episode_person_df.PERSON_ID). \
            drop(keep_person_id.ID)

        # Rank person_id by affinity based on duration in availables programs
        personality_available_df = base_personality_available_df. \
            groupBy("PERSON_ID", "FIRST_NAME", "LAST_NAME", "AFFINITY"). \
            agg(F.sum("DURATION").alias("AFF_DUR")). \
            withColumn("RANK_PERSON_AFFINITY",
                       F.row_number().over(Window.partitionBy("PERSON_ID").orderBy(F.desc("AFF_DUR")))). \
            select("PERSON_ID", "FIRST_NAME", "LAST_NAME", "AFFINITY", "RANK_PERSON_AFFINITY")

        return personality_available_df

    def next_week_duration(self, freely_available_programs_df):
        episode_person_df = load_snowflake_table(self.spark, self.options, "backend.rel_episode_person")
        # Compute Tv duration & nb of distinct program for each personality in the next 7 days
        person_next_duration_df = freely_available_programs_df. \
            join(episode_person_df, episode_person_df.EPISODE_ID == freely_available_programs_df.EPISODE_ID). \
            drop(episode_person_df.EPISODE_ID). \
            groupby("PROGRAM_ID", "PERSON_ID"). \
            agg(F.mean("DURATION").alias("MEAN_DURATION")). \
            groupby("PERSON_ID"). \
            agg(F.sum("MEAN_DURATION").alias("nxt_wk_dur"),
                F.countDistinct(F.col("PROGRAM_ID")).alias("cnt_nxt_wk_prog"))

        return person_next_duration_df

    def prepare_most_followed_person(self):
        user_follow_df = load_snowflake_table(self.spark, self.options, "backend.user_follow_person")

        count_follows_person_df = user_follow_df. \
            where((user_follow_df.SOURCE == "molotov") & (user_follow_df.CREATED_AT <= self.now)). \
            groupby("PERSON_ID"). \
            agg(F.countDistinct("USER_ID").alias("tot_fols"))

        return count_follows_person_df

    def prepare_new_followed_person(self):
        user_follow_df = load_snowflake_table(self.spark, self.options, "backend.user_follow_person")
        # Compute the number of follows in last 7 days
        count_new_follows_person_df = user_follow_df. \
            where(user_follow_df.SOURCE == "molotov"). \
            where((F.col("CREATED_AT") > self.now - self.FOLLOWS_LOOKBACK) & (F.col("CREATED_AT") <= self.now)). \
            groupby("PERSON_ID"). \
            agg(F.countDistinct("USER_ID").alias("new_fols"))

        return count_new_follows_person_df

    def join_info_to_person(self, freely_available_programs_df, personality_available_df):
        """
        For this steps, we extract the four kind of person ranking :
        - next week live time
        - total watch
        - number of follows

        1 - retrieve pop infos
        2 - join them to the programs

        :param freely_available_programs_df:
        :return:
        """

        # 1 - Retrieve tables
        next_week_person_duration_df = self.next_week_duration(freely_available_programs_df)
        count_follows_person_df = self.prepare_most_followed_person()
        count_new_follows_person_df = self.prepare_new_followed_person()

        # 2- join them
        def join_additional_information_on_future_personality(future_personality_df, col_to_df_dict, default_vals):
            for col_name, additional_df in col_to_df_dict.items():
                future_personality_df = future_personality_df.join(additional_df,
                                                                   additional_df.PERSON_ID == future_personality_df.PERSON_ID,
                                                                   "left"). \
                    drop(additional_df.PERSON_ID). \
                    select(future_personality_df["*"], additional_df[col_name]). \
                    fillna(default_vals[col_name], subset=[col_name])
            return future_personality_df

        # Filter on "FIRST_NAME is not null" allows to filter 'sujet' role that are not relevant (cf. Covid19)
        future_personality_with_infos = join_additional_information_on_future_personality(
            personality_available_df,
            {"nxt_wk_dur": next_week_person_duration_df,
             "cnt_nxt_wk_prog": next_week_person_duration_df,
             "tot_fols": count_follows_person_df,
             "new_fols": count_new_follows_person_df},
            {"nxt_wk_dur": 0, "cnt_nxt_wk_prog": 0,
             "tot_fols": 0, "new_fols": 0}). \
            filter(F.col("FIRST_NAME").isNotNull())

        return future_personality_with_infos

    def build_user_reco(self, user_affinity_preference_df, general_reco_df, user_df, user_penalized_df):
        """
        Based on available personalities and their rank affinities with user affinity preference, we compute an user to person
        affinity score.
        """

        # Compute a user to person affinity score
        # - We mutliply the rank affinities of the user and the person_id (the better ranks are the closer
        # the score will be to 1.
        # - Then we take the max by person_id if a person_id is present in many user affinity preference
        # - We then sum the affinity score & the base rating (and adding more weight to the affinity score)
        affinity_rating = F.round(1 / (F.col("USER_AFFINITY_RANK") * F.col("RANK_PERSON_AFFINITY")), 3)
        user_person_reco_df = general_reco_df. \
            crossJoin(user_df). \
            repartition(F.col("USER_ID")). \
            join(user_affinity_preference_df.select("USER_ID", "AFFINITY", "USER_AFFINITY_RANK"), ["USER_ID", "AFFINITY"], "left"). \
            withColumn("affinity_rating", F.when(F.col("USER_AFFINITY_RANK").isNull(), 0).otherwise(affinity_rating)). \
            groupBy("USER_ID", "PERSON_ID"). \
            agg(*(F.max("affinity_rating").alias("affinity_rating"), F.max("base_rating").alias("base_rating"))). \
            join(user_penalized_df, ["USER_ID", "PERSON_ID"], "left"). \
            fillna(value=0, subset=["penalization_score_norm"]). \
            withColumn("rating", 3 * F.col("affinity_rating") + F.col("base_rating") - F.col("penalization_score_norm")). \
            drop("base_rating", "affinity_rating"). \
            withColumn("count_reco", F.row_number().
                       over(Window.partitionBy("USER_ID").orderBy(F.desc("rating")))). \
            where(F.col("count_reco") <= self.USER_RECO_LIMIT). \
            drop("count_reco")

        return user_person_reco_df

    def build_user_reco_history(self):
        """
        In this step, we use RECO_USER_PERSON_LATEST to create a penalization score for person too
        present in the rail.

        1. We compute the penalization score as 1 / rail_ranking for a given update_date
        2. We apply this score for person that are a least present 7 times in the past month
        3. We normalize the score to have a score between 0 and 1.
        """
        struct_schema = StructType([
            StructField('PERSON_ID', IntegerType(), nullable=False),
            StructField('rating', FloatType(), nullable=False)
        ])

        user_reco_lookback = timedelta(days=30)
        reco_history = load_snowflake_table(self.spark, self.options, f"ml.{self.DAILY_RECO_TABLE}"). \
            where(F.col("UPDATE_DATE") >= self.now - user_reco_lookback). \
            withColumn("parsed_json", F.from_json("RECOMMENDATIONS", ArrayType(struct_schema))).\
            select("USER_ID", "UPDATE_DATE", F.explode(F.col("parsed_json")).alias("parsed_json")).\
            select("USER_ID", "UPDATE_DATE", F.col("parsed_json.PERSON_ID").alias("PERSON_ID"),
                   F.col("parsed_json.rating").alias("rating")). \
            withColumn("rail_rank", F.row_number().
                       over(Window.partitionBy("USER_ID", "UPDATE_DATE").orderBy(F.desc("rating"))))

        user_penalized_df = reco_history. \
            withColumn("penalization_score", 1 / F.col("rail_rank")). \
            groupby("USER_ID", "PERSON_ID"). \
            agg(F.sum("penalization_score").alias("penalization_score"), F.count(F.lit(1)).alias("occurence")). \
            where(F.col("occurence") > 7)

        user_penalized_df = compute_log10_normalization_score(user_penalized_df, "penalization_score", "USER_ID"). \
            select("USER_ID", "PERSON_ID", "penalization_score_norm")

        return user_penalized_df

    def build_non_personalized_reco(self, user_personality_df):

        scoring_methods = ["nxt_wk_dur", "cnt_nxt_wk_prog", "tot_fols", "new_fols"]

        # Normalization scoring methods
        for col_name in scoring_methods:
            user_personality_df = compute_log10_normalization_score(user_personality_df, col_name)

        # Rating specificity:
        # - Number of next week prog & sum of next week duration are quite corrolated and
        # describe nearly the same behavior, we take the mean of the 2 ratings
        # - We put a penality on total followers (not to have always the most liked personality)
        # - NB: It may possible that multiple person_ids have the same base_rating, then we randomly take one of them
        # for the rank (e.g orderby(F.desc("base_rating"),"PERSON_ID")) )
        future_personality_rating_df = user_personality_df. \
            withColumn("base_rating", F.round(
                (F.col("cnt_nxt_wk_prog_norm") + F.col("nxt_wk_dur_norm")) / 2 + 0.6 * F.col("tot_fols_norm") +
                F.col("new_fols_norm"), 2)). \
            withColumn("rank_person", F.dense_rank().
                       over(Window.orderBy(F.desc("base_rating"), "PERSON_ID"))). \
            where(F.col("rank_person") <= self.TOP_PERSON). \
            select("PERSON_ID", "AFFINITY", "RANK_PERSON_AFFINITY", "base_rating")

        return future_personality_rating_df

    def write_recos_to_snowflake(self, user_top_personality_df, table_name,
                                 write_mode="append"):
        user_top_personality_df = format_for_reco_output(user_top_personality_df,
                                                         "recommendations", ("PERSON_ID", "rating"))
        final_user_recos_df = user_top_personality_df. \
            withColumn("UPDATE_DATE", F.lit(self.now))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)


if __name__ == "__main__":
    job = PopularPersonalityJob()
    job.launch()
