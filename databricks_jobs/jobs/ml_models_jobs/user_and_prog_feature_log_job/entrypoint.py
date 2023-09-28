from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.helpers import DictFormatter, SocioDemoBuilder, \
    SocioDemoAggregator, PivotFormatter
from databricks_jobs.jobs.utils.affinities import AFFINITIES
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake, \
    load_snowflake_query_df


class UserAndProgFeatureLogJob(Job):
    """
    Listing of the generated features

    Channel :
    - Total watch spent on this channel among socio-demo groups
                                              affinities
                                              categories

    User :
    - watch time per affinity
                     device
                     category
    - socie-demo : gender, age

    Program :
    - PRODUCTION_YEAR
    - REF_PROGRAM_CATEGORY_ID
    - REF_PROGRAM_KIND_ID
    - PROGRAM_DURATION
    - AFFINITY

    - score allo-ciné rating
    - famous_persons_in_cast
    - total_nb_follow_from_cast
    """

    BANNED_AFFINITIES = ["Beauté & Bien-être", "Web & Gaming", "Horreur", "Santé & Médecine",
                         "Éducation", "Westerns", "Courts Métrages", "Cinéma"]
    PERSONALITY_TOP_K = 100

    def __init__(self, *args, **kwargs):
        super(UserAndProgFeatureLogJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=30)

        # Read options
        self.options = get_snowflake_options(self.conf, "PROD", "PUBLIC")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 - Build preliminary dfs
        current_fact_watch, user_df, top_pers_df = self.build_preliminary_tables()

        # 2 - Build the feature dfs
        user_feature_df = self.build_user_feature_log(current_fact_watch, user_df, top_pers_df). \
            withColumn("DATE_DAY", F.lit(self.now))
        program_feature_df = self.build_program_feature_log(top_pers_df). \
            withColumn("DATE_DAY", F.lit(self.now))
        channel_stats = self.build_channel_feature_log(current_fact_watch, user_df). \
            withColumn("DATE_DAY", F.lit(self.now))

        write_df_to_snowflake(user_feature_df, self.options, "ML.USER_FEATURE_LOG", "append")
        write_df_to_snowflake(program_feature_df, self.options, "ML.PROGRAM_FEATURE_LOG", "append")
        write_df_to_snowflake(channel_stats, self.options, "ML.CHANNEL_FEATURE_LOG", "append")

    def build_preliminary_tables(self):
        """
        What do we do :
        - Add prog affinity to the considered segment of fact_watch
        - Compute user age
        - Find top personalities for later processing
        """
        affinity_df = load_snowflake_table(self.spark, self.options, "EXTERNAL_SOURCES.DAILY_PROG_AFFINITY")
        current_fact_watch = load_snowflake_table(self.spark, self.options, "DW.FACT_WATCH")

        current_fact_watch = current_fact_watch. \
            where((F.col("REAL_START_AT") > self.now - self.delta) & (F.col("REAL_START_AT") < self.now)). \
            join(affinity_df, affinity_df.PROGRAM_ID == current_fact_watch.PROGRAM_ID). \
            drop(affinity_df.PROGRAM_ID). \
            select("USER_ID", "CHANNEL_ID", "PROGRAM_ID", "AFFINITY", "DURATION",
                   "ACTION_DEVICE_TYPE", "CATEGORY_ID")

        user_df = load_snowflake_table(self.spark, self.options, "backend.user_raw"). \
            select("ID", "GENDER", "BIRTHDAY"). \
            withColumn("age", F.floor(F.datediff(F.current_date(), F.col("BIRTHDAY"))) / 365). \
            drop("BIRTHDAY")

        top_pers_df = self.get_top_personalities_to_pids(top_k=self.PERSONALITY_TOP_K)

        return current_fact_watch, user_df, top_pers_df

    def build_channel_feature_log(self, fact_watch_df, user_df):
        """
        Features :
        - Watch spent on this channel among socio-demo groups
                                            affinities
                                            categories
        """
        # 1 - Compute socio-demo stats for the channel
        socio_demos = [
            SocioDemoBuilder(3, 100, 'M'),
            SocioDemoBuilder(14, 100, 'M'),
            SocioDemoBuilder(14, 25, 'M'),
            SocioDemoBuilder(14, 35, 'M'),
            SocioDemoBuilder(14, 50, 'M'),
            SocioDemoBuilder(24, 50, 'M'),
            SocioDemoBuilder(24, 60, 'M'),
            SocioDemoBuilder(34, 60, 'M'),
            SocioDemoBuilder(49, 100, 'M'),
            SocioDemoBuilder(3, 100, 'F'),
            SocioDemoBuilder(14, 100, 'F'),
            SocioDemoBuilder(14, 25, 'F'),
            SocioDemoBuilder(14, 35, 'F'),
            SocioDemoBuilder(14, 50, 'F'),
            SocioDemoBuilder(24, 50, 'F'),
            SocioDemoBuilder(24, 60, 'F'),
            SocioDemoBuilder(34, 60, 'F'),
            SocioDemoBuilder(49, 100, 'F')
        ]
        socio_demo_aggregators = [SocioDemoAggregator(sociodemo, "DURATION") for sociodemo in socio_demos]

        watch_type_per_channel_df = fact_watch_df. \
            join(user_df, user_df.ID == fact_watch_df.USER_ID). \
            select("CHANNEL_ID", *[sda.op_fn() for sda in socio_demo_aggregators]). \
            groupBy("CHANNEL_ID"). \
            agg(*[F.sum(sda.formatted_col_name).alias(sda.formatted_col_name) for sda in socio_demo_aggregators])

        # 2 - Compute watch time among affinities and categories
        channel_formatter = dict()
        pivot_cols = {
            "AFFINITY": [x for x in AFFINITIES if
                         x not in self.BANNED_AFFINITIES + ["Séries", "Film", "Documentaires"]],
            "CATEGORY_ID": list(range(11))
        }
        for column_name in ["AFFINITY", "CATEGORY_ID"]:
            # The following object handles the pivot and the proper naming of the resulting columns
            channel_formatter[column_name] = PivotFormatter("DURATION", column_name,
                                                            pivot_cols[column_name], prefix="channel")

        pivoted_df = dict()
        for i, column_name in enumerate(["AFFINITY", "CATEGORY_ID"]):
            # The following object handles the pivot and the proper naming of the resulting columns
            pivoted_df[column_name] = channel_formatter[column_name].pivot_and_sum(fact_watch_df, "CHANNEL_ID")

        channel_feature_df = watch_type_per_channel_df.alias("main"). \
            join(pivoted_df["AFFINITY"].alias("affinity"), on="CHANNEL_ID", how="left"). \
            drop("affinity.CHANNEL_ID"). \
            join(pivoted_df["CATEGORY_ID"].alias("category"), on="CHANNEL_ID", how="left"). \
            drop("category.CHANNEL_ID")
        return channel_feature_df

    def build_user_feature_log(self, fact_watch_df, user_df_with_age, prog_personality_df):
        """
        Features :

        watch time per affinity
        watch time per device
        watch time per category
        socie-demo : gender, age
        """
        # 1 - Execute pivot on watch durations, columns are renamed to be readable
        # Build the formatter with the defined pivot columns

        prepared_df = fact_watch_df
        user_formatter = dict()
        pivot_cols = {
            "AFFINITY": [x for x in AFFINITIES if
                         x not in self.BANNED_AFFINITIES + ["Séries", "Film", "Documentaires"]],
            "ACTION_DEVICE_TYPE": ["phone", "desktop", "tv", "tablet", "smart_display"],
            "CATEGORY_ID": list(range(11))
        }

        for column_name in ["ACTION_DEVICE_TYPE", "AFFINITY", "CATEGORY_ID"]:
            # The following object handles the pivot and the proper naming of the resulting columns
            user_formatter[column_name] = PivotFormatter("DURATION", column_name, pivot_cols[column_name])

        pivoted_df = dict()
        for i, column_name in enumerate(["ACTION_DEVICE_TYPE", "AFFINITY", "CATEGORY_ID"]):
            # The following object handles the pivot and the proper naming of the resulting columns
            pivoted_df[column_name] = user_formatter[column_name].pivot_and_sum(prepared_df, "USER_ID")

        # 2 - Handle the watch time on personalities
        personality_ids = list(map(str, range(1, self.PERSONALITY_TOP_K + 1)))
        f = DictFormatter("DURATION", "PERSON_ID", personality_ids)
        # Cast to string type in order to go through the BoW featurizer later on
        prog_personality_df = prog_personality_df.withColumn("PERSON_ID", F.col("PERSON_ID").cast(StringType()))
        with_person_df = fact_watch_df. \
            join(prog_personality_df, on="PROGRAM_ID"). \
            drop(prog_personality_df.PROGRAM_ID)

        pivoted_df["PERSON_ID"] = f.build_sum_dict(with_person_df, "USER_ID")

        # 3 - Join all info together
        return pivoted_df["AFFINITY"].alias("affinity"). \
            join(pivoted_df["ACTION_DEVICE_TYPE"].alias("device_type"), on="USER_ID"). \
            drop("device_type.USER_ID"). \
            join(pivoted_df["CATEGORY_ID"].alias("category"), on="USER_ID"). \
            drop("category.USER_ID"). \
            join(pivoted_df["PERSON_ID"].alias("person"), on="USER_ID"). \
            drop("person.USER_ID"). \
            join(user_df_with_age, user_df_with_age.ID == pivoted_df["AFFINITY"].USER_ID). \
            drop(user_df_with_age.ID)

    def build_program_feature_log(self, top_pers_df):
        """
        Features for program

        - basics :  "PRODUCTION_YEAR", "REF_PROGRAM_CATEGORY_ID", "REF_PROGRAM_KIND_ID", "PROGRAM_DURATION
        - Stats : allo-ciné rating, cast_nb_follows, famous_persons_in_cast
        """
        backend_df = load_snowflake_table(self.spark, self.options, "backend.program"). \
            select("ID", "PRODUCTION_YEAR", "REF_PROGRAM_CATEGORY_ID", "REF_PROGRAM_KIND_ID",
                   F.col("DURATION").alias("PROGRAM_DURATION"))
        affinity_df = load_snowflake_table(self.spark, self.options, "EXTERNAL_SOURCES.DAILY_PROG_AFFINITY"). \
            select("PROGRAM_ID", "AFFINITY"). \
            withColumn("rank", F.row_number().over(Window.partitionBy("PROGRAM_ID", "AFFINITY").orderBy("PROGRAM_ID"))). \
            where("rank = 1"). \
            drop("rank")

        rating_df = PopularRecoJob.prepare_external_rating_df(self.spark, self.options, list(range(12)))
        celeb_score_df = PopularRecoJob.prepare_famous_people_movies_df(self.spark, self.options,
                                                                        allowed_category_list=list(range(12)))

        famous_cast_df = backend_df. \
            join(top_pers_df, top_pers_df.PROGRAM_ID == backend_df.ID). \
            groupBy(top_pers_df.PROGRAM_ID). \
            agg(F.collect_set(top_pers_df.PERSON_ID).alias("FAMOUS_CAST"))

        return backend_df. \
            join(famous_cast_df, famous_cast_df.PROGRAM_ID == backend_df.ID, how="left"). \
            drop(famous_cast_df.PROGRAM_ID). \
            join(affinity_df, backend_df.ID == affinity_df.PROGRAM_ID). \
            drop(backend_df.ID). \
            join(rating_df, rating_df.PROGRAM_ID == affinity_df.PROGRAM_ID, how="left"). \
            drop(rating_df.PROGRAM_ID). \
            join(celeb_score_df, celeb_score_df.PROGRAM_ID == affinity_df.PROGRAM_ID, how="left"). \
            drop(celeb_score_df.PROGRAM_ID)

    @staticmethod
    def apply_pivot_formatters(prepared_df, pivot_cols, group_by_col="USER_ID", agg_column="DURATION", prefix=None):
        """
        Overview :

        We do something like
        for col in cols:
            df.groupBy("USER_ID", col).agg(F.sum("DURATION")).groupBy("USER_ID").agg(to_dict(col, "DURATION"))

        Why do we need an external class ?
        To properly handle the resulting names
        """
        formatters = dict()
        for column_name, possible_values in pivot_cols.items():
            # The following object handles the pivot and the proper naming of the resulting columns
            formatters[column_name] = DictFormatter(agg_column, column_name, possible_values, prefix=prefix)

        map_df = dict()
        for i, column_name in enumerate(pivot_cols.keys()):
            # The following object handles the pivot and the proper naming of the resulting columns
            map_df[column_name] = formatters[column_name].build_sum_dict(prepared_df, group_by_col)

        return map_df

    def get_top_personalities_to_pids(self, top_k=500):
        """
        The returned person id is in [1, top_k] in order to avoid the indexing step for the learning step
        """
        query = \
            f"""
            with temp as (
              select PERSON_ID, count(distinct USER_ID) as cnt 
              from backend.user_follow_person 
              where source = 'molotov' 
              group by PERSON_ID
              order by cnt desc
              limit {top_k}
            ),
             rel_person_prog as (
                select PROGRAM_ID, PERSON_ID
                from backend.rel_program_person
                UNION (
                  select program_id, person_id 
                  from backend.rel_episode_person 
                  join backend.episode 
                  on episode_id = id 
                 )
            )
              select p.PROGRAM_ID, temp.PERSON_ID as REF_PERSON_ID, row_number() OVER(ORDER BY cnt) as PERSON_ID
                from temp
                join rel_person_prog as p
                on p.PERSON_ID = temp.PERSON_ID
            """
        return load_snowflake_query_df(self.spark, self.options, query).persist()


if __name__ == "__main__":
    job = UserAndProgFeatureLogJob()
    job.launch()
