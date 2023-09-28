from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.channels.ensemble import M6_TNT_CHANNELS, TF1_TNT_CHANNELS
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, BANNED_KIND_IDS, \
    build_user_to_paying_tv_bundle, build_product_to_tv_bundle, build_user_csa_preference, \
    FREE_BUNDLE_CHANNELS, build_tvbundle_hashes, build_user_to_allowed_channels, build_equivalence_code_to_channels
from databricks_jobs.jobs.utils.popular_reco_utils import complete_recos, format_for_reco_output, keep_top_X_programs, \
    struct_schema, build_empty_user_csa_pref
from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_snowflake_query_df, \
    write_df_to_snowflake, unpivot_fact_audience, get_snowflake_options, format_tuple_for_sql


class PopularRecoJob(Job):
    N_DAYS_POPULARITY = 21
    MIN_DURATION_IN_MIN = 10
    REQUIRED_NB_H_PER_AFF = 2
    N_RECO = 25
    MAX_RECO_PER_CAT = 5
    PRIME_START = 8
    PRIME_END = 23
    MIN_POP_THRESHOLD_IN_HOURS = 1000
    AUTHORIZED_TVBUNDLES = (25, 90, 26, 60, 31, 142, 146)
    DAILY_RECO_TABLE = "RECO_USER_PROGS_THIS_WEEK_VAR"
    AB_TEST_VARIATION = 'A'

    def __init__(self, *args, **kwargs):
        super(PopularRecoJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=7)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :

        1 - Load available programs (all broadcasts in the next 7 days on usual Molotov bundles)
        2 - Stitch popularity infos on ech prog
        3 - Compute top K infos based on popularity and affinity
        4 - Broadcast recos on user with enough information
        5 - Filter to keep only N recos
            * The bundle available to the user is used to filter programs
            * a formula based on rank of programs is used (specific part of this_week)
            * several rules are used to avoid extrem results (again specific, eg: banned kinds, max nb result per affinity)
        6 - Complete with generic recommendation
            * the basic reco is based on bundles => build_best_of_bundle
            * the user 0 recommendation is used to provide something for inactive or unknown users
              => build_non_personalised_reco
        7 - Store results
        8 - Similar to user 0 other fake users are added bundle wise
        """
        self.logger.info("Launching popularity reco job")
        # 1 - Preliminary tables
        # 1.1 - Available broadcast programs
        freely_available_programs_df = self.prepare_freely_available_programs_df()
        # 1.2 - Fact_audience dfs
        fact_audience_from_today_df, flat_audience_df = self.build_fa_dfs()
        # 1.3 - Watch history, used to see if a user prefers a type of program
        user_watch_history = self.build_user_watch_history(self.spark, self.options, self.now)
        # 1.4 - Channel per user info
        prod_to_bundle_df = build_product_to_tv_bundle(self.spark, self.options)
        tvbundle_per_user = build_user_to_paying_tv_bundle(self.spark, self.options, prod_to_bundle_df,
                                                           self.now).persist()
        channels_per_user = build_user_to_allowed_channels(self.spark, self.options, tvbundle_per_user,
                                                           authorized_bundles=self.AUTHORIZED_TVBUNDLES)

        # 2 - Pre processing step : add popularity infos on programs
        prog_with_info_df = self.join_pop_info_to_programs(freely_available_programs_df).persist()

        # 3 - Build ranking per affinity
        top_k_progs_per_aff_df = keep_top_X_programs(prog_with_info_df, use_affinity=True)

        # 3.5 - Save this table by affinity and bundle split
        top_k_simple_df = keep_top_X_programs(prog_with_info_df, use_affinity=False, top_k=500)
        self.save_intermediary_results(self.spark, self.write_options, top_k_simple_df,
                                       table_name_basic="RAW_TOP_POPULAR_RECO",
                                       top_k=100)

        # 4 - Based on user affinities, find what matches
        user_reco_history = self.build_user_reco_history()
        full_reco_df = self.build_user_recos(top_k_progs_per_aff_df, flat_audience_df, user_watch_history,
                                             user_reco_history)

        # 5 - Start selecting program to reach only N_RECOS propositions
        user_csa_perf_df = build_user_csa_preference(self.spark, self.options)
        selected_reco_df = self.select_among_categories(full_reco_df, channels_per_user, user_csa_perf_df,
                                                        self.N_RECO, self.MAX_RECO_PER_CAT,
                                                        include_free=True).persist()

        # 6 - Complete with generic recos
        # 6.1 - Build a bundle based recommendation
        generic_user_recos_df, user_id_to_hash, hash_to_bundle = \
            self.build_best_of_bundle(self.spark, prog_with_info_df, channels_per_user, include_free=True)
        # 6.2 - User 0 recommendation, for new and inactive users
        general_reco_df = self.build_non_personalised_reco(prog_with_info_df, channels_per_user)
        # 6.3 - Actual reco completion
        combined_default_reco = generic_user_recos_df.union(general_reco_df)
        user_bookmarks = self.build_user_bookmarks()
        final_selected_reco_df = complete_recos(selected_reco_df, combined_default_reco, user_watch_history,
                                                user_bookmarks, user_id_to_hash, self.N_RECO)
        # 7 - final write down of the results
        # 7.1 - Existing users are added
        self.write_recos_to_snowflake(final_selected_reco_df)
        # 7.2 - User 0 is also added
        self.write_recos_to_snowflake(format_for_reco_output(general_reco_df, "recommendations"), write_mode="append")
        # 8 - Fake users are added, they correspond to default recos for some bundles
        premium_recos_df, _, hash_to_bundle = \
            self.build_best_of_bundle(self.spark, prog_with_info_df, channels_per_user, include_free=False)
        self.write_additional_bundle_recos(premium_recos_df)

    def prepare_freely_available_programs_df(self):
        """
        We get all the broadcast programs that match the timeframe [now, now + 7 days]

        A little bit of filtering is applied to avoid programs not worthy of reco
        """
        prog_affinity_df = load_snowflake_table(self.spark, self.options, "external_sources.DAILY_PROG_AFFINITY")

        def is_around_prime_time(col_name):
            hour_fn = F.udf(lambda x: x.hour, IntegerType())
            return (hour_fn(col_name) >= self.PRIME_START) & (hour_fn(col_name) <= self.PRIME_END)

        is_film_or_new_tv_show = (F.col("REF_PROGRAM_CATEGORY_ID") == 1) | \
                                 ((F.col("REF_PROGRAM_CATEGORY_ID") == 2) &
                                  ((F.col("EPISODE_NUMBER") <= 3) & (F.col("SEASON_NUMBER") == 1)))

        #  2 - List available programs for broadcast
        # Some basic filtering is done in build_broadcast_df_with_episode_info
        # Additional filtering is done to remove duplicates and keep only films and tv shows
        episode_df = build_episode_df(self.spark, self.options)
        broadcast_with_episode_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_df,
                                                                         self.now, self.now + self.delta,
                                                                         self.MIN_DURATION_IN_MIN)

        with_program_infos_df = broadcast_with_episode_df. \
            where(F.col("TVBUNDLE_ID").isin(*self.AUTHORIZED_TVBUNDLES)). \
            where(is_around_prime_time("START_AT")). \
            where(F.col("PRODUCTION_YEAR").isNull() | (F.col("PRODUCTION_YEAR") > 1970)). \
            withColumn("start_index", F.row_number().
                       over(Window.partitionBy(broadcast_with_episode_df.PROGRAM_ID).
                            orderBy("START_AT"))). \
            where(F.col("start_index") == 1). \
            where(is_film_or_new_tv_show)

        # 3 - Join with prog_affinity
        prog_affinity_df_fixed = prog_affinity_df.dropDuplicates()
        freely_available_programs_df = with_program_infos_df. \
            join(prog_affinity_df_fixed,
                 prog_affinity_df_fixed.PROGRAM_ID == with_program_infos_df.PROGRAM_ID, "left"). \
            drop(prog_affinity_df_fixed.PROGRAM_ID). \
            select("PROGRAM_ID", "REF_PROGRAM_KIND_ID", "REF_PROGRAM_CATEGORY_ID", "START_AT",
                   "AFFINITY", "CHANNEL_ID", "EPISODE_ID", "REF_CSA_ID")

        return freely_available_programs_df

    def join_pop_info_to_programs(self, freely_available_prgrams_df):
        """
        For this steps, we extract the four kind of popularites :
        - total_distinct_bookmarks
        - total_celeb_points
        - total_watch_time
        - external_rating

        1 - retrieve pop infos
        2 - join them to the programs

        :param freely_available_prgrams_df:
        :return:
        """

        # 1 - retrieve pop infos
        bookmark_df = self.prepare_total_distinct_bookmarks_df()
        rating_df = self.prepare_external_rating_df(self.spark, self.options)
        celeb_df = self.prepare_famous_people_movies_df(self.spark, self.options)
        like_df = self.compute_user_likes(self.spark, self.options, self.now - self.delta)
        rebroadcast_df = self.prepare_rebroadcast()

        # 2- join them
        def join_additional_information_on_future_programs(future_prog_df, col_to_df_dict, default_vals):
            """
            future_prog_df: main pyspark dataframe on which additional info is joined
            col_to_df_dict: dict(col_name, dataframe) we only add the col_name for each dataframe in the dict
            """

            for col_name, additional_df in col_to_df_dict.items():
                future_prog_df = future_prog_df.join(additional_df,
                                                     additional_df.PROGRAM_ID == future_prog_df.PROGRAM_ID, "left"). \
                    drop(additional_df.PROGRAM_ID). \
                    fillna(default_vals[col_name], subset=[col_name]). \
                    select(*future_prog_df.columns, F.col(col_name))
            return future_prog_df

        future_program_with_infos = join_additional_information_on_future_programs(
            freely_available_prgrams_df,
            {"total_distinct_bookmarks": bookmark_df, "total_celeb_points": celeb_df,
             "external_rating": rating_df,
             "nb_likes": like_df, "total_rebroadcast": rebroadcast_df},
            {"total_distinct_bookmarks": 0, "nb_likes": 0,
             "total_celeb_points": 0, "external_rating": 0, "total_rebroadcast": 0})

        return future_program_with_infos

    def build_user_recos(self, future_program_with_top_infos, flat_audience_df, user_history_watch_df,
                         user_reco_history):
        """
        In this step, we use user affinity and their watch history to get program that may interest the user

        Steps are:
        1 - Inner join the reco with the user affinities
        2 - Affinity preference is joined, it tells what the user has most watched

        :param future_program_with_top_infos: DataFrame
        :return: DataFrame
        """
        # 1 - Join user with top prog matching their affinity
        user_top_programs_df = future_program_with_top_infos. \
            join(flat_audience_df, future_program_with_top_infos.AFFINITY == flat_audience_df.category). \
            select("USER_ID", "PROGRAM_ID", "AFFINITY", "best_rank", "reco_origin", "total_rebroadcast",
                   "START_AT", "CHANNEL_ID", "EPISODE_ID", "REF_CSA_ID")

        # 2 - Join the affinity preference information
        user_affinity_preference_df = self.compute_user_affinity_preference(self.spark, self.options,
                                                                            self.REQUIRED_NB_H_PER_AFF,
                                                                            user_history_watch_df)
        user_top_programs_df = user_top_programs_df. \
            join(user_affinity_preference_df,
                 (user_affinity_preference_df.USER_ID == user_top_programs_df.USER_ID) &
                 (user_affinity_preference_df.AFFINITY == user_top_programs_df.AFFINITY)). \
            drop(user_affinity_preference_df.USER_ID). \
            drop(user_affinity_preference_df.AFFINITY). \
            select(*user_top_programs_df.columns,
                   user_affinity_preference_df.USER_AFFINITY_RANK). \
            withColumn("start_rank", F.percent_rank().over(Window.partitionBy("USER_ID").orderBy("START_AT")))

        user_top_programs_df = user_top_programs_df. \
            join(user_reco_history, (user_reco_history.USER_ID == user_top_programs_df.USER_ID) &
                 (user_reco_history.PROGRAM_ID == user_top_programs_df.PROGRAM_ID), "left"). \
            drop(user_reco_history.USER_ID). \
            drop(user_reco_history.PROGRAM_ID). \
            fillna(value=0, subset=["visibility"]). \
            withColumn("total_rebroadcast", F.col("total_rebroadcast") + F.lit(5.0) * F.col("visibility"))

        return user_top_programs_df

    @classmethod
    def select_among_categories(cls, full_reco_df, user_to_bundle, user_csa_pref_df,
                                n_recos: int, max_reco_per_cat: int, include_free: bool):
        """
        We reduce the number of proposition per user to n_recos
        To do so, we select them by order of 'best_rank' (given by popularity type in each 'reco_origin')
        include_free allows to force selection among the paying bundles only

        We run the following steps :
        1 - User to bundle info to remove non eligible programs for the user + removed banned kinds
        2 - Ranking function computation
        3 - Keep top X and only Y per affinity
        4 - Duplicate removal

        full_reco_df format :
            format : "USER_ID", "PROGRAM_ID", "AFFINITY", "reco_origin", "USER_AFFINITY_RANK", "best_rank", "start_rank",
                     "CHANNEL_ID", "EPISODE_ID", "REF_CSA_ID"

        out columns :
            format : "USER_ID", "PROGRAM_ID", "row_number", "ranking", "AFFINITY", "reco_origin"
        """
        user_to_bundle = user_to_bundle. \
            alias("bundle"). \
            withColumn("HAS_OPTION", F.lit(True))
        # 1.a - We join with TV Bundle information to remove program not eligible for the user
        condition_is_free_or_m6_or_tf1 = (F.lit(include_free) & (
            (full_reco_df.CHANNEL_ID.isin(FREE_BUNDLE_CHANNELS)) |
            (full_reco_df.CHANNEL_ID.isin(*M6_TNT_CHANNELS.chnl_ids())) |
            (full_reco_df.CHANNEL_ID.isin(*TF1_TNT_CHANNELS.chnl_ids()))))
        selected_programs_df = full_reco_df.alias("default"). \
            join(user_to_bundle, (F.col("bundle.USER_ID") == full_reco_df.USER_ID) & (
                F.col("bundle.CHANNEL_ID") == full_reco_df.CHANNEL_ID), "left"). \
            filter(condition_is_free_or_m6_or_tf1 | user_to_bundle.HAS_OPTION.isNotNull())

        # TODO : readd "1.b - we keep progs with csa_id < to the max allowed per user"

        # 1.5 - Removed unwanted categories
        removed_categories = ["Information & Politique", "Divertissement", "Investigation & Reportage",
                              "Talk shows", "Enfants"]
        selected_programs_df = selected_programs_df. \
            where(~F.col("AFFINITY").isin(*removed_categories))

        # 2 - We compute a ranking function to get the program we should keep
        # This is in fact a scoring function based on different criteria : popularity criterion,
        # affinity preference of the user, how soon is the program
        selected_programs_df = selected_programs_df. \
            withColumn(
                "ranking",
                F.col("USER_AFFINITY_RANK") + F.col("best_rank") + F.lit(7.0) * F.col("start_rank") +
                F.col("total_rebroadcast")
            )

        # 3 and 4 - We need to remove duplicates (prog_number=1) if any and create one row per user x affinity
        # Keep at most 5 recos per affinity (affinity_row_number) at at most n_recos in total (max_reco_row_number)
        selected_programs_df = selected_programs_df. \
            withColumn("prog_number", F.row_number().
                       over(Window.partitionBy("PROGRAM_ID", "default.USER_ID").orderBy("ranking"))). \
            filter(F.col("prog_number") == 1). \
            withColumn("affinity_row_number", F.row_number().over(
                Window.partitionBy("default.USER_ID", "AFFINITY").orderBy("ranking"))). \
            where(F.col("affinity_row_number") <= max_reco_per_cat). \
            drop("affinity_row_number"). \
            withColumn("row_number", F.row_number().over(Window.partitionBy("default.USER_ID").orderBy("ranking"))). \
            filter(F.col("row_number") <= F.lit(n_recos))

        return selected_programs_df. \
            select("default.USER_ID", "PROGRAM_ID", "row_number", "ranking", "AFFINITY",
                   "default.CHANNEL_ID", "reco_origin", "EPISODE_ID"). \
            withColumn("PROGRAM_ID", F.col("PROGRAM_ID").cast(IntegerType())). \
            withColumn("CHANNEL_ID", F.col("CHANNEL_ID").cast(IntegerType())). \
            withColumn("EPISODE_ID", F.col("EPISODE_ID").cast(IntegerType())). \
            withColumn("rating", 1.0 / F.col("ranking"))

    def build_non_personalised_reco(self, prog_with_info_df, channels_per_user, default_id=0):
        top_k_progs_df = keep_top_X_programs(
            prog_with_info_df, use_affinity=False,
            scoring_methods=("external_rating", "total_celeb_points", "nb_likes")
        )
        global_reco_df = top_k_progs_df. \
            withColumn("USER_ID", F.lit(default_id)). \
            withColumn("USER_AFFINITY_RANK", F.lit(1)). \
            withColumn("start_rank", F.lit(1))

        user_csa_perf_df = build_empty_user_csa_pref(self.spark)
        selected_global_reco_df = self.select_among_categories(global_reco_df, channels_per_user, user_csa_perf_df,
                                                               self.N_RECO, self.N_RECO, True)
        return selected_global_reco_df

    @classmethod
    def build_best_of_bundle(cls, spark, prog_with_info_df, channels_per_user, include_free=True):
        """
        We need to build one set of recommendation per bundle combination
        - Keep best programs
        - Compute all bundle combinations and give the mapping user_id_to_bundle_hash
        - Hack : consider hash_id as user_id and assign them bundles rights
        - select_among_categories will build for us the right recommendation based on bundle rights

        Example :
        Step 1: as usual
        Step 2: user_bundles = {OCS, M+} ==> hash = 12345

        we build a dataframe based on all known bundles_hashes
        hash    tvbundle
        12345   30
        12345   90

        Step 3:
        hash_to_bundle => hacked_user_id_to_tvbundle

        Step 4:
        select_among_categories(best_of_df, hacked_user_id_to_tvbundle)

        :param prog_with_info_df:
        :param user_to_tvbundle_df:
        :return:
        """
        # 1 - Select best programs
        top_k_progs_df = keep_top_X_programs(
            prog_with_info_df, use_affinity=False,
            scoring_methods=("external_rating", "total_celeb_points", "nb_likes"),
            top_k=500
        )
        # Need for the prog filtering later
        user_csa_perf_df = build_empty_user_csa_pref(spark)

        # 2 - Compute all bundle combinations
        # 25=Free, 90=MLT+, 50=26=Extended, 60=36=OCS, 77=AdultSwim, 31=Ciné+, 98=StarZ
        hash_to_bundle, user_id_to_hash = build_tvbundle_hashes(channels_per_user)

        # 4 - Filter the proper program id based on tvbundle info
        # 4.1 - Prepare the input of select_among_categories
        best_of_df = hash_to_bundle. \
            crossJoin(F.broadcast(top_k_progs_df)). \
            withColumn("USER_AFFINITY_RANK", F.lit(4)). \
            withColumn("start_rank", F.lit(3.5)). \
            select("USER_ID", "PROGRAM_ID", "AFFINITY", "reco_origin", "USER_AFFINITY_RANK", "EPISODE_ID",
                   "best_rank", "start_rank", "total_rebroadcast", top_k_progs_df.CHANNEL_ID, "REF_CSA_ID")
        # 4.2 - inputs are best_of_df: fake_user_id -> program_list, hash_to_bundle: fake_user_id -> channel_ids
        selected_reco_df = cls.select_among_categories(best_of_df, hash_to_bundle, user_csa_perf_df,
                                                       cls.N_RECO, cls.N_RECO, include_free)
        # We need to return user_id_to_hash to do the mapping user_id -> bundle_hash later on.
        return selected_reco_df, user_id_to_hash, hash_to_bundle

    #########################
    #    Helper functions
    #########################

    @staticmethod
    def save_intermediary_results(spark, options, top_k_df,
                                  table_name_aff="RAW_AFFINITY_POPULAR_RECO",
                                  table_name_bundle="RAW_BUNDLE_POPULAR_RECO",
                                  table_name_basic=None,
                                  top_k=50):
        """
        We need to :

        - remove duplicate program scheduled at different hours
        - drop the programs with no info (best_rank = 200)
        - find the top 50 per affinity or bundle
        """
        equ_code_to_channels = build_equivalence_code_to_channels(spark, options)
        top_k_df = top_k_df.\
            join(equ_code_to_channels, equ_code_to_channels.CHANNEL_ID == top_k_df.CHANNEL_ID).\
            drop(equ_code_to_channels.CHANNEL_ID)

        def keep_top_k_over_col(df, col_name, topk=top_k):
            return df. \
                withColumn("prog_number", F.row_number().
                           over(Window.partitionBy("PROGRAM_ID", col_name).orderBy("best_rank"))). \
                filter(F.col("prog_number") == 1). \
                drop("prog_number"). \
                where("best_rank < 200"). \
                withColumn("rank", F.row_number().over(Window.partitionBy(col_name).orderBy("best_rank"))). \
                where(f"rank <= {topk}"). \
                drop("best_rank")

        results_per_bundle = keep_top_k_over_col(top_k_df, "EQUIVALENCE_CODE")
        resuts_per_affinity = keep_top_k_over_col(top_k_df, "AFFINITY")
        top_k_results = keep_top_k_over_col(
            top_k_df.where(F.col("EQUIVALENCE_CODE").isin(*["FREE", "OPTION_100H", "EXTENDED", "OCS", "CINE_PLUS"])),
            F.lit(0),
            topk=10000
        )  # Trick to keep the same func for top k

        if table_name_aff:
            write_df_to_snowflake(resuts_per_affinity, options, table_name_aff, "overwrite")
        if table_name_bundle:
            write_df_to_snowflake(results_per_bundle, options, table_name_bundle, "overwrite")
        if table_name_basic:
            write_df_to_snowflake(top_k_results, options, table_name_basic, "overwrite")

        return results_per_bundle, resuts_per_affinity

    def build_user_bookmarks(self):
        # Select available bookmarks
        actual_bookmark_df = load_snowflake_table(self.spark, self.options, "backend.actual_recording"). \
            where(F.col("DELETED_AT").isNull()).\
            select("USER_ID", "PROGRAM_ID")
        # Select scheduled bookmarks available next days
        scheduled_bookmark_df = load_snowflake_table(self.spark, self.options, "backend.scheduled_recording"). \
            where(F.col("DELETED_AT").isNull()).\
            select("USER_ID", "PROGRAM_ID")
        user_bookmarks_df = actual_bookmark_df.union(scheduled_bookmark_df)
        return user_bookmarks_df

    @staticmethod
    def build_user_watch_history(spark, options, now):
        # 1 - Add user watch history
        user_watch_lookback = timedelta(days=90)
        min_watch_time_in_mins = 10
        user_history_watch_df = load_snowflake_table(spark, options, "dw.fact_watch"). \
            where(F.col("REAL_START_AT") >= now - user_watch_lookback). \
            groupby("USER_ID", "PROGRAM_ID"). \
            agg(F.sum("DURATION").alias("total_duration")). \
            where(F.col("total_duration") > min_watch_time_in_mins * 60). \
            persist()
        return user_history_watch_df

    def build_user_reco_history(self):
        """
            In this step, we use reco_user_progs_this_week_latest and fact_page to create a penalization score for program too
            present in the rail

            Steps are:
            1 - Parse user reco history to keep program_id, rank and date of recommendation
            2 - Select different days where at least one home page is viewed
            3 - We consider that home_page viewed = this_week viewed to compute visibility score:
                - occurence = nb of different days viewing a program on this_week rail
                - visibility = sum(1/program_rank) over 30 lasst days if occurence > 7

            :return: DataFrame
            """
        user_reco_lookback = timedelta(days=30)
        reco_history = load_snowflake_table(self.spark, self.options, f"ml.{self.DAILY_RECO_TABLE}"). \
            where(F.col("UPDATE_DATE") >= self.now - user_reco_lookback). \
            where(F.col("VARIATIONS") == self.AB_TEST_VARIATION)

        page_history = load_snowflake_table(self.spark, self.options, "dw.fact_page"). \
            where(F.col("EVENT_DATE") >= self.now - user_reco_lookback). \
            select("USER_ID", "EVENT_DATE", "PAGE_NAME")

        user_history_reco_df = reco_history. \
            withColumn("parsed_json", F.from_json("RECOMMENDATIONS", ArrayType(struct_schema))).\
            select("USER_ID", "UPDATE_DATE", F.explode(F.col("parsed_json")).alias("parsed_json")).\
            select("USER_ID", "UPDATE_DATE", F.col("parsed_json.PROGRAM_ID").alias("PROGRAM_ID"),
                   F.col("parsed_json.ranking").alias("ranking")). \
            withColumn("rail_rank", F.row_number().
                       over(Window.partitionBy("USER_ID", "UPDATE_DATE").orderBy("ranking")))

        user_page_with_history_reco_df = user_history_reco_df. \
            join(page_history, (page_history.EVENT_DATE == user_history_reco_df.UPDATE_DATE) &
                 (page_history.USER_ID == user_history_reco_df.USER_ID)). \
            drop(page_history.USER_ID). \
            where(F.col("PAGE_NAME") == "home"). \
            select("USER_ID", "UPDATE_DATE", "PROGRAM_ID", "rail_rank", "PAGE_NAME"). \
            distinct()

        # Each time a program appears in the recommendation a score of 1/rank is associated
        # The occurence represents the number of home page viewed x program is in the rail
        # When the occurence reach 8 for a program he receives the visibility score (sum of score for each occurence)
        user_prog_visibility_df = user_page_with_history_reco_df. \
            withColumn("score", 1 / F.col("rail_rank")). \
            groupby("USER_ID", "PROGRAM_ID"). \
            agg(F.sum("score").alias("visibility"), F.count("PAGE_NAME").alias("occurence")). \
            where(F.col("occurence") > 7). \
            select("USER_ID", "PROGRAM_ID", "visibility")

        return user_prog_visibility_df

    @staticmethod
    def prepare_famous_people_movies_df(spark, options, tv_bundles=(25, 90, 26, 31, 60, 126),
                                        allowed_category_list=(1, -1)):
        divider = 10
        max_celebs_contributing_to_movie_score = 20
        query = \
            f"""
            with celeb_score as --used to compute the score of the most liked famous person
            (
              select
                count(*) as MAX_CELEB_SCORE
              from
                backend.person
                left outer join backend.user_follow_person on PERSON_ID = ID
              where
                source = 'molotov'
              group by
                PERSON_ID,
                FIRST_NAME,
                LAST_NAME
              order by
                MAX_CELEB_SCORE desc
              limit
                1
            ), 
            like_count as -- sum of likes for each famous person
            (
              select
                PERSON_ID,
                LEAST(
                  coalesce(count(*), 0),
                  (
                    select
                      MAX_CELEB_SCORE / {divider}
                    from
                      celeb_score
                  ) :: int
                ) as CNT_TRESHOLD,
                count(*) as cnt_old --max_celeb_score/10 -> diviseur = 10
              from
                backend.person
                left outer join backend.user_follow_person on PERSON_ID = ID
              where
                source = 'molotov'
              group by
                PERSON_ID
            ),
            rel_person_prog as --which famous person participate on whiche program
            (
              select
                PROGRAM_ID,
                PERSON_ID
              from
                backend.rel_program_person
              UNION
              (
                select
                  program_id,
                  person_id
                from
                  backend.rel_episode_person
                  join backend.episode on episode_id = id
              )
            ),
            pers_with_like_by_program as (
              select
                rel.PROGRAM_ID,
                lc.PERSON_ID,
                lc.CNT_TRESHOLD,
                row_number() over (
                  partition by rel.PROGRAM_ID
                  order by
                    lc.CNT_TRESHOLD desc
                ) as rank
              from
                like_count as lc
                join rel_person_prog as rel on lc.PERSON_ID = rel.PERSON_ID
              order by
                PROGRAM_ID,
                CNT_TRESHOLD desc
            ),
            like_count_by_prog as (
              select
                prog.ID as PROGRAM_ID,
                SUM(CNT_TRESHOLD) as TOTAL_CELEB_POINTS
              from
                pers_with_like_by_program as pwlbp
                join BACKEND.PROGRAM as prog on pwlbp.PROGRAM_ID = prog.ID
              where
                rank < {max_celebs_contributing_to_movie_score + 1}
              group by
                prog.ID,
                prog.TITLE
              order by
                TOTAL_CELEB_POINTS desc
            ),
            available_channels as (
              select
                CHANNEL_ID
              from
                backend.rel_tvbundle_channel
              where
                TVBUNDLE_ID in {format_tuple_for_sql(tv_bundles)}
            )
            select
              distinct lcbp.PROGRAM_ID,
              lcbp.TOTAL_CELEB_POINTS
            from
              like_count_by_prog as lcbp
              inner join backend.program as bp on bp.ID = lcbp.PROGRAM_ID
              inner join backend.broadcast as bb on bb.PROGRAM_ID = bp.ID
              inner join available_channels as ac on bb.CHANNEL_ID = ac.CHANNEL_ID
            where
              bp.REF_PROGRAM_CATEGORY_ID in {format_tuple_for_sql(allowed_category_list)}
              and bb.START_AT >= current_date
              and bb.START_AT < current_date + 7
              and TOTAL_CELEB_POINTS > 5000
            """
        return load_snowflake_query_df(spark, options, query)

    def prepare_total_distinct_bookmarks_df(self):
        """
        A kind of popularity : when users with bookmark capability want to save a show.
        We use distinct (PROG_ID, USER_ID)
        :return: DataFrame
        """
        bookmark_lookback = timedelta(days=14)
        bookmark_df = load_snowflake_table(self.spark, self.options, "backend.SCHEDULED_RECORDING"). \
            where(F.col("SCHEDULED_AT") >= self.now - bookmark_lookback). \
            groupby("PROGRAM_ID"). \
            agg(F.countDistinct("USER_ID").alias("total_distinct_bookmarks")). \
            where(F.col("total_distinct_bookmarks") >= 200)
        return bookmark_df

    @staticmethod
    def prepare_external_rating_df(spark, options, ref_category_list=(1, 2), min_rating=2.5):
        # Based on external rating, we use only the allo ciné user ratings as closer to the crowd
        program_df = load_snowflake_table(spark, options, "backend.program")
        rating_df = load_snowflake_table(spark, options, "backend.program_rating"). \
            where(F.col("PROGRAM_RATING_TYPE_ID") == 2). \
            select("PROGRAM_ID", "RATING"). \
            join(program_df, program_df.ID == F.col("PROGRAM_ID")). \
            where(F.col("REF_PROGRAM_CATEGORY_ID").isin(*ref_category_list)). \
            select("PROGRAM_ID", "RATING"). \
            withColumnRenamed("RATING", "external_rating"). \
            where(F.col("external_rating") > min_rating)
        return rating_df

    def prepare_recent_pop_df(self):
        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch")
        # Recent popularity as total time spent
        recent_pop_lookback = timedelta(days=60)
        recent_popularity_df = fact_watch_df. \
            join(program_df, program_df.ID == fact_watch_df.PROGRAM_ID). \
            drop(program_df.DURATION). \
            where(F.col("REAL_START_AT") > self.now - recent_pop_lookback). \
            where(F.col("ref_program_category_id") == 2). \
            groupby("PROGRAM_ID"). \
            agg(F.round(F.sum("DURATION") / F.lit(3600), 0).alias("total_watch_duration")). \
            where(F.col("total_watch_duration") > self.MIN_POP_THRESHOLD_IN_HOURS)
        return recent_popularity_df

    def prepare_rebroadcast(self):
        broadcast_df = load_snowflake_table(self.spark, self.options, "backend.broadcast")
        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        broadcast_lookback = timedelta(days=30)
        rebroadcast_df = broadcast_df. \
            join(program_df, program_df.ID == broadcast_df.PROGRAM_ID). \
            drop(program_df.DURATION). \
            drop(program_df.ID). \
            where((F.col("START_AT") >= self.now - broadcast_lookback) & (F.col("START_AT") < self.now)). \
            where((~program_df.REF_PROGRAM_KIND_ID.isin(*BANNED_KIND_IDS))). \
            where(program_df.REF_PROGRAM_CATEGORY_ID == 1). \
            groupby("PROGRAM_ID"). \
            agg(F.count("PROGRAM_ID").alias("total_rebroadcast"))

        return rebroadcast_df

    @staticmethod
    def compute_user_likes(spark, options, start, min_nb_likes=100, category_filter=[1]):
        program_df = load_snowflake_table(spark, options, "backend.program")
        like_df = load_snowflake_table(spark, options, "backend.user_channel_item")

        like_df = like_df. \
            where(F.col("CREATED_AT") >= start). \
            groupBy("PROGRAM_ID"). \
            agg(F.count("USER_ID").alias("nb_likes")). \
            where(F.col("nb_likes") > min_nb_likes)
        return like_df. \
            join(program_df, program_df.ID == like_df.PROGRAM_ID). \
            where(F.col("REF_PROGRAM_CATEGORY_ID").isin(*category_filter)). \
            select("PROGRAM_ID", "nb_likes")

    @staticmethod
    def compute_user_affinity_preference(spark, options, required_nb_h_per_aff, user_history_watch_df):
        """
        Compute the preferred affinities of a user
        If the user has 20 affinities, we would like to know which ones are
        the most important for the future recommendation

        :param user_history_watch_df: DataFrame
        :return: DataFrame
        """
        # 1 - Get user watch time per affinity
        daily_prog_affinity_df = load_snowflake_table(spark, options,
                                                      "external_sources.DAILY_PROG_AFFINITY")
        user_affinity_watch_df = user_history_watch_df. \
            join(daily_prog_affinity_df,
                 user_history_watch_df.PROGRAM_ID == daily_prog_affinity_df.PROGRAM_ID). \
            drop(daily_prog_affinity_df.PROGRAM_ID). \
            groupBy("USER_ID", "AFFINITY"). \
            agg(F.sum(user_history_watch_df.total_duration).alias("time_on_affinity"))

        # 2 - Filter and reorder
        removed_categories = ["Information & Politique", "Divertissement", "Investigation & Reportage",
                              "Talk shows", "Jeux"]
        user_aff_with_rank_df = user_affinity_watch_df. \
            where(~F.col("AFFINITY").isin(*removed_categories)). \
            withColumn("USER_AFFINITY_RANK",
                       F.rank().over(Window.partitionBy("USER_ID").orderBy(F.desc("time_on_affinity")))). \
            where(F.col("USER_AFFINITY_RANK") <= 7). \
            where((F.col("time_on_affinity") > required_nb_h_per_aff * 3600))

        return user_aff_with_rank_df

    def build_fa_dfs(self):
        # 1.1 - Load fact audience from today
        fact_audience_from_today_df = load_snowflake_table(self.spark, self.options, "dw.fact_audience")
        # 1.2 - We need to have a format like (USER_ID, AFFINITY) which is like unpivoting the table
        non_category_columns = {"USER_ID", "RFM7_CLUSTER", "RFM28_CLUSTER"}
        category_columns = set(fact_audience_from_today_df.columns).difference(non_category_columns)
        flat_audience_df = \
            unpivot_fact_audience(fact_audience_from_today_df, category_columns, non_category_columns). \
            select("USER_ID", "category")
        return fact_audience_from_today_df, flat_audience_df

    def write_recos_to_snowflake(self, user_top_programs_df, table_name=DAILY_RECO_TABLE,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        final_user_recos_df = user_top_programs_df. \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)

    def write_additional_bundle_recos(self, bundle_recos):
        """
        The goal of this function is to write additional fake user recos
        The idea would be : user -1 = default reco for Molotov+ etc

        25=Free, 90=MLT+, 50=26=Extended, 60=36=OCS, 77=AdultSwim, 31=Ciné+, 98=StarZ

        bundle_recos: DataFrame[HASH_ID, PROGRAM_ID, other_cols] 1 to many relationship
        hash_to_bundle: DataFrame[HASH_ID, CHANNEL_ID] 1 to many relationship
        """
        equ_code_to_channels = build_equivalence_code_to_channels(self.spark, self.options)
        bundle_to_channels_df = equ_code_to_channels. \
            groupBy("EQUIVALENCE_CODE"). \
            agg(F.sort_array(F.collect_set("CHANNEL_ID")).alias("available_channels")). \
            withColumn("HASH_ID", F.hash("available_channels")).\
            select("EQUIVALENCE_CODE", "HASH_ID")

        # Assign fake user_ids for collection of bundles
        user_id_rewritting = F.when(F.col("EQUIVALENCE_CODE") == F.lit("OPTION_100H"), -1). \
            when(F.col("EQUIVALENCE_CODE") == F.lit("EXTENDED"), -2). \
            when(F.col("EQUIVALENCE_CODE") == F.lit("OCS"), -3). \
            when(F.col("EQUIVALENCE_CODE") == F.lit("CINE_PLUS"), -4). \
            when(F.col("EQUIVALENCE_CODE") == F.lit("CINE_SERIES"), -5). \
            otherwise(F.lit(None))

        df = bundle_recos. \
            join(bundle_to_channels_df, bundle_recos.USER_ID == bundle_to_channels_df.HASH_ID). \
            withColumn("USER_ID", user_id_rewritting). \
            dropna()

        self.write_recos_to_snowflake(format_for_reco_output(df, "recommendations"), write_mode="append")
        return df


if __name__ == "__main__":
    job = PopularRecoJob()
    job.launch()
