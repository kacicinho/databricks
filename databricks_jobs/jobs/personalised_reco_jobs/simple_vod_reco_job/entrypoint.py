from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.db_common import build_episode_df, build_vod_df_with_episode_infos, BANNED_KIND_IDS, \
    FREE_BUNDLE_CHANNELS, build_tvbundle_hashes, build_user_to_paying_tv_bundle, build_product_to_tv_bundle, \
    build_user_to_allowed_channels
from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from databricks_jobs.jobs.utils.popular_reco_utils import format_for_reco_output, certainty_fn, keep_top_X_programs
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake, \
    create_spark_df_from_data


class VodRecoJob(Job):
    MAX_RECOS = 25
    MAX_RECO_PER_ORIGIN = 8
    DAILY_RECO_TABLE = "RECO_USER_PROGS_THIS_DAY_LATEST"

    def __init__(self, *args, **kwargs):
        super(VodRecoJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=3)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :

        """
        self.logger.info("Launching vod reco job")

        # 1 - Preliminary tables
        vod_df = self.build_available_vod_df()

        # 2 - Pre processing step : add popularity infos on programs
        prog_with_info_df = self.join_pop_info_to_programs(vod_df). \
            persist()

        # 3 - Build ranking among popularity categories
        top_k_progs_per_aff_df = keep_top_X_programs(
            prog_with_info_df, use_affinity=False,
            scoring_methods=("distinct_replay_watch", "avg_replay_duration", "external_rating",
                             "nb_likes", "total_celeb_points")
        )
        top_k_progs_per_aff_df = top_k_progs_per_aff_df. \
            drop(top_k_progs_per_aff_df.START_AT). \
            drop(top_k_progs_per_aff_df.END_AT)
        PopularRecoJob.save_intermediary_results(self.spark, self.write_options, top_k_progs_per_aff_df,
                                                 table_name_aff="RAW_AFFINITY_VOD_RECO",
                                                 table_name_bundle="RAW_BUNDLE_VOD_RECO")

        reco_df = self.select_recos(top_k_progs_per_aff_df)

        # 4.1 Write recos to daily table
        self.write_recos_to_snowflake(reco_df, self.DAILY_RECO_TABLE)

    def build_available_vod_df(self):
        """
        TVBUNDLE for vod :
        25 - Free bundle
        90 - Molotov+
        26 - Molotov Extended
        """
        prog_affinity_df = load_snowflake_table(self.spark, self.options, "external_sources.DAILY_PROG_AFFINITY")
        episode_info_df = build_episode_df(self.spark, self.options)
        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df,
                                                 self.now, self.delta, min_duration_in_mins=15,
                                                 video_type=("REPLAY",))

        vod_df = vod_df. \
            join(prog_affinity_df, prog_affinity_df.PROGRAM_ID == vod_df.PROGRAM_ID, "left"). \
            drop(prog_affinity_df.PROGRAM_ID). \
            where("TVBUNDLE_ID in (25, 90, 26)"). \
            drop("TVBUNDLE_ID"). \
            where(F.col("PRODUCTION_YEAR").isNull() | (F.col("PRODUCTION_YEAR") > 1970)). \
            where(~F.col("REF_PROGRAM_KIND_ID").isin(BANNED_KIND_IDS)). \
            withColumn("prog_rank", F.row_number().over(
                Window.partitionBy("PROGRAM_ID", "CHANNEL_ID").orderBy("PROGRAM_ID"))). \
            where("prog_rank = 1"). \
            drop("prog_rank")
        return vod_df

    def join_pop_info_to_programs(self, freely_available_prgrams_df):
        """
        For this steps, we extract the four kind of popularites :
        - total_distinct_bookmarks
        - total_celeb_points
        - replay_watch_df
        - external_rating

        1 - retrieve pop infos
        2 - join them to the programs

        :param freely_available_prgrams_df:
        :return:
        """

        # 1 - retrieve pop infos
        distinct_watch_df = self.prepare_distinct_replay_watch_df()
        replay_watch_df = self.prepare_recent_replay_pop_df()
        external_rating_df = PopularRecoJob.prepare_external_rating_df(self.spark, self.options, min_rating=3.0)
        nb_likes = PopularRecoJob.compute_user_likes(self.spark, self.options, self.now - timedelta(days=30),
                                                     min_nb_likes=400,
                                                     category_filter=[1, 2, 5, 8])
        famous_cast_df = PopularRecoJob.prepare_famous_people_movies_df(self.spark, self.options, tuple(range(100)))

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
            {"distinct_replay_watch": distinct_watch_df, "avg_replay_duration": replay_watch_df,
             "external_rating": external_rating_df, "nb_likes": nb_likes, "total_celeb_points": famous_cast_df},
            {"distinct_replay_watch": 0, "avg_replay_duration": 0, "external_rating": 0, "nb_likes": 0,
             "total_celeb_points": 0})

        return future_program_with_infos

    ##############
    #   Helpers
    ##############

    def prepare_distinct_replay_watch_df(self):
        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch")
        # Recent popularity as total time spent
        recent_pop_lookback = timedelta(days=7)
        recent_popularity_df = fact_watch_df. \
            join(program_df, program_df.ID == fact_watch_df.PROGRAM_ID). \
            drop(program_df.DURATION). \
            where(fact_watch_df.DURATION > 30). \
            where(F.col("ASSET_TYPE").isin("replay", "vod")). \
            where(F.col("REAL_START_AT") > self.now - recent_pop_lookback). \
            groupby("PROGRAM_ID"). \
            agg(F.countDistinct("USER_ID").alias("distinct_replay_watch"))
        return recent_popularity_df

    def prepare_recent_replay_pop_df(self, lambda_hours=150):
        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch")
        # Recent popularity as total time spent
        recent_pop_lookback = timedelta(days=2)
        recent_popularity_df = fact_watch_df. \
            join(program_df, program_df.ID == fact_watch_df.PROGRAM_ID). \
            drop(program_df.DURATION). \
            where(F.col("REAL_START_AT") > self.now - recent_pop_lookback). \
            where(F.col("ASSET_TYPE").isin("replay", "vod")). \
            groupby("PROGRAM_ID"). \
            agg((certainty_fn("DURATION", lambda_hours) * F.round(F.avg("DURATION") / F.lit(3600))).
                alias("avg_replay_duration"))
        return recent_popularity_df

    def select_recos(self, vod_df):
        # 1 - Build bundle information for users
        prod_to_bundle_df = build_product_to_tv_bundle(self.spark, self.options)
        user_to_tvbundle_df = build_user_to_paying_tv_bundle(self.spark, self.options, prod_to_bundle_df,
                                                             self.now). \
            union(create_spark_df_from_data(self.spark, {"USER_ID": [0], "TVBUNDLE_ID": 25}))

        channels_per_user = build_user_to_allowed_channels(self.spark, self.options, user_to_tvbundle_df)
        hash_to_bundle_df, user_id_to_hash_df = build_tvbundle_hashes(channels_per_user)
        all_users_df = user_to_tvbundle_df.select("USER_ID").distinct()

        # 1.2 - Remove already seen films
        prog_df = load_snowflake_table(self.spark, self.options, "backend.program")
        user_watch_df = PopularRecoJob.build_user_watch_history(self.spark, self.options, self.now)
        user_watch_df = user_watch_df. \
            join(prog_df, prog_df.ID == user_watch_df.PROGRAM_ID). \
            where("REF_PROGRAM_CATEGORY_ID = 1")

        # 2 - Cross join top_k_progs x possible_bundles
        reco_df = vod_df. \
            where("best_rank < 200"). \
            crossJoin(hash_to_bundle_df). \
            where(vod_df.CHANNEL_ID.isin(*FREE_BUNDLE_CHANNELS) |
                  (hash_to_bundle_df.CHANNEL_ID == vod_df.CHANNEL_ID)). \
            drop(hash_to_bundle_df.CHANNEL_ID). \
            withColumn("user_prog_rank", F.row_number().over(
                Window.partitionBy("USER_ID", "PROGRAM_ID").orderBy("best_rank"))). \
            where("user_prog_rank = 1"). \
            drop("user_prog_rank"). \
            withColumn("score", F.col("best_rank") + F.when(F.col("reco_origin") == "distinct_replay_watch",
                                                            F.lit(0)).otherwise(F.lit(4))). \
            withColumn("rank",
                       F.row_number().over(Window.partitionBy("USER_ID").orderBy("score"))). \
            where(F.col("rank") <= self.MAX_RECOS). \
            withColumn("rating", F.lit(1.0) / F.col("rank")). \
            drop("rank")

        # 3 - Assign the reco to each user
        reco_df = all_users_df. \
            join(user_id_to_hash_df, all_users_df.USER_ID == user_id_to_hash_df.USER_ID, "left"). \
            drop(user_id_to_hash_df.USER_ID). \
            join(reco_df, reco_df.USER_ID == user_id_to_hash_df.HASH_ID). \
            drop(reco_df.USER_ID). \
            select("USER_ID", "PROGRAM_ID", "AFFINITY", "reco_origin", "rating",
                   reco_df.CHANNEL_ID, reco_df.EPISODE_ID)

        # 4 - Remove already seen films
        reco_df = reco_df. \
            join(user_watch_df, (reco_df.PROGRAM_ID == user_watch_df.PROGRAM_ID) &
                 (reco_df.USER_ID == user_watch_df.USER_ID), "leftanti")

        return format_for_reco_output(reco_df, "recommendations",
                                      field_names=("PROGRAM_ID", "AFFINITY", "reco_origin", "rating",
                                                   "CHANNEL_ID", "EPISODE_ID"))

    def write_recos_to_snowflake(self, vod_reco_df, table_name,
                                 write_mode="append"):
        final_user_recos_df = vod_reco_df. \
            withColumn("UPDATE_DATE", F.lit(self.now))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)


if __name__ == "__main__":
    job = VodRecoJob()
    job.launch()
