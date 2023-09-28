from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.db_common import build_broadcast_df_with_episode_info
from databricks_jobs.db_common import build_episode_df, FREE_BUNDLE_CHANNELS, build_tvbundle_hashes, \
    build_user_to_paying_tv_bundle, build_product_to_tv_bundle, MANGO_CHANNELS, \
    build_user_to_allowed_channels, build_vod_df_with_episode_infos
from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from databricks_jobs.jobs.utils.popular_reco_utils import format_for_reco_output, keep_top_X_programs
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake, \
    create_spark_df_from_data


class ReplayRecoJob(Job):
    MAX_RECOS = 25
    MAX_RECO_PER_ORIGIN = 8
    AUTHORIZED_TVBUNDLES = (25, 90, 26, 60, 31, 142, 146)
    DAILY_RECO_TABLE = "USER_REPLAY_RECOMMENDATIONS_VAR"
    AB_TEST_VARIATION = 'A'

    def __init__(self, *args, **kwargs):
        super(ReplayRecoJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=2)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :
        1 - Extract the main data about episodes
        2 - Compute popularity score per episode
        3 - Keep top N and filter based on paying paying users availability
        """
        # 1 - Extract the main data about episodes
        episode_df = build_episode_df(self.spark, self.options)
        prog_affinity_df = load_snowflake_table(self.spark, self.options, "external_sources.DAILY_PROG_AFFINITY")
        replay_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_df,
                                                    self.now, self.delta, min_duration_in_mins=15,
                                                    video_type=("REPLAY",))
        mango_vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_df,
                                                       self.now, self.delta, min_duration_in_mins=15,
                                                       video_type=("VOD",)). \
            where(F.col("CHANNEL_ID").isin(*MANGO_CHANNELS))

        first_aired_df = self.build_first_broadcast_time_df(episode_df)
        episode_watch_df = self.compute_episode_live_stats()

        # 2 - Compute popularity score per episode
        score_df, reweighted_score_columns = \
            self.build_enriched_program_df(replay_df, first_aired_df, episode_watch_df)
        score_df = score_df. \
            join(mango_vod_df, mango_vod_df.PROGRAM_ID == score_df.PROGRAM_ID, "leftanti")

        # Per program, we keep only the episode with the best reweighted score
        score_df = score_df.\
            withColumn("episode_rank", F.row_number().over(
                Window.partitionBy("PROGRAM_ID").orderBy(F.desc(reweighted_score_columns[0]))
            )). \
            where("episode_rank = 1"). \
            drop("episode_rank")

        # 3 - Keep top N and filter based on paying paying users availability
        top_x_replays = keep_top_X_programs(score_df, False, reweighted_score_columns)
        top_x_replays = top_x_replays.\
            join(prog_affinity_df, prog_affinity_df.PROGRAM_ID == top_x_replays.PROGRAM_ID, "left").\
            drop(prog_affinity_df.PROGRAM_ID)
        PopularRecoJob.save_intermediary_results(self.spark, self.write_options, top_x_replays,
                                                 table_name_aff="RAW_AFFINITY_REPLAY_RECO",
                                                 table_name_bundle="RAW_BUNDLE_REPLAY_RECO")
        final_reco = self.select_recos(top_x_replays)
        self.write_recos_to_snowflake(final_reco, self.DAILY_RECO_TABLE, "append")

    @staticmethod
    def build_enriched_program_df(replay_df, first_aired_df, episode_watch_df):
        """
        Two main steps :
        - Filtering of fresh content
        - Scoring based on the age and bundle

        :param replay_df:
        :param first_aired_df:
        :param episode_watch_df:
        :return:
        """
        # 1 - Join all the infos and keep only the fresh contents
        # Depending on the category, a replay should stay longer in the rail
        category_based_expiration = \
            F.when(F.col("REF_PROGRAM_CATEGORY_ID").isin(1, 2, 8), 7). \
            when(F.col("REF_PROGRAM_CATEGORY_ID").isin(3), 1). \
            when(F.col("REF_PROGRAM_CATEGORY_ID").isin(4, 5, 9), 3). \
            otherwise(4)  # Indéterminé category is difficult to anticipate

        joined_replay_df = replay_df. \
            join(first_aired_df, replay_df.EPISODE_ID == first_aired_df.EPISODE_ID). \
            drop(first_aired_df.EPISODE_ID). \
            join(episode_watch_df, replay_df.EPISODE_ID == episode_watch_df.EPISODE_ID). \
            drop(episode_watch_df.EPISODE_ID)

        filtered_episode_watch_df = joined_replay_df. \
            where((joined_replay_df.age <= 7) | joined_replay_df.age.isNull()). \
            where(~(replay_df.REF_PROGRAM_KIND_ID.isin(36, 43) & (replay_df.REF_PROGRAM_CATEGORY_ID == 4))). \
            where((joined_replay_df.age <= category_based_expiration) | joined_replay_df.age.isNull())

        # 2 - Scoring : adapted based on bundle, freshness of the program
        # Approximate ratio between the number of super active and the number of subscriber to an offer
        bundle_based_reweighting = \
            F.when(F.col("aired_on_free") == 1, F.lit(1)). \
            when(F.col("aired_on_m_plus") == 1, F.lit(3)). \
            when(F.col("aired_on_m_ext") == 1, F.lit(18)).\
            when(F.col("aired_on_cine_plus") == 1, F.lit(83)). \
            when(F.col("aired_on_ocs") == 1, F.lit(86)).\
            otherwise(1)

        def build_mix_fn(age_col, expiration_expr, reweighting_expr):
            def mix_fn(score_col):
                return (F.lit(1) - F.lit(0.5) * (age_col / expiration_expr)) * score_col * reweighting_expr
            return mix_fn

        mix_fn = build_mix_fn(first_aired_df.age, category_based_expiration, bundle_based_reweighting)
        score_columns = ["total_duration", "engaged_watchers"]
        reweighted_score_columns = [f"reweighted_{score_col}" for score_col in score_columns]

        score_df = filtered_episode_watch_df. \
            select(filtered_episode_watch_df.EPISODE_ID, filtered_episode_watch_df.CHANNEL_ID,
                   filtered_episode_watch_df.PROGRAM_ID,
                   *[mix_fn(F.col(score_col)).alias(r_score_col)
                     for score_col, r_score_col in zip(score_columns, reweighted_score_columns)])

        return score_df, reweighted_score_columns

    def select_recos(self, score_df):
        """
        Broadcast of the top N recos to all users (only paying for now), no personalisation

        - K different default recos are computed based on the available bundle for the user
        - The broadcast assign the default reco to the right users
        - Already seen movies are also removed

        :param score_df:
        :return:
        """
        # 1 - Build bundle information for users
        prod_to_bundle_df = build_product_to_tv_bundle(self.spark, self.options)
        user_to_tvbundle_df = build_user_to_paying_tv_bundle(self.spark, self.options, prod_to_bundle_df,
                                                             self.now). \
            union(create_spark_df_from_data(self.spark, {"USER_ID": [0], "TVBUNDLE_ID": 25}))

        channels_per_user = build_user_to_allowed_channels(self.spark, self.options, user_to_tvbundle_df,
                                                           authorized_bundles=self.AUTHORIZED_TVBUNDLES)
        hash_to_bundle_df, user_id_to_hash_df = build_tvbundle_hashes(channels_per_user)
        all_users_df = user_to_tvbundle_df.select("USER_ID").distinct()

        # 1.2 - Remove already seen films
        prog_df = load_snowflake_table(self.spark, self.options, "backend.program")
        user_watch_df = PopularRecoJob.build_user_watch_history(self.spark, self.options, self.now)
        user_watch_df = user_watch_df. \
            join(prog_df, prog_df.ID == user_watch_df.PROGRAM_ID). \
            where("REF_PROGRAM_CATEGORY_ID = 1")

        # 2 - Cross join top_k_progs x possible_bundles
        reco_df = score_df. \
            where("best_rank < 200"). \
            crossJoin(hash_to_bundle_df). \
            where(score_df.CHANNEL_ID.isin(*FREE_BUNDLE_CHANNELS) |
                  (hash_to_bundle_df.CHANNEL_ID == score_df.CHANNEL_ID)). \
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
            select("USER_ID", "PROGRAM_ID", "reco_origin", "rating",
                   reco_df.CHANNEL_ID, reco_df.EPISODE_ID)

        # 4 - Remove already seen films
        reco_df = reco_df. \
            join(user_watch_df, (reco_df.PROGRAM_ID == user_watch_df.PROGRAM_ID) &
                 (reco_df.USER_ID == user_watch_df.USER_ID), "leftanti")

        return format_for_reco_output(reco_df, "recommendations",
                                      field_names=("PROGRAM_ID", "reco_origin", "rating",
                                                   "CHANNEL_ID", "EPISODE_ID"))

    def build_first_broadcast_time_df(self, episode_info_df):
        """
        We compute :
        - Smallest timestamp when the episode appeared in all broadcast
        - On which bundle the program appeared
        Note : M6 bundle is treated as free for now
        """
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now - timedelta(days=90), self.now,
                                                            min_duration_in_mins=5). \
            where(F.col("TVBUNDLE_ID").isin(*self.AUTHORIZED_TVBUNDLES))

        result_df = broadcast_df. \
            groupBy("EPISODE_ID", "PROGRAM_ID"). \
            agg(F.min("start_at").alias("first_aired"),
                *[F.max(broadcast_df.TVBUNDLE_ID.isin(*bundle_ids)).alias(name)
                  for bundle_ids, name in [((25,), "aired_on_free"),
                                           ((90, 142, 146), "aired_on_m_plus"), ((26,), "aired_on_m_ext"),
                                           ((31,), "aired_on_cine_plus"), ((60,), "aired_on_ocs")]]). \
            withColumn("age", F.datediff(F.lit(self.now), "first_aired"))

        for col_name in ["aired_on_free", "aired_on_m_plus", "aired_on_m_ext", "aired_on_cine_plus", "aired_on_ocs"]:
            result_df = result_df. \
                withColumn(col_name, F.max(col_name).over(Window.partitionBy("PROGRAM_ID")))

        return result_df. \
            drop("PROGRAM_ID")

    def compute_episode_live_stats(self):
        """
        Compute per episode_id on fact_watch
        - "total_duration"
        - "engaged_watchers"
        """
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch"). \
            where(F.col("REAL_START_AT") > self.now - timedelta(days=7))
        backend_episode_df = load_snowflake_table(self.spark, self.options, "backend.episode")

        # Engaged duration : 70% of the program has been watched, if it represents less than 20mins then 20 mins
        # Sometimes we don't have a duration, then it's 1 hour for someone to be engaged
        engaged_duration_fn = F.greatest(
            F.when(backend_episode_df.DURATION > 0, 0.7 * backend_episode_df.DURATION).otherwise(60 * 60),
            F.lit(20 * 60)
        )

        return fact_watch_df. \
            join(backend_episode_df, backend_episode_df.ID == fact_watch_df.EPISODE_ID). \
            drop(backend_episode_df.ID). \
            drop(backend_episode_df.PROGRAM_ID). \
            groupBy("EPISODE_ID", "PROGRAM_ID"). \
            agg(F.sum(fact_watch_df.DURATION).alias("total_duration"),
                F.countDistinct(
                    F.when(fact_watch_df.DURATION > engaged_duration_fn,
                           fact_watch_df.USER_ID).otherwise(F.lit(None))).alias("engaged_watchers")
                ). \
            withColumn("total_duration", F.max("total_duration").over(Window.partitionBy("PROGRAM_ID"))). \
            withColumn("engaged_watchers", F.max("engaged_watchers").over(Window.partitionBy("PROGRAM_ID"))).\
            drop("PROGRAM_ID")

    ##############
    #   Helpers
    ##############

    def write_recos_to_snowflake(self, vod_reco_df, table_name,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        final_user_recos_df = vod_reco_df. \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)


if __name__ == "__main__":
    job = ReplayRecoJob()
    job.launch()
