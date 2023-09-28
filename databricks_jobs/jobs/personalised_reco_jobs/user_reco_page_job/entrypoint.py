from datetime import timedelta
from databricks_jobs.db_common import build_vod_df_with_episode_infos, build_episode_df
from databricks_jobs.jobs.utils.user_item_reco_page import \
    get_available_prog_similarity_table, get_user_history, add_similar_programs, \
    limit_recos, add_prog_rank, get_fact_watch, get_similarities
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
import pyspark.sql.functions as F
from pyspark.sql import Window


class UserRecoPageJob(Job):
    # RECO PARAMS
    nb_prog_to_make_recos_on = 10
    reco_per_prog = 10
    user_watch_history_days_look_back = 360  # we would like this to be infinite

    unwanted_categories = (3, 3)
    # Catégories :
    # 3: sport

    unwanted_kinds = (34, 43, 45, 52)
    # Kinds :
    # 34 : pornographiques
    # 43 : information et journaux télévisés
    # 45 : information
    # 52 : météo

    # TABLES NAMES
    # read
    similarity_table = "ML.ITEM_BASED_FILTERING_PROGRAM_SIMILARITY"
    # write
    flat_reco_table = 'ML.USER_ITEM_RECO_PAGE_FLAT_RECOS'
    factored_daily_recos_table = 'ML.USER_ITEM_RECO_PAGE_RECO_READY_RECOS'

    def __init__(self, *args, **kwargs):
        super(UserRecoPageJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.delta = timedelta(days=3)
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 get data from snowflake
        fact_watch = get_fact_watch(self.spark, self.options,
                                    days_look_back=self.user_watch_history_days_look_back,
                                    unwanted_categories=self.unwanted_categories,
                                    unwanted_kinds=self.unwanted_kinds)

        similarities = get_similarities(self.spark, self.options, self.similarity_table,
                                        unwanted_categories=self.unwanted_categories,
                                        unwanted_kinds=self.unwanted_kinds)

        episode_info_df = build_episode_df(self.spark, self.options)
        available_programs = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df,
                                                             self.now, self.delta, min_duration_in_mins=15)

        # 2 process dataframes
        available_similar_programs = get_available_prog_similarity_table(similarities, available_programs,
                                                                         limit_sim_prog=self.reco_per_prog * 5)
        users_history = get_user_history(fact_watch, limit_date_rank=self.nb_prog_to_make_recos_on * 2)
        users_history_with_recos = add_similar_programs(users_history, available_similar_programs)
        users_history_with_recos_and_prog_rank = add_prog_rank(users_history_with_recos)

        # 3 compute flat and grouped recos
        flat_recos = limit_recos(users_history_with_recos_and_prog_rank, self.nb_prog_to_make_recos_on,
                                 self.reco_per_prog)

        factored_daily_recos = flat_recos \
            .orderBy(F.col('FINAL_SIM_RANK'), F.col('FINAL_DATE_RANK')) \
            .withColumn('rank', F.row_number().over(
                Window.partitionBy('USER_ID').orderBy(F.col('FINAL_SIM_RANK'), F.col('FINAL_DATE_RANK')))) \
            .groupby("USER_ID").agg(F.collect_list(F.struct(
                F.col("rank").alias("rank"),
                F.col("FINAL_DATE_RANK").alias("source_program_recency_rank"),
                F.col("PROGRAM_ID_I").alias("source_program_id"),
                F.col("PROGRAM_ID_J").alias("reco_program_id"),
                F.col("FINAL_SIM_RANK").alias("reco_similarity_rank"),
                F.col("PROGRAM_J_CAT").alias("reco_ref_cat_id")
            )))

        # 5 write recos
        write_df_to_snowflake(flat_recos, self.write_options, self.flat_reco_table, "overwrite")
        write_df_to_snowflake(factored_daily_recos, self.write_options, self.factored_daily_recos_table, "overwrite")


if __name__ == "__main__":
    job = UserRecoPageJob()
    job.launch()
