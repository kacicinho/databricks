from datetime import date, timedelta

from pyspark.sql.window import Window
import pyspark.sql.functions as F

from databricks_jobs.common import Job
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake


class NextWatchJob(Job):
    TABLE_NAME = "USER_NEXT_WATCH_RECO"
    UNWANTED_CAT = [3, 3]
    # Catégories :
    # 3: sport
    UNWANTED_KINDS = [34, 43, 45, 52]
    # Kinds :
    # 34 : pornographiques
    # 43 : information et journaux télévisés
    # 45 : information
    # 52 : météo
    MIN_USER_WATCH_TIME_IN_MINUTES = 15
    MIN_DURATION_BROADCAST_MINUTES = 5
    USER_WATCH_WINDOW = timedelta(days=15)

    def __init__(self, *args, **kwargs):
        super(NextWatchJob, self).__init__(*args, **kwargs)

        self.now = date.today()
        self.delta = timedelta(days=3)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 -Prepare season ,episode and available program tables
        fact_episode = self.build_fact_tables()
        prog_df = self.prepare_program_df()

        # 2 - Find what has been seen by the user in the last X days and what is the next episode
        user_next_episode_df = self.build_user_next_episode(fact_episode)

        # 3 - Find in subsequent progs what is compatible with the user to watch next
        user_available_next_watch_df = self.build_user_available_watch(prog_df, user_next_episode_df)

        # 4 - Write the intermediate table of user next episodes
        write_df_to_snowflake(user_available_next_watch_df, self.write_options, "NEXT_WATCH_SERIES", mode="overwrite")

        # 5 - Build next watch reco with aggregation of item filtering reco and next episode
        user_next_watch_reco = self.build_next_watch_recos(user_available_next_watch_df)

        # 6 - Create User 0 reco based on watch trending
        user_reco_0 = self.build_non_personalized_reco()

        # 7 Write down final reco result
        self.write_recos(user_next_watch_reco, user_reco_0)

    def prepare_program_df(self):
        episode_info_df = build_episode_df(self.spark, self.options)
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False)

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True)
        prog_df = broadcast_df.union(vod_df). \
            select("PROGRAM_ID", "EPISODE_ID", "REF_PROGRAM_CATEGORY_ID", "REF_PROGRAM_KIND_ID"). \
            withColumn("IS_AVAILABLE", F.lit(1)). \
            dropDuplicates()

        return prog_df

    def build_fact_tables(self):
        """ Refactoring of season and episode :

        the first episode of each series is used as reference and episode number is the distance with the first episode
        "Reset" season (when episode number is set to 1 at each new season) are transformed to match with the above model

        "Reset" seasons are identified looking at the number of first and last episode of the serie
        """
        season_df = load_snowflake_table(self.spark, self.options, "backend.season"). \
            withColumnRenamed("NUMBER", "SEASON_NUMBER")
        episode_df = load_snowflake_table(self.spark, self.options, "backend.episode"). \
            withColumnRenamed("NUMBER", "EPISODE_NUMBER")

        # Refactoring of season table
        fact_season = season_df. \
            join(episode_df, season_df.ID == episode_df.SEASON_ID). \
            drop(episode_df.PROGRAM_ID). \
            where(F.col("EPISODE_NUMBER") > 0). \
            groupBy("SEASON_ID", "PROGRAM_ID", "SEASON_NUMBER"). \
            agg(F.countDistinct("EPISODE_NUMBER").alias("SEASON_EPISODE_NUMBER"),
                F.min("EPISODE_NUMBER").alias("MIN_EPI_NUM"),
                F.max("EPISODE_NUMBER").alias("MAX_EPI_NUM")). \
            withColumn("IS_RESET_SEASON",
                       F.when((F.col("MAX_EPI_NUM") - F.col("MIN_EPI_NUM") + 1 == F.col("SEASON_EPISODE_NUMBER")) &
                              (F.col("MIN_EPI_NUM") == 1), F.lit(1)).otherwise(F.lit(0))). \
            withColumn("CUMULATIVE_EPISODE",
                       F.when(F.col("IS_RESET_SEASON") == 1,
                              F.sum("SEASON_EPISODE_NUMBER").
                              over(Window.partitionBy("PROGRAM_ID").orderBy("SEASON_NUMBER").
                                   rowsBetween(Window.unboundedPreceding, Window.currentRow))).otherwise(F.lit(None)))

        tot_episode_number = F.col("CUMULATIVE_EPISODE") - F.col("SEASON_EPISODE_NUMBER") + F.col("EPISODE_NUMBER")
        fact_episode = episode_df. \
            where(F.col("EPISODE_NUMBER") > 0). \
            join(fact_season, "SEASON_ID"). \
            drop(fact_season.PROGRAM_ID). \
            withColumn("TOTAL_EPI_NUM", F.when(F.col("IS_RESET_SEASON") == 0, F.col("EPISODE_NUMBER")).
                       otherwise(tot_episode_number)). \
            select("PROGRAM_ID", episode_df.ID.alias("EPISODE_ID"), "SEASON_ID", "MIN_EPI_NUM", "MAX_EPI_NUM",
                   "SEASON_NUMBER", "SEASON_EPISODE_NUMBER", "TOTAL_EPI_NUM")

        return fact_episode

    def build_user_next_episode(self, fact_episode):
        """
        Build user next episode based on user history and previous fact episode table
        """
        watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch_7d_agg"). \
            where(F.col("DATE_RECEIVED_AT").between(self.now - self.USER_WATCH_WINDOW, self.now)). \
            where(F.col("WATCH_FRAC_DURATION") > 0.8). \
            select("USER_ID", "EPISODE_ID", "PROGRAM_ID", "DATE_RECEIVED_AT"). \
            distinct()

        user_history = watch_df.alias("fw"). \
            join(fact_episode.alias("fe1"), F.col("fe1.EPISODE_ID") == F.col("fw.EPISODE_ID")). \
            withColumn("EPISODE_RANK", F.row_number().over(Window.partitionBy("fw.PROGRAM_ID", "fw.USER_ID").
                                                           orderBy(F.desc("fe1.TOTAL_EPI_NUM")))). \
            where(F.col("EPISODE_RANK") == 1). \
            select("fw.USER_ID", "fw.DATE_RECEIVED_AT", "fw.PROGRAM_ID", "fe1.EPISODE_ID", "fe1.TOTAL_EPI_NUM")

        user_next_episodes = user_history. \
            join(fact_episode.alias("fe2"), (F.col("fe2.PROGRAM_ID") == F.col("fw.PROGRAM_ID")) &
                 (F.col("fe2.TOTAL_EPI_NUM") == F.col("fe1.TOTAL_EPI_NUM") + 1)). \
            select("fw.USER_ID",
                   "fw.DATE_RECEIVED_AT",
                   F.col("fw.PROGRAM_ID").alias("LAST_PROGRAM_ID"),
                   F.col("fe1.EPISODE_ID").alias("LAST_EPISODE_ID"),
                   F.col("fe1.TOTAL_EPI_NUM").alias("LAST_EPISODE_NUM"),
                   F.col("fe2.PROGRAM_ID").alias("NEXT_PROGRAM_ID"),
                   F.col("fe2.EPISODE_ID").alias("NEXT_EPISODE_ID"),
                   F.col("fe2.TOTAL_EPI_NUM").alias("NEXT_EPISODE_NUM")). \
            distinct()

        return user_next_episodes

    @staticmethod
    def build_user_available_watch(prog_df, user_next_episode_df):
        """
        IS_AVAILABLE is set to 1 if next episode is available to watch in next 3 days
        """
        user_available_watch_df = user_next_episode_df. \
            join(prog_df, prog_df.EPISODE_ID == user_next_episode_df.NEXT_EPISODE_ID, "left"). \
            drop(prog_df.EPISODE_ID). \
            fillna(0, subset=["IS_AVAILABLE"])

        return user_available_watch_df

    def build_next_watch_recos(self, user_available_next_watch_df):
        """
        Build user next watch reco:

        -Split program in to categories SeriesxMovies vs Others
        -Aggregate next episode and item filtering reco
        - Results are ranked as following :
            -First Next episode and item filtering program for SeriesxMovies
            -Then Next episode and item filtering program for Others
        """
        item_filtering_recos = load_snowflake_table(self.spark, self.options, "ML.USER_ITEM_RECO_PAGE_FLAT_RECOS"). \
            where(F.col("PROGRAM_ID_I") != F.col("PROGRAM_ID_J")). \
            withColumn("IS_SERIE_MOVIE", F.when(F.col("PROGRAM_J_CAT").isin(*[1, 2]), F.lit(1)).otherwise(F.lit(0))). \
            withColumn("SOURCE", F.lit("item_based_filtering")). \
            select("USER_ID", "PROGRAM_ID_I", "PROGRAM_ID_J", "PROGRAM_J_CAT", "FINAL_DATE_RANK", "FINAL_SIM_RANK",
                   "SOURCE", "IS_SERIE_MOVIE")

        user_available_next_watch_df = user_available_next_watch_df. \
            where(F.col("IS_AVAILABLE") == 1). \
            where(~F.col("REF_PROGRAM_CATEGORY_ID").isin(*self.UNWANTED_CAT)). \
            where(~F.col("REF_PROGRAM_KIND_ID").isin(*self.UNWANTED_KINDS)). \
            withColumn("IS_SERIE_MOVIE",
                       F.when(F.col("REF_PROGRAM_CATEGORY_ID").isin(*[1, 2]), F.lit(1)).otherwise(F.lit(0))). \
            withColumn("rank", F.rank().over(Window.partitionBy("USER_ID").orderBy(F.desc("IS_SERIE_MOVIE")))). \
            where(F.col("rank") == 1)

        user_available_next_watch_format_df = user_available_next_watch_df. \
            withColumn("PROGRAM_ID_I", F.col("PROGRAM_ID")). \
            withColumn("PROGRAM_ID_J", F.col("PROGRAM_ID")). \
            withColumn("SOURCE", F.lit("next_episode")). \
            withColumn("PROGRAM_J_CAT", F.col("REF_PROGRAM_CATEGORY_ID")). \
            withColumn("FINAL_DATE_RANK",
                       F.rank().over(Window.partitionBy("USER_ID").orderBy(F.desc("DATE_RECEIVED_AT")))). \
            withColumn("FINAL_SIM_RANK", F.lit(0)). \
            select("USER_ID", "PROGRAM_ID_I", "PROGRAM_ID_J", "PROGRAM_J_CAT", "FINAL_DATE_RANK", "FINAL_SIM_RANK",
                   "SOURCE", "IS_SERIE_MOVIE")

        next_watch_recos = item_filtering_recos. \
            union(user_available_next_watch_format_df). \
            withColumn("rank", F.row_number().over(
                Window.partitionBy("USER_ID").orderBy(F.desc("IS_SERIE_MOVIE"), F.col('FINAL_SIM_RANK'),
                                                      F.col('FINAL_DATE_RANK')))). \
            groupby("USER_ID"). \
            agg(F.collect_list(F.struct(
                F.col("rank").alias("rank"),
                F.col("SOURCE").alias("reco_source"),
                F.col("PROGRAM_ID_I").alias("source_program_id"),
                F.col("PROGRAM_ID_J").alias("reco_program_id"),
                F.col("PROGRAM_J_CAT").alias("reco_ref_cat_id")
            )).alias("recommendations"))

        return next_watch_recos

    def build_non_personalized_reco(self):
        """
        Build user 0 reco

        Select the most watched program and keep for each one the best ranked reco from item filtering recos

        Keep top 100 recommendations
        """
        item_filtering_recos = load_snowflake_table(self.spark, self.options, "ML.USER_ITEM_RECO_PAGE_FLAT_RECOS"). \
            where(F.col("PROGRAM_ID_I") != F.col("PROGRAM_ID_J")). \
            withColumn("IS_SERIE_MOVIE", F.when(F.col("PROGRAM_J_CAT").isin(*[1, 2]), F.lit(1)).otherwise(F.lit(0))). \
            withColumn("SOURCE", F.lit("item_based_filtering")). \
            select("USER_ID", "PROGRAM_ID_I", "PROGRAM_ID_J", "PROGRAM_J_CAT", "FINAL_DATE_RANK", "FINAL_SIM_RANK",
                   "SOURCE", "IS_SERIE_MOVIE")

        trending_recos = item_filtering_recos. \
            where(F.col("PROGRAM_J_CAT") == 1). \
            withColumn("program_count", F.count("USER_ID").over(Window.partitionBy("PROGRAM_ID_I"))). \
            groupBy("PROGRAM_ID_I", "PROGRAM_ID_J", "PROGRAM_J_CAT", "program_count"). \
            agg(F.mean("FINAL_SIM_RANK").alias("mean_sim_rank")). \
            withColumn("reco_rank",
                       F.row_number().over(Window.partitionBy("PROGRAM_ID_J").orderBy(F.desc("program_count")))). \
            where(F.col("reco_rank") == 1).\
            withColumn("rank", F.row_number().over(Window.orderBy("mean_sim_rank", F.desc("program_count")))). \
            where(F.col("rank") <= 100). \
            withColumn("SOURCE", F.lit("item_based_filtering")). \
            withColumn("USER_ID", F.lit(0)).\
            groupby("USER_ID").\
            agg(F.collect_list(F.struct(
                F.col("rank").alias("rank"),
                F.col("SOURCE").alias("reco_source"),
                F.col("PROGRAM_ID_I").alias("source_program_id"),
                F.col("PROGRAM_ID_J").alias("reco_program_id"),
                F.col("PROGRAM_J_CAT").alias("reco_ref_cat_id")
            )).alias("recommendations"))

        return trending_recos

    def write_recos(self, user_next_watch_reco, user_reco_0, mode='append'):
        final_reco = user_next_watch_reco.\
            union(user_reco_0).\
            withColumn("UPDATE_DATE", F.lit(self.now))

        write_df_to_snowflake(final_reco, self.write_options, self.TABLE_NAME, mode=mode)


if __name__ == "__main__":
    job = NextWatchJob()
    job.launch()
