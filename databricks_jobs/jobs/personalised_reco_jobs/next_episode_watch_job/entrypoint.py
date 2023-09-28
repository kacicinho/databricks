from datetime import datetime, timedelta

from pyspark.sql.functions import col, desc, rank, collect_set, row_number, lit, sum, struct, array_contains
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos
from databricks_jobs.jobs.utils.popular_reco_utils import load_snowflake_table, write_df_to_snowflake


class NextEpisodeWatchJob(Job):
    MIN_USER_WATCH_TIME_IN_MINUTES = 15
    MIN_DURATION_BROADCAST_MINUTES = 5

    def __init__(self, *args, **kwargs):
        super(NextEpisodeWatchJob, self).__init__(*args, **kwargs)

        self.now = datetime.now()
        self.delta = timedelta(days=3)

        self.snowflakeUser = self.conf.get("user_name", "")
        self.snowflakePassword = self.conf.get("password", "")

        self.options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflakeUser,
            "sfPassword": self.snowflakePassword,
            "sfDatabase": "PROD",
            "sfSchema": "public",
            "sfWarehouse": "DATABRICKS_XS_WH"
        }

        self.write_options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflakeUser,
            "sfPassword": self.snowflakePassword,
            "sfDatabase": "PROD",
            "sfSchema": "ML",
            "sfWarehouse": "DATABRICKS_XS_WH",
            "keep_column_case": "off"
        }

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 - Collect the infos on the next available programs
        episode_info_df = build_episode_df(self.spark, self.options)
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            self.MIN_DURATION_BROADCAST_MINUTES)
        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta)

        # 2 - Find what has been seen by the user in the last X days, persist for next queries
        user_episode_watch_df = self.build_user_last_watch_df(episode_info_df).persist()

        # 3 - Find in subsequent progs what is compatible ofr the user to watch next
        user_next_broadcast_watch_df = self.filter_programs_with_next_watch(user_episode_watch_df, broadcast_df)
        user_next_vod_watch_df = self.filter_programs_with_next_watch(user_episode_watch_df, vod_df)

        # 4 - Recombine the two sets of recos
        all_possible_watches_df = self.join_sources(user_next_broadcast_watch_df, user_next_vod_watch_df)

        # 5 - Finally write the table down
        write_df_to_snowflake(all_possible_watches_df, self.write_options, "EPISODE_FOLLOW_UP", mode="overwrite")

    def build_user_last_watch_df(self, episode_info_df):
        """
        To keep only the meaningful program propositions, we need to do  :

        1 - Program filtering for each user
        over the last month
        - User must watch at least X minutes of the program
        - The program must be among its top n most watched to be pushed
        - The user must be somewhat active compared to other users on this show

        2 - Getting the last seen episode
        Once shows are filtered, we keep only the id of the most recent broadcast for each program
        """
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch")
        watch_lookback = timedelta(days=30)

        # Window last episode over each (USER, PROGRAM)
        episode_window = Window.partitionBy("USER_ID", "PROGRAM_ID"). \
            orderBy(desc(struct("SEASON_NUMBER", "EPISODE_NUMBER")))
        # USER x PROGRAM window
        user_program_window = Window.partitionBy("USER_ID", "PROGRAM_ID")

        # Keep the last episode for each couple user_id x program_id,
        # for program with more than MIN_WATCH_TIME_IN_MINUTES mins
        last_episode_per_user_per_program_df = fact_watch_df. \
            join(episode_info_df, episode_info_df.EPISODE_ID == fact_watch_df.EPISODE_ID). \
            drop(episode_info_df.PROGRAM_ID). \
            drop(episode_info_df.EPISODE_ID). \
            drop(episode_info_df.DURATION). \
            where(col("REAL_START_AT") > self.now - watch_lookback). \
            withColumn("total_duration", sum("DURATION").over(user_program_window)). \
            where(col("total_duration") > self.MIN_USER_WATCH_TIME_IN_MINUTES * 60)

        # Collect he seen episode ids and keep value of the highest episode number / season number
        last_episode_per_user_per_program_df = last_episode_per_user_per_program_df. \
            select("USER_ID", "PROGRAM_ID", "total_duration", "EPISODE_ID", "SEASON_NUMBER", "EPISODE_NUMBER",
                   "total_duration"). \
            distinct(). \
            withColumn("seen_episodes", collect_set("EPISODE_ID").over(episode_window)). \
            withColumn("rank", rank().over(episode_window)). \
            where(col("rank") == 1)

        # Rename the fields for the next processing
        user_last_watch_and_program_percentiles = last_episode_per_user_per_program_df. \
            withColumnRenamed("EPISODE_NUMBER", "last_seen_episode"). \
            withColumnRenamed("SEASON_NUMBER", "last_seen_season"). \
            withColumnRenamed("REAL_START_AT", "last_watch_time"). \
            select("USER_ID", "PROGRAM_ID", "last_seen_episode", "seen_episodes",
                   "last_seen_season", "total_duration"). \
            distinct()

        return user_last_watch_and_program_percentiles

    def filter_programs_with_next_watch(self, user_last_watch, next_broadcast_df):
        """
        Set of rules to be in the set of next watches

        For broadcast : not already seen
        For tv shows : should be following or next season
        For films : we assume there is no follow-up

        :param user_last_watch:
        :param next_broadcast_df:
        :return:
        """

        # Broadcast interesting for the user
        is_next_season_for_user = (col("last_seen_season") + 1 == col("SEASON_NUMBER")) & (col("EPISODE_NUMBER") <= 1)
        is_following_episode_for_user = ((col("EPISODE_NUMBER") > col("last_seen_episode")) & (
            col("last_seen_season") == col("SEASON_NUMBER")))
        # Category : 2=film 4=Informations 5=Divertissement 8=Documentaires 9=Culture
        is_program_a_broadcast = (col("REF_PROGRAM_CATEGORY_ID").isin(*[4, 5, 8, 9]))
        is_program_a_show = (col("REF_PROGRAM_CATEGORY_ID") == 2)
        # Window for episode order
        episode_incr_window = Window.partitionBy("USER_ID", "PROGRAM_ID"). \
            orderBy(struct("SEASON_NUMBER", "EPISODE_NUMBER"))
        total_watch_window = Window.partitionBy("USER_ID").orderBy(desc("total_duration"))

        # 2 - Join on user watch history
        matching_episode_df = next_broadcast_df. \
            join(user_last_watch, "PROGRAM_ID", "inner"). \
            drop(user_last_watch.PROGRAM_ID)

        # Keep only the lowest episode_id
        continuation_episode_df = matching_episode_df. \
            where(~array_contains("seen_episodes", col("EPISODE_ID"))). \
            where(is_program_a_broadcast |
                  (is_program_a_show & (is_following_episode_for_user | is_next_season_for_user))). \
            withColumn("episode_order", row_number().over(episode_incr_window)). \
            where(col("episode_order") == 1). \
            drop("episode_order"). \
            withColumn("rank", rank().over(total_watch_window)). \
            withColumn("is_following_episode_for_user", is_following_episode_for_user). \
            withColumn("is_next_season_for_user", is_next_season_for_user). \
            select("USER_ID", "PROGRAM_ID", "EPISODE_ID", "EPISODE_NUMBER", "SEASON_NUMBER", "rank",
                   "is_following_episode_for_user", "is_next_season_for_user",
                   "last_seen_episode", "last_seen_season"). \
            withColumn("computation_date", lit(self.now.date()))

        return continuation_episode_df

    def join_sources(self, df1, df2):
        """
        Used to combine the sources from broadcast and vod
        """
        return df1. \
            withColumn("source", lit("broadcast")). \
            union(
                df2.withColumn("source", lit("vod"))
            ). \
            withColumn("row_number", row_number().over(Window.partitionBy("PROGRAM_ID", "USER_ID").
                                                       orderBy(desc("source")))). \
            where(col("row_number") == 1). \
            drop("row_number")


if __name__ == "__main__":
    job = NextEpisodeWatchJob()
    job.launch()
