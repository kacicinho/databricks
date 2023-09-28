from datetime import timedelta

import pyspark.sql.functions as F
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, load_snowflake_table, \
    write_df_to_snowflake, get_snowflake_connection, execute_snowflake_query


class RailClicAttributionJob(Job):
    TABLE_NAME = "FACT_RAIL_CLIC_ATTRIBUTION"

    def __init__(self, *args, **kwargs):
        super(RailClicAttributionJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=7)
        self.rails = ['this-week-recommendations', 'today-recommendations', 'tonight_1', 'vods-mango', 'trending-global']

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "DW", keep_column_case="off")

        self.logger.info(f"Running on date : {self.now}")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        # 1 - Delete past 7 days
        self.delete_past_7days()

        # 2 - Recompute clic attrbition on past 7 days.
        rail_clic_attribution_df = self.build_rail_clic_attribution()

        # 3 - Write updated data to snowflake
        write_df_to_snowflake(rail_clic_attribution_df, self.write_options, self.TABLE_NAME, 'append')

    def build_rail_clic_attribution(self):
        """
        We left join events (watch / bookmark / follow) to clics on program/person page
        NB: All clics done on the past 7 days will not have the full attribution window! You will need
        to wait 7 days to have the full event attribution for 7 days. However this table can be used with 
        lower attribution window (e.g 1h)"""

        # 1 - Load all clicks on program/ person page in the last 7 days
        page_df = load_snowflake_table(self.spark, self.options, "dw.fact_page"). \
            filter(F.col("origin_section").isin(*self.rails)). \
            filter(F.col("page_name").isin(*['program', 'person'])).\
            filter(F.col('event_date') >= self.now - self.delta). \
            select("user_id", "timestamp", "page_name", "origin_page", "origin_section", "ORIGIN_COMPONENT_RANK",
                   "program_id", "channel_id", "person_id").persist()

        watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch"). \
            withColumn("event_name", F.lit("watch")). \
            filter(F.col('received_at') >= self.now - self.delta). \
            select("user_id", "event_name", "received_at", "program_id", "channel_id", "duration", F.lit(None).alias("person_id"))

        bookmark_follow_df = load_snowflake_table(self.spark, self.options, "dw.fact_bookmark_follow"). \
            filter(F.col('timestamp') >= self.now - self.delta). \
            select("user_id", "event_name", F.col("timestamp").alias("received_at"), "program_id", "channel_id", F.lit(None).alias("duration"), "person_id")

        # 3 - Build attribution df
        clic_watch_attribution_df = page_df.alias("page_df"). \
            join(watch_df.alias("watch_df"),
                 (page_df.user_id == watch_df.user_id) &
                 (page_df.program_id == watch_df.program_id) &
                 (page_df.channel_id == watch_df.channel_id) &
                 (watch_df.received_at >= page_df.timestamp) & (watch_df.received_at < page_df.timestamp + F.expr('INTERVAL 7 DAYS')),
                 how='inner'). \
            filter((F.col("received_at").cast("long") - F.col("timestamp").cast("long") > 2) | F.col('received_at').isNull()). \
            drop(watch_df.user_id). \
            select(page_df.user_id, page_df.timestamp, page_df.page_name, page_df.origin_page, page_df.origin_section,
                   "watch_df.*", page_df.ORIGIN_COMPONENT_RANK)

        clic_bookmarkfollow_attribution_df = page_df.alias("page_df"). \
            join(bookmark_follow_df.alias("bookmark_follow_df"),
                 (page_df.user_id == bookmark_follow_df.user_id) &
                 (((page_df.program_id == bookmark_follow_df.program_id) & (page_df.channel_id == bookmark_follow_df.channel_id)) |
                 (page_df.person_id == bookmark_follow_df.person_id)) &
                 (bookmark_follow_df.received_at >= page_df.timestamp) & (bookmark_follow_df.received_at < page_df.timestamp + F.expr('INTERVAL 7 DAYS')),
                 how='inner'). \
            drop(bookmark_follow_df.user_id). \
            select(page_df.user_id, page_df.timestamp, page_df.page_name, page_df.origin_page, page_df.origin_section,
                   "bookmark_follow_df.*", page_df.ORIGIN_COMPONENT_RANK)

        rail_all_event_attribution_df = clic_watch_attribution_df. \
            union(clic_bookmarkfollow_attribution_df)

        rail_clic_attribution_df = page_df. \
            join(rail_all_event_attribution_df, ["user_id", "timestamp", "page_name", "origin_page", "origin_section"],
                 "left"). \
            drop(rail_all_event_attribution_df.program_id). \
            drop(rail_all_event_attribution_df.channel_id). \
            drop(rail_all_event_attribution_df.person_id). \
            drop(rail_all_event_attribution_df.ORIGIN_COMPONENT_RANK)

        return rail_clic_attribution_df

    def delete_past_7days(self):

        conn = get_snowflake_connection(self.conf, 'PROD')

        query = f"""
        DELETE FROM DW.{self.TABLE_NAME}
        WHERE timestamp >= TIMESTAMPADD('day', -7, '{self.now}')
        """

        execute_snowflake_query(conn, query)


if __name__ == "__main__":
    job = RailClicAttributionJob()
    job.launch()
