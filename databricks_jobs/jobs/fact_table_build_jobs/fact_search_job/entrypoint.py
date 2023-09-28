from datetime import timedelta

from pyspark.sql import functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.query_substring_remover import search_filtering
from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_snowflake_query_df, write_df_to_snowflake, \
    get_snowflake_options


class FactSearchJob(Job):

    def __init__(self, *args, **kwargs):
        super(FactSearchJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args() - timedelta(days=1)
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", sf_schema="PUBLIC")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """

        """
        # 1 - The search query set
        search_df = self.load_search_queries_df()
        filtered_search_df = self.filter_non_maximal_queries(search_df)
        # 2 - The result set
        clicks_df = self.load_search_click_events()
        # 3 - the final table is the combination of searches and results
        final_df = filtered_search_df. \
            withColumn("entity_id", F.lit(None)). \
            withColumn("entity_name", F.lit(None)). \
            withColumn("search_result_type", F.lit(None)). \
            select("USER_ID", "TIMESTAMP", "DEVICE_TYPE", "SEARCH_QUERY",
                   "SEARCH_RESULT_TYPE", "ENTITY_NAME", "ENTITY_ID"). \
            union(clicks_df)

        write_df_to_snowflake(final_df, self.write_options, "DW.fact_search", "append")

    def load_search_queries_df(self):
        search_df = load_snowflake_table(self.spark, self.options, "DW.fact_page")

        parsed_search_df = search_df. \
            where(F.col("EVENT_DATE") == self.now). \
            select("USER_ID", "TIMESTAMP", F.col("QUERY").alias("search_query"),
                   F.col("ACTION_DEVICE_TYPE").alias("device_type"))

        return parsed_search_df. \
            where(F.col("search_query").isNotNull())

    def load_search_click_events(self):
        query = \
            f"""
              select user_id, timestamp, 
                ACTION_DEVICE_TYPE as device_type, 
                PROPERTIES_JSON:search_result_query::varchar as search_query, 
                PROPERTIES_JSON:search_result_type::varchar as search_result_type, 
                coalesce(PROPERTIES_JSON:program_title::varchar, 
                         PROPERTIES_JSON:channel_name::varchar, 
                         PROPERTIES_JSON:full_name::varchar) as entity_name,
                coalesce(PROPERTIES_JSON:program_id, 
                         PROPERTIES_JSON:channel_id, 
                         PROPERTIES_JSON:kind_id,
                         PROPERTIES_JSON:person_id) as entity_id
              from SEGMENT.SNOWPIPE_ALL
              where event_name like '%clicked%'
              and event_date = '{str(self.now)}'
              and PROPERTIES_JSON:search_result_query::varchar is not null
            """
        return load_snowflake_query_df(self.spark, self.options, query). \
            select("USER_ID", "TIMESTAMP", "DEVICE_TYPE", "SEARCH_QUERY",
                   "SEARCH_RESULT_TYPE", "ENTITY_NAME", "ENTITY_ID")

    def filter_non_maximal_queries(self, df, cols=("USER_ID", "device_type")):
        """
        The goal of this function is to apply the UDF doing the prefix finding on the right time windows.
        """
        filtered_searches_df = df. \
            groupBy(*cols). \
            agg(F.collect_list(F.struct("search_query", "timestamp")).alias("searches")). \
            withColumn("searches", search_filtering("searches"))

        return filtered_searches_df. \
            withColumn("searches", F.explode("searches")). \
            select(*cols, "searches.search_query", "searches.timestamp")


if __name__ == "__main__":
    job = FactSearchJob()
    job.launch()
