from unittest import mock
import os
from unittest import TestCase
from datetime import datetime
from tests.unit.utils.mocks import multiplex_mock

from databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.entrypoint import FactSearchJob
from tests.unit.utils.mocks import create_spark_df_from_data
from pyspark.sql import SparkSession


class TestFactSearch(TestCase):

    @classmethod
    def setUp(self) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        self.job = FactSearchJob()

    def test_search_log_creation(self):
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name)):
            df = self.job.load_search_queries_df()
            rows = df.collect()
            self.assertEqual(len(rows), 6)

    def test_search_log_filtering(self):
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name)):
            df = self.job.load_search_queries_df()
            df2 = self.job.filter_non_maximal_queries(df)
            rows = df2.collect()

            # All users have the same two queries : a and ab
            self.assertEqual(len(rows), 3)
            self.assertSetEqual({r.USER_ID for r in rows}, {0, 1, 2})

    def test_end_to_end(self):
        uids = range(10)
        ts = datetime.now()
        data = {
            "USER_ID": uids,
            "timestamp": [ts for _ in uids],
            "device_type": ["tv" for _ in uids],
            "search_query": ["tuch" for _ in uids],
            "search_result_type": ["program" for _ in uids],
            "entity_name": ["Les Tuches" for _ in uids],
            "entity_id": [0 for _ in uids]
        }
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name),
                                 load_snowflake_query_df=lambda spark, options, query:
                                 create_spark_df_from_data(spark, data),
                                 write_df_to_snowflake=lambda df, *args: df.show()):
            self.job.launch()
