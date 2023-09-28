import os
from unittest import TestCase
from unittest import mock
import pyspark.sql.functions as F

from databricks_jobs.jobs.fact_table_build_jobs.rail_clic_attribution_job.entrypoint import RailClicAttributionJob
from tests.unit.utils.mock_attribution import multiplex_mock

from pyspark.sql import SparkSession


class TestRailClicAttributionJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = RailClicAttributionJob()

    def test_build_rail_clic_attribution(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.rail_clic_attribution_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):

            df = self.job.build_rail_clic_attribution()
            rows = df.collect()
            # Expected :
            # 1 row: user_id=1 and program_id=2 with no event match
            # 2 rows: user_id=1 and program_id=1, with 2 clics and 1 bookmark event each
            # 4 rows: user_id=1 and program_id=1, with 2 clics and 2 watch event each
            # 3 rows: user_id=2 and program_id=1, with no watch event, 3 clics with 1 follow_person event & 2 bookmark event
            # Total 10 rows
            self.assertEqual(len(rows), 10)

            # Total events: 4 watch, 4 bookmark, 1 follow, and 1 no event
            def find_rows(rows, value, field):
                return [r for r in rows if r[field] == value] 
            events = df.groupby("event_name").agg(F.count(F.lit(1)).alias("cnt"))
            rows = events.collect()

            watch = find_rows(rows, "watch", "event_name")
            bookmark = find_rows(rows, "bookmark_added", "event_name")
            follow = find_rows(rows, "follow_person", "event_name")
            no_event = find_rows(rows, None, "event_name")

            self.assertEqual(watch[0].cnt, 4)
            self.assertEqual(bookmark[0].cnt, 4)
            self.assertEqual(follow[0].cnt, 1)
            self.assertEqual(no_event[0].cnt, 1)
