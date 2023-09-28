import os
from unittest import TestCase
from unittest import mock

from databricks_jobs.jobs.fact_table_build_jobs.program_clic_attribution_job.entrypoint import ProgramClicAttributionJob
from tests.unit.utils.mock_program_attribution import multiplex_mock

from pyspark.sql import SparkSession


class TestProgramClicAttributionJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = ProgramClicAttributionJob()

    def test_build_program_clic_attribution(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.program_clic_attribution_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):

            df = self.job.build_program_clic_attribution()
            rows = df.collect()

            def find_rows(rows, value, field):
                return [r for r in rows if r[field] == value] 

            # Expected :
            # 6 rows
            self.assertEqual(len(rows), 6)

            # 3 sim_timestamp null
            sim_timestamps = find_rows(rows, None, "next_prog")
            self.assertEqual(len(sim_timestamps), 3)

            # sim_timestamp for program_id 8 should be the latest timestamp of program_id 12
            prog_8_row = find_rows(rows, 8, "program_id")
            prog_12_row = find_rows(rows, 12, "program_id")
            latest_timestamp = max([x.timestamp for x in prog_12_row])
            prog_12_latest_row = [x for x in prog_12_row if x.timestamp == latest_timestamp]

            self.assertEqual(prog_8_row[0].timestamp, prog_12_latest_row[0].next_prog.timestamp)
            self.assertEqual(prog_8_row[0].program_id, prog_12_latest_row[0].next_prog.program_id)
