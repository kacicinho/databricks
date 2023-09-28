import os
from unittest import TestCase
from unittest import mock

from databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs.affinity_rfm_job.entrypoint import AffinityRFMJob
from pyspark.sql import SparkSession
from tests.unit.utils.mocks import multiplex_mock


class TestRFMSegmentJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = AffinityRFMJob()

    def test_compute_rfm_stats_df(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_rfm_job.entrypoint.load_snowflake_table",
                        new=multiplex_mock):
            rfm_stats_dfs = self.job.compute_rfm_stats_job()
            rfm7_df = rfm_stats_dfs['RFM07_DAILY_STATS']
            rows = rfm7_df.collect()

            # Check number of expected df
            self.assertEqual(len(rfm_stats_dfs), 2)
            # Check expected rows
            self.assertEqual(len(rows), 2)
            # Check example of expected output
            for r in rows:
                if r.RFM_CLUSTER == 'actifs':
                    self.assertEqual(r.percentage, 0.8)

    def test_add_rfm7_rfm28_to_fact_audience_job(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_rfm_job.entrypoint.load_snowflake_table",
                        new=multiplex_mock):
            fact_audience_df = self.job.add_rfm7_rfm28_to_fact_audience_job()
            cols = fact_audience_df.columns
            rows = fact_audience_df.collect()

            # check inactifs_new_reg well added
            for r in rows:
                if r.USER_ID == 99:
                    self.assertEqual(r.RFM7_CLUSTER, 'inactifs_new_reg')

            # Check number of cols
            self.assertEqual(len(cols), 49)

    def test_end_to_end(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_rfm_job.entrypoint.load_snowflake_table",
                        new=multiplex_mock):
            with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                            "affinity_rfm_job.entrypoint.write_df_to_snowflake",
                            new=lambda df, *args, **kwargs: df.show()):
                self.job.launch()
