import unittest
import sys
from databricks_jobs.jobs.misc_jobs.sample.entrypoint import SampleJob
from pyspark.sql import SparkSession


class TestArgParsing(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.master("local[1]").getOrCreate()
        sys.argv = ["python", "test.py", "--date", "2021-03-07"]
        self.job = SampleJob(spark=self.spark)

    def test_sample(self):
        self.assertTrue("date" in self.job.conf)
        self.assertEqual(self.job.conf["date"], "2021-03-07")
