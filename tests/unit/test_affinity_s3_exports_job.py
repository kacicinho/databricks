from unittest import TestCase
from unittest import mock
import os
import shutil
import boto3
from moto import mock_cloudwatch

from databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs.affinity_s3_exports_job.entrypoint import AffinityS3ExportsJob
from tests.unit.utils.mocks import mock_fact_audience
from databricks_jobs.jobs.utils.utils import csv_to_dataframe
from pyspark.sql import SparkSession


class TestAffinityS3ExportsJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = AffinityS3ExportsJob()

        cls.mount_dir = "./mount"
        cls.exports_dir = "./exports"
        cls.job.mount_name = cls.mount_dir
        cls.job.dump_folder = cls.exports_dir
        if not os.path.exists(cls.mount_dir):
            os.mkdir(cls.mount_dir)
            os.mkdir(cls.exports_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.mount_dir):
            shutil.rmtree(cls.mount_dir)
        if os.path.exists(cls.exports_dir):
            shutil.rmtree(cls.exports_dir)

    @mock_cloudwatch
    def test_affinity_segment_extract(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_s3_exports_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, query: mock_fact_audience(spark)):

            cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id='xxxx',
                                             aws_secret_access_key='xxxx',
                                             region_name='us-east-1')

            # Check that the file has been sent at the right place
            csv_affinity_segment_path = self.job.prepare_affinity_segment_csv()
            self.assertTrue(os.path.exists(csv_affinity_segment_path))

            # Read the file and check that we have the right number of users
            df_affinity_segment = csv_to_dataframe(self.job.spark, csv_affinity_segment_path)
            self.assertEqual(df_affinity_segment.count(), 5)

            # Send metrics to cloudwatch
            self.job.send_cloudwatch_metrics(cloudwatch_client, df_affinity_segment,
                                             'AFFINITY S3 EXPORT JOB', 'AFFINITY_SEGMENT')

            # Check sent metrics
            sent_metrics = cloudwatch_client.list_metrics()['Metrics']
            self.assertEqual(len(sent_metrics), 1)
