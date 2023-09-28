from unittest import TestCase
from unittest import mock
import os
import shutil
import boto3

from moto import mock_cloudwatch
from databricks_jobs.jobs.third_party_export_jobs.onboarding_job.entrypoint import OnboardingJob
from tests.unit.utils.mocks import multiplex_mock, build_user_raw_mock, \
    multiplex_query_mock
from databricks_jobs.db_common import prepare_new_user_optout_csv
from pyspark.sql import SparkSession


class TestOnboardingJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = OnboardingJob()

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
    def test_onboarding_user_extract(self):
        with mock.patch("databricks_jobs.jobs.third_party_export_jobs.onboarding_job.entrypoint.load_snowflake_query_df",
                        new=lambda spark, options, query: build_user_raw_mock(spark)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_query_df=lambda spark, options, query: build_user_raw_mock(spark)):

                cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id='xxxx',
                                                 aws_secret_access_key='xxxx',
                                                 region_name='us-east-1')

                # Check that the file has been sent at the right place
                csv_optin_path = self.job.prepare_new_user_optin_csv("2021-04-01", "2021-04-02")
                csv_optout_path = prepare_new_user_optout_csv(self.job.spark, self.job.options,
                                                              self.job.dbutils, self.job.dump_folder,
                                                              "2021-04-01", "2021-04-02")
            self.assertTrue(os.path.exists(csv_optin_path))
            self.assertTrue(os.path.exists(csv_optout_path))

            alogs = self.job.metric_check_from_csv_dump(cloudwatch_client, csv_optin_path, 'user_optin_identification')
            alogs += self.job.metric_check_from_csv_dump(cloudwatch_client, csv_optin_path, 'user_optout')
            # No alerts are done user extract
            self.assertEqual(len(alogs), 0)

            # Check sent metrics
            sent_metrics = cloudwatch_client.list_metrics()['Metrics']
            self.assertEqual(len(sent_metrics), 2)

    @mock_cloudwatch
    def test_onboarding_watch_extract(self):
        with mock.patch("databricks_jobs.jobs.third_party_export_jobs.onboarding_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.utils.onboarding_utils.load_snowflake_query_df",
                            new=multiplex_query_mock):

                cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id='xxxx',
                                                 aws_secret_access_key='xxxx',
                                                 region_name='us-east-1')
                # Check that the file has been sent at the right place
                csv_path = self.job.prepare_watch_csv()
                self.assertTrue(os.path.exists(csv_path))

                self.job.metric_check_from_csv_dump(cloudwatch_client, csv_path, 'watch_consumption')

                # Check sent metrics
                sent_metrics = cloudwatch_client.list_metrics()['Metrics']
                self.assertEqual(len(sent_metrics), 10)
