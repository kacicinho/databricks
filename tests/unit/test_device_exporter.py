import datetime
import os
import shutil
from unittest import TestCase
from unittest import mock

import boto3
from moto import mock_cloudwatch

from databricks_jobs.jobs.third_party_export_jobs.device_exporter.entrypoint import DeviceExportJob
from tests.unit.utils.mocks import build_user_raw_mock, create_spark_df_from_data, multiplex_mock


class TestDeviceExporterJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        ssm_mock = mock.Mock()
        ssm_mock.get_parameter = lambda *args, **kwargs: {"Parameter": {"Value": ""}}

        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.device_exporter.entrypoint.boto3",
                                 client=lambda *args, **kwargs: ssm_mock):
            cls.job = DeviceExportJob()

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
    def test_device_ids_build(self):
        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.device_exporter.entrypoint",
                                 load_snowflake_query_df=lambda spark, options, query: build_user_raw_mock(spark),
                                 load_snowflake_table=multiplex_mock):
            now = datetime.date.today()
            ids = list(range(10))
            data = {"USER_ID": ids, "ADVERTISING_ID": ids, "IP": ["192.168.0.1" for _ in ids],
                    "TIMESTAMP": [now - datetime.timedelta(days=1) for _ in ids],
                    "DEVICE_TYPE": ["apple" if i % 2 == 0 else "android" for i in ids]}
            df = create_spark_df_from_data(self.job.spark, data)
            df_out = self.job.join_email_hash_and_ip(df)
            self.assertSetEqual(set(df_out.columns),
                                {"ip", "e_s", "silo", "ts", "ADVERTISING_ID", "USER_ID", "DEVICE_TYPE"})

    @mock_cloudwatch
    def test_end_2_end(self):
        os.mkdir("./mount/1510")
        os.mkdir("./mount/1510/1")
        client = boto3.client("cloudwatch", region_name="eu-west-1")
        now = datetime.date.today()
        job_event_date = now - datetime.timedelta(days=1)
        ids = list(range(10))
        data = {"USER_ID": ids, "ADVERTISING_ID": ids, "IP": ["192.168.0.1" for _ in ids],
                "TIMESTAMP": [job_event_date for _ in ids],
                "DEVICE_TYPE": ["ios" if i % 2 == 0 else "android" for i in ids]}
        df = create_spark_df_from_data(self.job.spark, data)
        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.device_exporter.entrypoint",
                                 load_snowflake_query_df=lambda spark, options, query: build_user_raw_mock(spark),
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.object(self.job, "build_advertiser_id_table", lambda *args, **kwargs: df):
                with mock.patch.object(self.job, "filter_optins", lambda x, *args, **kwargs: x.drop("USER_ID")):
                    with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.device_exporter."
                                             "entrypoint.boto3",
                                             client=lambda *args, **kwargs: client):
                        self.job.launch()

        timestamp = int(datetime.datetime(now.year, now.month, now.day).timestamp())
        android_path = f"./mount/1510/1/molotov_FR_aaid_{timestamp}.csv"
        apple_path = f"./mount/1510/1/molotov_FR_idfa_{timestamp}.csv"

        for id_col, path in [("aaid", android_path), ("idfa", apple_path)]:
            self.assertTrue(os.path.exists(path), f"{path} not found : {os.listdir('./mount/1510/1')}")
            with open(path) as f:
                line = f.readline()
            cols = list(map(lambda x: x.split("=")[0], line.strip().split(",")))
            values = list(map(lambda x: x.split("=")[1], line.strip().split(",")))
            self.assertSetEqual(set(cols), {"e_s", "ip", "ts", "silo", id_col})
            timestamp = int(datetime.datetime(job_event_date.year, job_event_date.month, job_event_date.day).timestamp())
            self.assertIn(str(timestamp), values)
