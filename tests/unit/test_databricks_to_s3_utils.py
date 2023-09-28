import datetime
import os
import tempfile
from unittest import TestCase
from unittest import mock

from databricks_jobs.jobs.utils.databricks_to_s3 import DatabricksToS3
from databricks_jobs.jobs.utils.local_dbfs import LocalDbutils
from databricks_jobs.jobs.utils.log_db import LogDB
from tests.unit.utils.mock_local_dbfs import MockDbutils
from tests.unit.utils.test_utils import file_exists_mock


class TestDatabricksToS3Utils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dbutils = LocalDbutils()

    def test_s3_exporter_calls(self):
        m = MockDbutils()

        with DatabricksToS3(m, "/mnt/local_dbfs_path", "aws_access_key", "aws_secret_key",
                            "s3_bucket_path") as transfer_obj:
            transfer_obj.from_dbfs_to_s3("dump.csv", "sub/bucket", "result.csv")

            m.fs.mount.assert_called_once()
            m.fs.mkdirs.assert_called_once()
            m.fs.mv.assert_called_once()

            m.fs.mount.assert_called_with("s3a://aws_access_key:aws_secret_key@s3_bucket_path", "/mnt/local_dbfs_path")

    def test_s3_exporter_file_creation(self):
        with tempfile.TemporaryDirectory() as d:
            mount_path = os.path.join(d, "mount_location")
            with DatabricksToS3(self.dbutils, mount_path, "aws_access_key", "aws_secret_key",
                                "s3_bucket_path") as transfer_obj:
                # Prepare a file that will be transferred with the defined logic
                file_path = os.path.join(mount_path, "file.csv")
                with open(file_path, "w") as f:
                    f.write("ok")
                # Do the move logic
                transfer_obj.from_dbfs_to_s3(file_path, "sub/bucket", "result.csv")

                # Assert a file exists at the expected path
                final_path = os.path.join(mount_path, "sub/bucket", "result.csv")
                self.assertTrue(os.path.exists(final_path))

                # Assert we have the right content in moved file
                with open(final_path, "r") as f:
                    content = f.readline()
                self.assertTrue("ok" in content)

    def test_onboarding_dump_daily_table_location(self):
        with tempfile.TemporaryDirectory() as d:
            mount_path = os.path.join(d, "mount_location")
            with DatabricksToS3(self.dbutils, mount_path, "aws_access_key", "aws_secret_key",
                                "s3_bucket_path") as transfer_obj:
                # Prepare a file that will be transferred with the defined logic
                file_path = os.path.join(mount_path, "file.csv")
                with open(file_path, "w") as f:
                    f.write("ok")

                transfer_obj.dump_daily_table(file_path, client="liveramp", date="2021-03-01",
                                              data_type="user_identity", identifier="basic")

                final_path = os.path.exists(
                    os.path.join(mount_path, "liveramp", "user_identity", "2021-03-01", "dump_basic.csv")
                )
                self.assertTrue(os.path.exists(final_path))

    def test_affinity_segment_dump_daily_table_location(self):
        with tempfile.TemporaryDirectory() as d:
            mount_path = os.path.join(d, "mount_location")
            with DatabricksToS3(self.dbutils, mount_path, "aws_access_key", "aws_secret_key",
                                "s3_bucket_path") as transfer_obj:
                # Prepare a file that will be transferred with the defined logic
                file_path = os.path.join(mount_path, "file2.csv")
                with open(file_path, "w") as f:
                    f.write("ok")

                transfer_obj.dump_daily_table(file_path, client="toto", date="",
                                              data_type="", prefix="", identifier="latest")

                final_path = os.path.exists(
                    os.path.join(mount_path, "toto", "", "", "latest.csv")
                )
                self.assertTrue(os.path.exists(final_path))

    def test_onboarding_csv_dump_dates(self):
        with tempfile.TemporaryDirectory() as d:
            mount_path = os.path.join(d, "mount_location")
            file_path = os.path.join(mount_path, "file.csv")

            with DatabricksToS3(self.dbutils, mount_path, "aws_access_key", "aws_secret_key",
                                "s3_bucket_path") as transfer_obj:
                with mock.patch("databricks_jobs.jobs.utils.log_db.file_exists", file_exists_mock):
                    # Mulitple dumps and check dump dates
                    for date_ in ["2021-04-16", "2021-04-27"]:
                        with open(file_path, "w") as f:
                            f.write("ok")

                        transfer_obj.dump_daily_table(file_path, client="liveramp", date=date_,
                                                      data_type="user_optin_identification", identifier="basic")

                    user_optin_dump_dates = LogDB.parse_dbfs_path(self.dbutils,
                                                                  f"{mount_path}/liveramp/user_optin_identification")

                    data_feeds_dump_dates = LogDB.parse_dbfs_path(self.dbutils, f"{mount_path}/liveramp/data_feeds")

                    # Check empty list if no dumps
                    self.assertEqual(len(data_feeds_dump_dates.db), 0)

                    # Check last dump date
                    self.assertEqual(datetime.datetime(2021, 4, 27).date(), user_optin_dump_dates.last_dump_date)

                    # Check first dump date
                    self.assertEqual(datetime.datetime(2021, 4, 16).date(), user_optin_dump_dates.first_dump_date)

                    # Check missing dates
                    self.assertIn(datetime.datetime(2021, 4, 17).date(), user_optin_dump_dates.get_missing_date)
            # Check unmount succesfully
            self.assertFalse(os.path.exists(file_path))
