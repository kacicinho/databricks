from unittest import TestCase
from unittest import mock
import os
import tempfile
import boto3
from moto import mock_s3

from databricks_jobs.jobs.misc_jobs.s3_to_rds_transfert_job.entrypoint import S3toRdsTransfertJob


class TestS3toRdsTransfertJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch.multiple("databricks_jobs.jobs.misc_jobs.s3_to_rds_transfert_job."
                                 "entrypoint.S3toRdsTransfertJob",
                                 parse_table_name_args=lambda *args, **kwargs: "RECO_PROG_PROGS_META_LATEST"):
            cls.job = S3toRdsTransfertJob()

    @mock_s3
    def test_list_s3_objects_job(self):

        with tempfile.TemporaryDirectory() as d:
            dir_path = os.path.join(d)
            file_path = os.path.join(dir_path, "file.txt")

            with open(file_path, "w") as f:
                f.write("ok")

            conn = boto3.resource('s3', region_name='us-east-1')
            # We need to create the bucket since this is all in Moto's 'virtual' AWS account
            conn.create_bucket(Bucket='test')
            conn.meta.client.upload_file(file_path, 'test', 'my_folder/file.txt')
            filenames = self.job.list_s3_filename(conn, 'test', 'my_folder')
            self.assertEqual(filenames[0], 'my_folder/file.txt')
