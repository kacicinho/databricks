import datetime
import os


class DatabricksToS3:

    def __init__(self, dbutils, dbfs_mount_name, s3_access_key, s3_secret_key, s3_bucket):
        s3_encoded_secret_key = s3_secret_key.replace("/", "%2F")

        # S3 related path construction
        self.dbutils = dbutils
        self.S3_BUCKET_URL = "s3a://{}:{}@{}".format(s3_access_key, s3_encoded_secret_key, s3_bucket)
        self.now = datetime.datetime.now()
        self.bucket_local_path = dbfs_mount_name

    def __enter__(self):
        self.dbutils.fs.mount(self.S3_BUCKET_URL, self.bucket_local_path)
        return self

    def __exit__(self, *args):
        self.dbutils.fs.unmount(self.bucket_local_path)

    def from_dbfs_to_s3(self, csv_dump_path, s3_suffix_path, filename, use_mkdir=True):
        """
        We copy a file from dbfs to another location on dbfs.
        This new location will be a mounted s3 bucket

        :param csv_dump_path: file to copy from dbfs like "/dbfs/FileStore/data_monetization/latest.csv"
        :param s3_suffix_path: path on S3 will be S3_BUCKET/{s3_suffix_path}/filename
        :param filename: final name of `csv_dump_path` once on s3
        :param use_mkdir: can be disabled to avoid scanning sub directory with restricted rights
        :return:
        """
        target_dir = f"{self.bucket_local_path}/{s3_suffix_path}"
        if use_mkdir:
            self.dbutils.fs.mkdirs(target_dir)
        self.dbutils.fs.mv(csv_dump_path, f"{target_dir}/{filename}")

    def dump_daily_table(self, current_dump, client, data_type, date, prefix="dump_", identifier="basic"):
        """
        Expected s3 path /mtv-onboarding/{client_name or common}/{data_type}/{date}/dump_{additional_identifier}.csv
        We format args properly for from_dbfs_to_s3 function
        """
        path_segments = [str(x) for x in (client, data_type, date) if x]
        final_s3_path = os.path.join(*path_segments) if len(path_segments) > 0 else ""
        self.from_dbfs_to_s3(current_dump, final_s3_path, prefix + identifier + ".csv")
