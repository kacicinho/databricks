import boto3

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.mysqlclient import MySqlClient
from databricks_jobs.jobs.utils.s3_to_rds_transfert_utils import db_lookup_list


class S3toRdsTransfertJob(Job):

    def __init__(self, *args, **kwargs):
        super(S3toRdsTransfertJob, self).__init__(*args, **kwargs)

        # Retreive conf vars
        self.env = self.conf.get("env", "DEV")
        self.snowflake_user = self.conf.get("user_name", "")
        self.snowflake_password = self.conf.get("password", "")
        self.db_host = self.conf.get("db_host", "")
        self.db_name = self.conf.get("db_name", "")
        self.db_username = self.conf.get("db_username", "")
        self.db_password = self.conf.get("db_password", "")
        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")
        self.bucket_name = self.conf.get("s3_output_bucket", "")

        # Retreive additional params
        self.now = self.parse_date_args()
        self.table_name = self.parse_table_name_args()

        # Set schema vars from table_name
        sf_to_rds_lookup = db_lookup_list.get_lookup(self.table_name)
        if sf_to_rds_lookup is not None:
            self.schema = sf_to_rds_lookup.get_schema
            self.primary_key = ",".join(self.schema["keys"])
            self.create_fields = ",".join([f"{field['name']} {field['type']} {field['constraint']}"
                                           for field in self.schema["fields"]])
            self.fields = ",".join([field["name"] for field in self.schema["fields"]])
            self.reco_field = "RATING" \
                if sf_to_rds_lookup.schema == "prog_best_rate_schema" else "RECOMMENDATIONS"
        else:
            raise Exception(f"{self.table_name} is not present in lookup")

        self.client = MySqlClient(self.db_host, self.db_username, self.db_password, self.db_name, self.logger)

        self.logger.info(f"Running on date : {self.now}")

    @staticmethod
    def add_more_args(p):
        """
        Additional args are required in order to get the tvspots
        """
        p.add_argument("--table_name", default=None, type=str)

    def parse_table_name_args(self):
        if self.parsed_args.table_name is not None:
            return self.parsed_args.table_name
        else:
            raise Exception("table_name parameter is not referenced")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 - List all files in bucket
        s3 = boto3.resource("s3", aws_access_key_id=self.aws_access_key,
                            aws_secret_access_key=self.aws_secret_key,
                            region_name=self.region_name)

        filenames = self.list_s3_filename(s3, self.bucket_name, self.table_name)

        # 2 - Iterate through each file of folder
        for filename in filenames:
            self.logger.info(f"Working on : {self.bucket_name}:{self.table_name}:{filename}")
            # Create RDS target table if not exists
            self.create_target_table()
            # Upload records to RDS target table
            self.upload_new_data_target_table(filename)

        # 3 - Delete old records from RDS table
        self.delete_old_reco()

        # 4 - Cleanup the data on the bucket
        for key in s3.Bucket(self.bucket_name).objects.filter(Prefix=f"{self.table_name}/").all():
            key.delete()

    def list_s3_filename(self, s3, bucket_name, prefix_filter):
        all_objects = list(s3.Bucket(bucket_name).objects.filter(Prefix=f"{prefix_filter}/").all())
        return [file.key for file in all_objects]

    def create_target_table(self):
        """Create RDS target table if not exists"""
        query = f"""
            create TABLE IF NOT EXISTS {self.table_name} (
            {self.create_fields}
            ,primary key ({self.primary_key})
            )
        """
        self.client.run_query((query,))

    def upload_new_data_target_table(self, filename):
        """Create temporary table, load new records into it then inserts data to
        target table if not exists, else update"""

        create_temp_table_query = f"""CREATE TEMPORARY TABLE temp LIKE {self.table_name}"""

        load_query = f""" 
            LOAD DATA FROM S3 's3://{self.bucket_name}/{filename}'
            INTO TABLE temp
            FIELDS 
                TERMINATED BY ';'
                OPTIONALLY ENCLOSED BY '"'
            LINES 
                TERMINATED BY '\n'
            ({self.fields})"""

        insert_temp_to_target_query = f"""REPLACE INTO {self.table_name}
        SELECT * FROM temp"""

        query_op = (create_temp_table_query, load_query, insert_temp_to_target_query)
        self.client.run_query(query_op)
        self.logger.info(f"Succesfully uploaded {filename}")

    def delete_old_reco(self):
        """Delete old records from target table"""
        delete_old_reco_query = f"""
        DELETE
        FROM {self.table_name}
        WHERE UPDATE_DATE < '{self.now}'
        """
        self.client.run_query((delete_old_reco_query,))
        self.logger.info("Successfully deleted old records")


if __name__ == "__main__":
    job = S3toRdsTransfertJob()
    job.launch()
