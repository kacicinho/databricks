import os
from datetime import datetime, timedelta

import boto3
import pyspark.sql.functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.data_quality_utils import QualityLog, \
    send_metrics_from_qlog
from databricks_jobs.jobs.utils.databricks_to_s3 import DatabricksToS3
from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_snowflake_query_df, \
    get_snowflake_options


class DeviceExportJob(Job):

    def __init__(self, *args, **kwargs):
        super(DeviceExportJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.timestamp_now = datetime(self.now.year, self.now.month, self.now.day).timestamp()
        self.logger.info(f"Running on date : {self.now}")
        self.silo_id = 715881

        self.client = "liveramp-match-partner"

        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")
        self.ssm_client = boto3.client('ssm', aws_access_key_id=self.aws_access_key,
                                       aws_secret_access_key=self.aws_secret_key,
                                       region_name="eu-west-1")

        self.output_bucket = "com-liveramp-eu-customer-uploads"
        self.liveramp_access_key = "AKIAWW7ZHARILU2GT6BW"
        response = self.ssm_client.get_parameter(
            Name="/mtv/team_data/match_partner_key",
            WithDecryption=True
        )
        self.liveramp_secret_key = response["Parameter"]["Value"]
        self.slack_token = self.conf.get("slack_token", "")
        self.mount_name = "/mnt/s3-dev-device-exports"
        self.dump_folder = "/device-exports"
        self.table_name = "DEVICE_EXPORTER"

        self.options = get_snowflake_options(self.conf, "PROD", sf_schema="DW")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id=self.aws_access_key,
                                         aws_secret_access_key=self.aws_secret_key,
                                         region_name=self.region_name)

        with DatabricksToS3(self.dbutils, self.mount_name, self.liveramp_access_key, self.liveramp_secret_key,
                            self.output_bucket) as uploader:
            # Active devices exported every day
            # 1.1 - Get devices
            active_device_df = self.build_advertiser_id_table(self.now - timedelta(days=1), self.now)
            # 1.2 - Complete infos : IP, hashed email
            active_hash_device_df = self.join_email_hash_and_ip(active_device_df)
            # 1.3 - Keep only the optins
            export_df = self.filter_optins(active_hash_device_df).persist()
            # 1.4 - Do the final export
            for out_extension, filter_name in [("aaid", "android"), ("idfa", "ios")]:
                df = export_df. \
                    withColumnRenamed("ADVERTISING_ID", out_extension). \
                    where(F.lower(export_df.DEVICE_TYPE) == filter_name). \
                    drop("DEVICE_TYPE").\
                    persist()

                csv_dir_path = os.path.join(self.dump_folder, self.table_name, out_extension)
                fields = ["e_s", out_extension, "ip", "ts", "silo"]
                df.rdd.\
                    map(lambda x: ",".join(["{}={}".format(a, b) for a, b in zip(fields, [x[f] for f in fields])])).\
                    coalesce(1).\
                    saveAsTextFile(csv_dir_path)

                csv_path = csv_dir_path + "/part-00000"

                self.metric_check_from_df(cloudwatch_client, df, filter_name)
                # Expected path is  s3://com-liveramp-eu-customer-uploads/{cli_id}/{country_id}/*
                # rule was country_id = 2 ==> FR, 1 = GB (not used), but apparently export must be done on 1
                uploader.from_dbfs_to_s3(csv_path, "1510/1",
                                         f"molotov_FR_{out_extension}_{str(int(self.timestamp_now))}.csv",
                                         use_mkdir=False)
                uploader.dbutils.fs.rm(csv_dir_path, recurse=True)

    def build_advertiser_id_table(self, start_date, end_date):
        """
        We need all (USER_ID, ADVERTISING_ID) pairs and a corresponding IP and max_timestamp with them
        """
        query = f"""
        select USER_ID, ADVERTISING_ID, DEVICE_TYPE, IP, MAX(timestamp) as TIMESTAMP, count(TIMESTAMP) as cnt
        from SEGMENT.SNOWPIPE_ALL 
        where ADVERTISING_ID is not null and EVENT_DATE >= '{start_date}' and EVENT_DATE < '{end_date}' and  
        action_device_type in ('tablet', 'phone')
        group by USER_ID, ADVERTISING_ID, DEVICE_TYPE, IP
        qualify row_number() over(partition by USER_ID, ADVERTISING_ID order by cnt) = 1
        """
        return load_snowflake_query_df(self.spark, self.options, query)

    def join_email_hash_and_ip(self, df):
        """
        :param df: expected cols : USER_ID, ADVERTISING_ID, IP, TIMESTAMP

        e_s : sha256 of the email
        """
        email_df = load_snowflake_table(self.spark, self.options, "backend.user_raw")
        return df. \
            join(email_df, email_df.ID == df.USER_ID). \
            withColumn("e_s", F.sha2(F.lower("EMAIL"), 0)). \
            select("USER_ID", "e_s", "ADVERTISING_ID", F.col("IP").alias("ip"),
                   F.unix_timestamp("TIMESTAMP").alias("ts"), "DEVICE_TYPE"). \
            withColumn("silo", F.lit(self.silo_id))

    def filter_optins(self, df):
        """
        Filter out users without the right consents

        We go for the full acceptation in order to have an export
        """
        query = \
            """
            select USER_ID
            from DW.FACT_CMP_USER_CONSENTS
            join backend.user_raw
            on ID = USER_ID
            where
                TRANSFORM_SUBSCRIPTION_EMAIL_MOLOTOV(EMAIL) = FALSE AND
                CONSENT_ENABLED(purposes:enabled, array_construct(1,2,3,4,5,6,7,8,9,10)) AND
                CONSENT_ENABLED(specialfeatures:enabled, array_construct(1,2)) AND
                CONSENT_ENABLED(custompurposes:enabled, array_construct(1,2)) AND
                CONSENT_ENABLED(vendorsconsent:enabled, array_construct(97))
            """
        optin_df = load_snowflake_query_df(self.spark, self.options, query)
        return df. \
            join(optin_df, optin_df.USER_ID == df.USER_ID). \
            drop(optin_df.USER_ID). \
            drop(df.USER_ID)

    def metric_check_from_df(self, cloudwatch_client, df, device_name):
        quality_log_list = self.build_quality_log_list(df, device_name)
        _, alogs = send_metrics_from_qlog(cloudwatch_client, self.spark, quality_log_list, self.table_name)
        return alogs

    def build_quality_log_list(self, df, device_name):
        return [QualityLog(
            db_options=self.options,
            schema=self.client.upper(),
            table=self.table_name,
            date=self.now,
            alert=True,
            agg_dict=[F.count("*").alias("device_id|active|count")],
            group_by=None,
            filter_by=None,
            filter_name=device_name,
            df=df
        )]


if __name__ == "__main__":
    job = DeviceExportJob()
    job.launch()
