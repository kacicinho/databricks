import os
from datetime import datetime, timedelta

import boto3
from pyspark.sql import Window
from pyspark.sql import functions as F

from databricks_jobs.common import Job
from databricks_jobs.db_common import prepare_new_user_optout_csv
from databricks_jobs.jobs.utils.data_quality_utils import QualityLog, \
    agg_dict_to_expr, send_slack_notif_from_alogs, \
    send_metrics_from_qlog
from databricks_jobs.jobs.utils.databricks_to_s3 import DatabricksToS3
from databricks_jobs.jobs.utils.log_db import LogDB, LogPath
from databricks_jobs.jobs.utils.onboarding_utils import get_watch_consent_df, prepare_lookup_channel_csv
from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_snowflake_query_df, \
    get_pypsark_hash_function, dump_dataframe_to_csv, unpivot_fact_audience, trunc_datetime, \
    csv_to_dataframe, get_snowflake_options


class OnboardingJob(Job):

    def __init__(self, *args, **kwargs):
        super(OnboardingJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.logger.info(f"Running on date : {self.now}")
        self.job_name = "ONBOARDING JOB"
        self.alert_on = self.parse_alert_on_args()

        self.watch_duration_threshold = 300

        self.paths = ["user_optin_identification", "user_optout", "watch_consumption", "lookup_feeds"]
        self.client = "liveramp"

        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")
        self.output_bucket = self.conf.get("s3_output_bucket", "")
        self.slack_token = self.conf.get("slack_token", "")
        self.mount_name = "/mnt/s3-dev-onboarding-exports"
        self.dump_folder = "/exports"

        self.options = get_snowflake_options(self.conf, "PROD", sf_schema="public")
        self.write_options = get_snowflake_options(self.conf, "PROD", sf_schema="ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        alogs = []

        # Init Cloudwatch Client
        cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id=self.aws_access_key,
                                         aws_secret_access_key=self.aws_secret_key,
                                         region_name=self.region_name)

        with DatabricksToS3(self.dbutils, self.mount_name, self.aws_access_key,
                            self.aws_secret_key, self.output_bucket) as uploader:

            # 1 - Get list of all dump dates for each data feeds
            log_s3_bucket = {path: LogDB.parse_dbfs_path(
                self.dbutils, f"{self.mount_name}/{self.client}/{path}") for path in self.paths}
            user_optout_last_dump_date = log_s3_bucket["user_optout"].last_dump_date
            user_optin_last_dump_date = log_s3_bucket["user_optin_identification"].last_dump_date

            # 2 - Extract new montly opted-in/out users since last dump after the 2nd of each month.
            # If no dumps has been done yet, the first dump will take all the users since the
            # USER_START_BACKUP_DAY defined in log_db.py
            # 2.1 - Optin users
            if self.now.day > 1 and trunc_datetime(self.now) > trunc_datetime(user_optin_last_dump_date):
                new_user_optin_csv_path = self.prepare_new_user_optin_csv(
                    user_optin_last_dump_date, datetime.strftime(self.now, '%Y-%m-%d'))
                uploader.dump_daily_table(
                    new_user_optin_csv_path, self.client, "user_optin_identification",
                    self.now, identifier="user_optin_identification")
                # Send volume log metrics to cloudwatch
                optin_path = LogPath.from_args(uploader.bucket_local_path, self.client,
                                               "user_optin_identification", str(self.now),
                                               "dump_user_optin_identification.csv").path
                alogs += self.metric_check_from_csv_dump(cloudwatch_client, optin_path, 'user_optin_identification')

            # 2.2 - Optout users
            if self.now.day > 1 and trunc_datetime(self.now) > trunc_datetime(user_optout_last_dump_date):
                new_user_optout_csv_path = prepare_new_user_optout_csv(
                    self.spark, self.options, self.dbutils, self.dump_folder,
                    user_optout_last_dump_date, datetime.strftime(self.now, '%Y-%m-%d'))
                uploader.dump_daily_table(
                    new_user_optout_csv_path, self.client, "user_optout",
                    self.now, identifier="user_optout")
                # Send user volume log metrics to cloudwatch
                optout_path = LogPath.from_args(uploader.bucket_local_path, self.client,
                                                "user_optout", str(self.now), "dump_user_optout.csv").path
                alogs += self.metric_check_from_csv_dump(cloudwatch_client, optout_path, 'user_optout')

            # 3 - Extract daily watch for optin users and send csv to S3
            watch_csv_path = self.prepare_watch_csv()
            uploader.dump_daily_table(
                watch_csv_path, self.client, "watch_consumption",
                self.now, identifier="watch_consumption")
            # Send watch volume log metrics to cloudwatch
            watch_path = LogPath.from_args(uploader.bucket_local_path, self.client,
                                           "watch_consumption", str(self.now),
                                           "dump_watch_consumption.csv").path
            alogs += self.metric_check_from_csv_dump(cloudwatch_client, watch_path, 'watch_consumption')

            # 4 - Lookup descriptions for channel_ids (dumped only if no dumps before)
            if len(log_s3_bucket["lookup_feeds"].db) == 0:
                lookup_channel_csv_path = prepare_lookup_channel_csv(self.spark, self.options,
                                                                     self.dump_folder, self.dbutils)
                uploader.dump_daily_table(lookup_channel_csv_path, self.client, "lookup_feeds",
                                          self.now, identifier="channels")

        # 5 - Finally send logs to slack if alert/warning
        send_slack_notif_from_alogs(alogs, self.slack_token, self.alert_on)

    def prepare_new_user_optin_csv(self, start_date, end_date):
        """
        Query new opted-in users since last dump with identification of users.
        """

        # 1 - Query new optin users since last dump
        query_new_user_optin = \
            """
            SELECT
                us.ID,
                EMAIL,
                FIRST_NAME,
                LAST_NAME,
                BIRTHDAY
            FROM BACKEND.USER_RAW us
            JOIN DW.FACT_CMP_USER_CONSENTS cs
            ON
                us.ID = cs.USER_ID
            WHERE
                us.CREATED_AT >= '{0}' and us.CREATED_AT < '{1}' AND
                TRANSFORM_SUBSCRIPTION_EMAIL_MOLOTOV(EMAIL) = FALSE AND
                CONSENT_ENABLED(purposes:enabled, array_construct(1,2,3,4,5,6,7,8,9,10)) AND
                CONSENT_ENABLED(specialfeatures:enabled, array_construct(1,2)) AND
                CONSENT_ENABLED(custompurposes:enabled, array_construct(1,2)) AND
                CONSENT_ENABLED(vendorsconsent:enabled, array_construct(97))
            """.format(start_date, end_date)
        user_new_optin_df = load_snowflake_query_df(self.spark, self.options, query_new_user_optin)

        # 3 - Extract
        final_df = user_new_optin_df. \
            withColumn("external_id", get_pypsark_hash_function()). \
            select("external_id", "EMAIL", "FIRST_NAME", "LAST_NAME", "BIRTHDAY")

        # 4 - Dump the result to csv
        csv_dir_path = os.path.join(self.dump_folder, "user_optin_identification_dump")
        temporary_csv_path = dump_dataframe_to_csv(self.dbutils, final_df, csv_dir_path)
        return temporary_csv_path

    def prepare_watch_csv(self):
        """
        Query user watch consumption on the provided date.
        """
        watch_consent_df = get_watch_consent_df(self.spark, self.options, (self.now - timedelta(days=1)))

        # Load fact audience aff from today
        fa_raw = load_snowflake_table(self.spark, self.options, "dw.fact_audience_aff_only")
        fact_audience_from_today_df = fa_raw.where(F.col('"date"') == self.now)

        # Unpivot the table to have a format like (USER_ID, AFFINITY)
        non_category_columns = {"USER_ID", '"date"'}
        category_columns = set(fa_raw.columns).difference(non_category_columns)
        flat_audience_df = \
            unpivot_fact_audience(fact_audience_from_today_df, category_columns, non_category_columns). \
            select("USER_ID", "category", "score")

        # Get the best matching affinity segment per user
        w = Window.partitionBy('USER_ID').orderBy(F.desc("score"))
        flat_audience_df = flat_audience_df. \
            withColumn('rank_score',
                       F.row_number().over(w))

        flat_audience_df = flat_audience_df. \
            filter(flat_audience_df.rank_score == 1). \
            drop('rank_score')

        # Add category col to watch df
        watch_cols = watch_consent_df.columns
        watch_cols.remove("USER_ID")

        watch_joined_affinity_df = watch_consent_df. \
            join(flat_audience_df, watch_consent_df.USER_ID == flat_audience_df.USER_ID). \
            select(watch_consent_df.USER_ID, *watch_cols, "category")

        # Select desired cols and keep only users with duration < 8 hours.
        watch_joined_affinity_df = watch_joined_affinity_df. \
            select("USER_ID",
                   "CHANNEL_ID",
                   "WATCH_START",
                   "WATCH_END",
                   "category"). \
            withColumnRenamed("USER_ID", "ID"). \
            withColumnRenamed("category", "SEGMENT"). \
            withColumn("DURATION", F.unix_timestamp(F.col("WATCH_END")) - F.unix_timestamp(F.col("WATCH_START"))). \
            filter(F.col("DURATION") < 8 * 60 * 60)

        # Filter only on consent user
        final_df = watch_joined_affinity_df. \
            withColumn("external_id", get_pypsark_hash_function()). \
            select("external_id", "channel_id", "watch_start", "watch_end", "segment")

        # Dump the result to csv
        csv_dir_path = os.path.join(self.dump_folder, "watch_consumption_dump")
        temporary_csv_path = dump_dataframe_to_csv(self.dbutils, final_df, csv_dir_path)
        return temporary_csv_path

    def build_quality_log_list(self, df, identifier):

        def to_qlog_list(agg_dict, identifier, df, alert, *additional_expr):
            exprs = agg_dict_to_expr(agg_dict)
            exprs.extend(additional_expr)
            return [QualityLog(self.options, self.client.upper(), identifier.upper(),
                               self.now, alert, exprs, None, None, None, df)]

        if identifier == 'watch_consumption':
            df = df.withColumn('DURATION',
                               F.unix_timestamp(F.col("WATCH_END")) - F.unix_timestamp(F.col("WATCH_START")))
            agg_dict = {
                'EXTERNAL_ID': {'external_id|none|count': F.count, 'external_id|none|countDistinct': F.countDistinct},
                'DURATION': {'duration|none|avg': F.mean, 'duration|none|sum': F.sum, 'duration|none|max': F.max,
                             'duration|none|min': F.min}
            }
            additional_expr = [F.expr('percentile(duration, array(0.1))')[0].alias('duration|none|10%_percentile'),
                               F.expr('percentile(duration, array(0.5))')[0].alias('duration|none|50%_percentile'),
                               F.expr('percentile(duration, array(0.75))')[0].alias('duration|none|75%_percentile'),
                               F.expr('percentile(duration, array(0.95))')[0].alias('duration|none|95%_percentile')
                               ]
            quality_log_list = to_qlog_list(agg_dict, identifier, df, True, *additional_expr)

        elif identifier in ('user_optin_identification', 'user_optout'):
            agg_dict = {'EXTERNAL_ID': {'external_id|none|count': F.count}}
            quality_log_list = to_qlog_list(agg_dict, identifier, df, False)
        else:
            raise Exception('The identifier provided is not correct.')
        return quality_log_list

    def metric_check_from_csv_dump(self, cloudwatch_client, csv_path, identifier):
        df = csv_to_dataframe(self.spark, csv_path)
        quality_log_list = self.build_quality_log_list(df, identifier)
        _, alogs = send_metrics_from_qlog(cloudwatch_client, self.spark, quality_log_list, self.job_name)
        return alogs


if __name__ == "__main__":
    job = OnboardingJob()
    job.launch()
