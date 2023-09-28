import os

import boto3
import botocore
from pyspark.sql import functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.data_quality_utils import QualityLog, \
    agg_dict_to_expr, send_slack_notif_from_alogs, \
    send_metrics_from_qlog
from databricks_jobs.jobs.utils.databricks_to_s3 import DatabricksToS3
from databricks_jobs.jobs.utils.log_db import LogPath
from databricks_jobs.jobs.utils.utils import get_snowflake_options, \
    dump_dataframe_to_csv, csv_to_dataframe, load_snowflake_table


class AffinityS3ExportsJob(Job):

    def __init__(self, *args, **kwargs):
        super(AffinityS3ExportsJob, self).__init__(*args, **kwargs)

        self.slack_token = self.conf.get("slack_token", "")
        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")
        self.output_bucket = self.conf.get("s3_output_bucket", "")
        self.mount_name = "/mnt/s3segments"
        self.dump_folder = "/segmentation"
        self.job_name = 'AFFINITY S3 EXPORT JOB'
        self.alert_on = self.parse_alert_on_args()

        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, 'PROD', **{"keep_column_case": "on"})
        self.logger.info(f"Running on date : {self.now}")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Exports Affinity segments to S3
        """
        alogs = []
        self.logger.info("Launching Affinity S3 Exports job")
        with DatabricksToS3(self.dbutils, self.mount_name, self.aws_access_key,
                            self.aws_secret_key, self.output_bucket) as uploader:

            # Upload affinity segments to s3
            for (table_name, csv_name, metric_name) in [("DW.FACT_AUDIENCE", "latest", "AFFINITY_SEGMENT"),
                                                        ("DW.FACT_AUDIENCE_AFF_REGIE_FLAG", "data_regie",
                                                         "REGIE_SEGMENT")]:
                affinity_segment_csv_path = self.prepare_affinity_segment_csv(table_name, csv_name)
                uploader.dump_daily_table(
                    affinity_segment_csv_path, '', '', '', prefix='', identifier=csv_name)

                # Send volume log metrics to cloudwatch
                try:
                    cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id=self.aws_access_key,
                                                     aws_secret_access_key=self.aws_secret_key,
                                                     region_name=self.region_name)

                    affinity_path = LogPath.from_args(uploader.bucket_local_path, "{}.csv".format(csv_name)).path
                    df = csv_to_dataframe(self.spark, affinity_path)
                    alogs += self.send_cloudwatch_metrics(cloudwatch_client, df, self.job_name, metric_name)
                except botocore.exceptions.ClientError as e:
                    self.logger.error(str(e))

        # Finally send logs to slack if alert/warning
        send_slack_notif_from_alogs(alogs, self.slack_token, self.alert_on)

    def prepare_affinity_segment_csv(self, table_name="DW.FACT_AUDIENCE", csv_name="latest"):
        """
        Prepare new affinity segments
        """
        fact_audience_df = load_snowflake_table(self.spark, self.options, table_name)

        csv_dir_path = os.path.join(self.dump_folder, csv_name)
        temporary_csv_path = dump_dataframe_to_csv(self.dbutils, fact_audience_df, csv_dir_path)
        return temporary_csv_path

    def send_cloudwatch_metrics(self, cloudwatch_client, df, job_name, metric_name="AFFINITY_SEGMENT"):

        exprs = agg_dict_to_expr({'USER_ID': {'user_id|none|count': F.count}})
        quality_log_list = [QualityLog(self.options, 'DATA_S3', metric_name,
                                       self.now, True, exprs, None, None, None, df)]

        _, alogs = send_metrics_from_qlog(cloudwatch_client, self.spark, quality_log_list, job_name)

        return alogs


if __name__ == "__main__":
    job = AffinityS3ExportsJob()
    job.launch()
