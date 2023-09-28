from datetime import timedelta

import boto3
import pyspark.sql.functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.affinities import AFFINITIES
from databricks_jobs.jobs.utils.data_quality_conf import common_ml_tables, common_backend_tables
from databricks_jobs.jobs.utils.data_quality_utils import QualityLog, AlertLog, \
    agg_dict_to_expr, extract_ml_metrics, compare_ml_value, common_agg_dict_expr, \
    send_slack_notif_from_alogs, send_metrics_from_qlog
from databricks_jobs.jobs.utils.tcfv2_utils import is_consent
from databricks_jobs.jobs.utils.utils import get_snowflake_options, get_mysql_options


class DataQualityJob(Job):

    def __init__(self, *args, **kwargs):
        super(DataQualityJob, self).__init__(*args, **kwargs)

        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")
        self.slack_token = self.conf.get("slack_token", "")
        self.snowflake_options = get_snowflake_options(self.conf, 'PROD', 'DW')
        self.mysql_options = get_mysql_options(self.conf)
        self.now = self.parse_date_args()
        self.alert_on = self.parse_alert_on_args()
        self.logger.info(f"Running on date : {self.now}")
        self.job_name = 'DATA QUALITY JOB'

        # Special agg_dict operations:
        # count mean/ sum/ max/ min duration
        duration_expr = agg_dict_to_expr({'DURATION': {'duration|none|avg': F.mean, 'duration|none|sum': F.sum,
                                                       'duration|none|max': F.max, 'duration|none|min': F.min}})

        # sum on affinities
        aff_expr = agg_dict_to_expr({**{f'"{k}"': {f'user_id|{k}|sum': F.sum} for k in AFFINITIES}})

        # Basic count/ count distinct on key id of ml table (both snowflake & rds mysql)
        ml_options_dict = {"mysql": self.mysql_options, None: self.snowflake_options}
        ml_common_qlog_list = [QualityLog(option, table_identity.schema, table_identity.table_name,
                                          self.now, True, common_agg_dict_expr(table_identity.col_name), None,
                                          [F.to_date(F.col('UPDATE_DATE')) == self.now],
                                          filter_name)
                               for table_identity in common_ml_tables for filter_name, option in
                               ml_options_dict.items()]

        # Basic count/ count distinct on key id of backend snowflake tables
        backend_common_qlog_list = [QualityLog(self.snowflake_options, table_identity.schema, table_identity.table_name,
                                               self.now, True, common_agg_dict_expr(table_identity.col_name)) for
                                    table_identity
                                    in common_backend_tables]

        # Special agg operation on snowflake dw tables
        dw_main_qlog_list = [
            QualityLog(self.snowflake_options, "DW", "FACT_AUDIENCE", self.now, True,
                       aff_expr + common_agg_dict_expr("user_id")),
            QualityLog(self.snowflake_options, "DW", "FACT_REGISTERED", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), ['REG_TYPE'],
                       [((F.to_date(F.col('RECEIVED_AT')) == self.now - timedelta(days=1)) & (
                        F.col('ACTIVE_REG')))], 'active', None, {"threshold_up": 50, "threshold_down": 50}),
            QualityLog(self.snowflake_options, "DW", "FACT_CMP_USER_CONSENTS", self.now, True,
                       common_agg_dict_expr("user_id"), None,
                       [is_consent('PURPOSES', 'SPECIALFEATURES', 'CUSTOMPURPOSES')], 'consent'),
            QualityLog(self.snowflake_options, "DW", "FACT_CMP_USER_CONSENTS", self.now, True,
                       common_agg_dict_expr("user_id")),
            QualityLog(self.snowflake_options, "DW", "FACT_WATCH", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None,
                       [F.to_date(F.col('RECEIVED_AT')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_WATCH", self.now - timedelta(days=1), False,
                       common_agg_dict_expr("user_id") + duration_expr, ['DEVICE_TYPE'],
                       [F.to_date(F.col('RECEIVED_AT')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "DAILY_SUB_PRED_FLAG", self.now, True,
                       common_agg_dict_expr("user_id")),
            QualityLog(self.snowflake_options, "DW", "DAILY_SUB_PRED_FLAG", self.now, True,
                       common_agg_dict_expr("user_id"), ['SUB_CLUSTER']),
            QualityLog(self.snowflake_options, "DW", "FACT_USER_SUBSCRIPTION", self.now, True,
                       common_agg_dict_expr("user_id")),
            QualityLog(self.snowflake_options, "DW", "FACT_USER_SUBSCRIPTION", self.now, True,
                       common_agg_dict_expr("user_id"), ['PLATFORM']),
            QualityLog(self.snowflake_options, "DW", "FACT_USER_SUBSCRIPTION_DAY", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None, [F.to_date(F.col('DATE_DAY')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_RFM_WINDOW_SLIDE_01D_SIZE_28D", self.now, True,
                       common_agg_dict_expr("user_id"), ['RFM_CLUSTER'], [F.to_date(F.col('UPDATE_DATE')) == self.now]),
            QualityLog(self.snowflake_options, "DW", "FACT_RFM_WINDOW_SLIDE_01D_SIZE_07D", self.now, True,
                       common_agg_dict_expr("user_id"), ['RFM_CLUSTER'], [F.to_date(F.col('UPDATE_DATE')) == self.now]),
            QualityLog(self.snowflake_options, "DW", "FACT_SEARCH", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None, [F.to_date(F.col('TIMESTAMP')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_PAGE", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None, [F.to_date(F.col('TIMESTAMP')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_BOOKMARK_FOLLOW", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None, [F.to_date(F.col('TIMESTAMP')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_RAIL_CLIC_ATTRIBUTION", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), None, [F.to_date(F.col('TIMESTAMP')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "FACT_RAIL_CLIC_ATTRIBUTION", self.now - timedelta(days=1), True,
                       common_agg_dict_expr("user_id"), ['ORIGIN_SECTION'], [F.to_date(F.col('TIMESTAMP')) == self.now - timedelta(days=1)]),
            QualityLog(self.snowflake_options, "DW", "DIM_PRODUCT", self.now, True,
                       common_agg_dict_expr("product_id")),
            QualityLog(self.snowflake_options, "DW", "DIM_PRODUCT_RATE_PLAN", self.now, True,
                       common_agg_dict_expr("id")),
            QualityLog(self.snowflake_options, "DW", "DIM_PRODUCT_RATE_PLAN_CHARGE", self.now, True,
                       common_agg_dict_expr("id")),
            QualityLog(self.snowflake_options, "DW", "DIM_TAXATION", self.now, True,
                       common_agg_dict_expr("id")),
        ]

        self.qlog_list = dw_main_qlog_list + ml_common_qlog_list + backend_common_qlog_list

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Based on the snowflake_quality_conf, performs sanity checks in Snowflake Tables.
        """
        self.logger.info("Launching Data quality job")

        # 1 - Init cloudwatch client
        cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id=self.aws_access_key,
                                         aws_secret_access_key=self.aws_secret_key,
                                         region_name=self.region_name)

        # 2 - Calculate and send each metric to cloudwatch. Quality check on historic datapoints.
        sent_metrics, alogs = send_metrics_from_qlog(cloudwatch_client, self.spark, self.qlog_list, self.job_name)

        # 3 - Quality check exactitude of ml metric mysql vs snowflake
        d_ml = extract_ml_metrics(sent_metrics)
        for table, values in d_ml.items():
            is_valid, text = compare_ml_value(values)
            if not is_valid:
                alogs.append(
                    AlertLog(f'ALERT - {self.job_name}', 'Difference of volume between snowflake & mysql: ', None,
                             f"{table}: ", text))

        # 4 - Send logs to slack if alert/warning
        send_slack_notif_from_alogs(alogs, self.slack_token, self.alert_on)


if __name__ == "__main__":
    job = DataQualityJob()
    job.launch()
