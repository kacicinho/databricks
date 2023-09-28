from unittest import TestCase
from unittest import mock
import os
import datetime
from datetime import timedelta
import boto3
from operator import itemgetter
from moto import mock_cloudwatch

from tests.unit.utils.mocks import multiplex_mock
from databricks_jobs.jobs.utils.data_quality_utils import CloudWatchLog, SlackBlockBuilder, QualityLog, AlertLog, \
    CloudWatchMetric, send_metrics, agg_dict_to_expr, extract_ml_metrics, compare_ml_value
from databricks_jobs.jobs.misc_jobs.data_quality_job.entrypoint import DataQualityJob
import pyspark.sql.functions as F
from pyspark.sql import SparkSession


class TestDataQualityJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.now = datetime.datetime.now().date()
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = DataQualityJob()

    @mock_cloudwatch
    @mock.patch.multiple("databricks_jobs.jobs.utils.data_quality_utils",
                         load_snowflake_table=multiplex_mock,
                         load_mysql_table=multiplex_mock)
    def test_send_metrics(self):
        cloudwatch_client = boto3.client('cloudwatch', aws_access_key_id='xxxx',
                                         aws_secret_access_key='xxxx',
                                         region_name='us-east-1')

        fact_reg_expr = agg_dict_to_expr({'USER_ID': {'user_id|test_alias_filter|count': F.count}}) + [F.sum(F.when(F.col('FILE_NAME').isNull(), 1)).alias('row|reg_no_amplitude_event|count')]

        quality_log_list = [
            QualityLog({"sfUrl": "xxx"}, "DW", "FACT_REGISTERED", self.now - timedelta(days=1), True, fact_reg_expr, ['REG_TYPE'],
                       [((F.to_date(F.col('RECEIVED_AT')) == self.now - timedelta(days=1)) & (F.col('ACTIVE_REG') == True))], 'active')
        ]

        for qlog in quality_log_list:
            df = qlog.query_df(self.job.spark)
            metrics = qlog.df_to_cloudwatch_metrics(df)
            send_metrics(cloudwatch_client, metrics)

        def flatten_dimension(d):
            return {dim['Name']: dim['Value'] for dim in d}

        sent_metrics = cloudwatch_client.list_metrics()['Metrics']
        # # check output format of metric data
        for metrics in sent_metrics:
            if metrics['MetricName'] == 'user_id':
                self.assertEqual(metrics['Namespace'], 'Data/quality')
                self.assertEqual(len(metrics['Dimensions']), 9)
                flatten_dim = flatten_dimension(metrics['Dimensions'])
                self.assertEqual(flatten_dim['version'], '3.0')
                self.assertEqual(flatten_dim['database'], 'snowflake')
                self.assertEqual(flatten_dim['schema'], 'DW')
                self.assertEqual(flatten_dim['table'], 'FACT_REGISTERED')
                self.assertEqual(flatten_dim['agg_column'], 'reg_type')
                self.assertIn(flatten_dim['agg_column_value'], ('b2c', 'b2b'))
                self.assertEqual(flatten_dim['agg_operation'], 'count')
                self.assertEqual(flatten_dim['alias_filter'], 'test_alias_filter')
                self.assertEqual(flatten_dim['filter_name'], 'active')

        # Check number of metrics sent
        self.assertEqual(len(sent_metrics), 2)

    def test_cloudwatch_log(self):

        end_date = datetime.datetime(2021, 9, 16, 8, 0).date()

        ex_1 = [
            {'Timestamp': datetime.datetime(2021, 9, 9, 9, 8), 'Maximum': 700, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 11, 10, 8), 'Maximum': 500, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 12, 11, 7), 'Maximum': 600, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 12, 11, 7), 'Maximum': 600, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 13, 4, 7), 'Maximum': 300, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 14, 11, 7), 'Maximum': 600, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 15, 11, 7), 'Maximum': 600, 'Unit': 'Count'}
        ]

        ex_2 = [
            {'Timestamp': datetime.datetime(2021, 9, 10, 11, 7), 'Maximum': 200, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 11, 10, 8), 'Maximum': 200, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 12, 11, 7), 'Maximum': 300, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 13, 11, 7), 'Maximum': 200, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 14, 11, 7), 'Maximum': 200, 'Unit': 'Count'},
            {'Timestamp': datetime.datetime(2021, 9, 15, 11, 7), 'Maximum': 400, 'Unit': 'Count'}
        ]

        ex_3 = []

        def format_metrics(datapoints):
            return sorted([CloudWatchMetric(datapoint["Timestamp"].date(), datapoint['Maximum']) for datapoint in datapoints], key=itemgetter(1))

        cwlogs = [CloudWatchLog(format_metrics(ex_1)), CloudWatchLog(format_metrics(ex_2)), CloudWatchLog(format_metrics(ex_3))]
        answers = [[['2021-09-10'], ['2021-09-12'], 14.29], [['2021-09-09'], [], None], [[], [], None]]

        for cwlog, ans in zip(cwlogs, answers):

            # check missing dates
            missing_dates = cwlog.get_missing_dates(end_date)
            self.assertListEqual(missing_dates, ans[0])

            # check multiple values
            multiple_values = cwlog.get_multiple_values
            self.assertListEqual(multiple_values, ans[1])

            # check variation_per
            value_7d = cwlog.get_value_7_days(end_date)
            variation_per = None
            if value_7d is not None:
                variation_per = cwlog.calculate_variation_per(value_7d, 800)
            self.assertEqual(variation_per, ans[2])

    def test_alert_blocks(self):

        end_date = datetime.datetime(2021, 9, 16, 0, 0).date()
        metric_ex = {
            'MetricName': 'user_id',
            'Dimensions': [
                {'Name': 'version', 'Value': '3.0'},
                {'Name': 'schema', 'Value': 'DW'},
                {'Name': 'table', 'Value': 'FACT_REGISTERED'},
                {'Name': 'database', 'Value': 'snowflake'},
                {'Name': 'agg_operation', 'Value': 'count'},
                {'Name': 'alias_filter', 'Value': 'none'},
                {'Name': 'filter_name', 'Value': 'active'},
                {'Name': 'agg_column', 'Value': 'reg_type'},
                {'Name': 'agg_column_value', 'Value': 'b2c'}
            ],
            'Unit': 'Count',
            'Timestamp': datetime.datetime(2021, 9, 19, 0, 0),
            'Value': 8
        }

        alogs = [
            AlertLog('WARNING', 'Missing dates:', metric_ex, ['2021-09-01']),
            AlertLog('WARNING', 'Missing dates:', metric_ex, ['2021-08-12']),
            AlertLog('ALERT', 'High variation:', metric_ex, 58, "%"),
            AlertLog('ALERT', 'High variation:', metric_ex, 72, "%"),
            AlertLog('WARNING', 'Mulitple values:', metric_ex, ['2021-10-01']),
            AlertLog('WARNING', 'Missing tables:', None, "DW.test_table", " snowflake"),
            AlertLog('WARNING', 'Missing D-7 metric:', metric_ex, end_date - timedelta(days=7), "")
        ]

        blockbuilder = SlackBlockBuilder(alogs)
        alert_dict = blockbuilder.sort_alert_to_dict()
        self.assertIn('WARNING', alert_dict.keys())
        self.assertIn('ALERT', alert_dict.keys())
        self.assertIn('Missing dates:', alert_dict['WARNING'].keys())
        self.assertIn('Mulitple values:', alert_dict['WARNING'].keys())
        self.assertIn('Missing tables:', alert_dict['WARNING'].keys())
        self.assertIn('High variation:', alert_dict['ALERT'].keys())
        self.assertEqual(len(alert_dict['ALERT']['High variation:'].keys()), 3)
        self.assertEqual(len(alert_dict['ALERT']['High variation:']['value']), 2)

        blocks = blockbuilder.alert_to_blocks(alert_dict)
        self.assertEqual(len(blocks), 14)

    def test_exactitude_metrics(self):

        dim_mysql = [
            {'Name': 'schema', 'Value': 'ML'},
            {'Name': 'agg_column', 'Value': 'none'},
            {'Name': 'database', 'Value': 'jdbc'},
            {'Name': 'agg_operation', 'Value': 'count'},
            {'Name': 'agg_column_value', 'Value': 'none'},
            {'Name': 'version', 'Value': '3.0'},
            {'Name': 'alias_filter', 'Value': 'none'},
            {'Name': 'filter_name', 'Value': 'mysql'},
            {'Name': 'table', 'Value': 'RECO_CHANNEL_TAGS_LATEST'}]

        dim_sf = [
            {'Name': 'schema', 'Value': 'ML'},
            {'Name': 'agg_column', 'Value': 'none'},
            {'Name': 'database', 'Value': 'snowflake'},
            {'Name': 'agg_operation', 'Value': 'count'},
            {'Name': 'agg_column_value', 'Value': 'none'},
            {'Name': 'version', 'Value': '3.0'},
            {'Name': 'alias_filter', 'Value': 'none'},
            {'Name': 'filter_name', 'Value': 'none'},
            {'Name': 'table', 'Value': 'RECO_CHANNEL_TAGS_LATEST'}]

        metrics = [
            {'MetricName': 'user_id',
             'Dimensions': dim_sf,
                'Unit': 'Count',
                'Timestamp': datetime.datetime(2021, 9, 19, 0, 0),
             'Value': 8},
            {'MetricName': 'user_id',
             'Dimensions': dim_mysql,
                'Unit': 'Count',
                'Timestamp': datetime.datetime(2021, 9, 19, 0, 0),
             'Value': 13}
        ]

        # test wrong values
        d_ml = extract_ml_metrics(metrics)
        is_valid, text = compare_ml_value(d_ml['RECO_CHANNEL_TAGS_LATEST'])
        self.assertEqual(is_valid, False)
        self.assertEqual(text, "snowflake: 8 vs jdbc: 13")

        # test missing value
        d_ml['RECO_CHANNEL_TAGS_LATEST'].pop('snowflake')
        is_valid, text = compare_ml_value(d_ml['RECO_CHANNEL_TAGS_LATEST'])
        self.assertEqual(is_valid, False)
        self.assertEqual(text, "'snowflake' missing")
