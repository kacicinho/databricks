import copy
import datetime
from collections import Counter, defaultdict
from datetime import timedelta
from operator import itemgetter
from typing import NamedTuple, Any, List

from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from unidecode import unidecode

from databricks_jobs.jobs.utils.slack_plugin import send_slack_message
from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_mysql_table

NAMESPACE = 'Data/quality'


class QualityLog(NamedTuple):
    """
    Enables to perform quality check from a pyspark agg_dict config on a pyspark df and transforms
    output for cloudwatch.
    Args:
        db_options (dict): jdbc/ snwoflake connector connection options
        schema (str): schema of the table
        table (str): name of the table
        date (datetime.date): date on which to perform the query
        alert (bool): whether to compute alert on metrics
        agg_dict (list): list of pyspark sql functions. Expected format: [<function>(F.col(<col_name>)).alias(<alias_name>)]
        where <alias_name> in format <metricname>|<alias_filter≥|<agg_operation>. The separator must be a pipe '|'.
        group_by (list): list of cols to group by
        filter_by (list): list of pyspark sql filter conditions
        filter_name (str): filter name to apply for cloudwatch metrics
        df (SparkDataFrame): pyspark df on which to perform queries. If None, it will load the table from schema/ table 
        and db_options provided.
        threshold_conf (dict): Dict to set the threshold variation for the metric. The variation should be between 0 and 100. 
        Must be in the following format, example: {"threshold_up": 60, "threshold_down": 40}
    """

    db_options: dict
    schema: str
    table: str
    date: datetime.date
    alert: bool
    agg_dict: list
    group_by: list = None
    filter_by: list = None
    filter_name: str = None
    df: SparkDataFrame = None
    threshold_conf: dict = None

    @property
    def table_name(self):
        return ".".join([self.schema, self.table])

    @property
    def db_name(self):
        if "sfUrl" in self.db_options:
            db_name = "snowflake"
        elif "url" in self.db_options:
            db_name = "jdbc"
        else:
            raise Exception("Incorrect db_options. Please review your db options.")
        return db_name

    def query_df(self, spark):

        if self.df:
            df = self.df
        else:
            try:
                df = load_snowflake_table(spark, self.db_options, self.table_name) if self.db_name == "snowflake" \
                    else load_mysql_table(spark, self.db_options, self.table)
            except Py4JJavaError as e:
                if "doesn't exist" in str(e.java_exception):
                    return None
                else:
                    raise Exception(e)

        if self.filter_by:
            df = df.where(*self.filter_by)

        if self.group_by:
            df = df.groupBy(*self.group_by)

        return df.agg(*self.agg_dict)

    def df_to_cloudwatch_metrics(self, df):
        """ Takes a pyspark df query result and build a list of metrics ready
        to be sent to cloudwatch."""

        metrics = []

        # Remove group by fields in agg cols
        agg_cols = df.columns
        if self.group_by:
            agg_cols = [x for x in agg_cols if x not in self.group_by]

        # Prepare format for cloudwatch
        for row in df.collect():
            metrics += [{**self.build_metric_dimensions(row, col_name),
                         **{'Timestamp': datetime.datetime(self.date.year, self.date.month, self.date.day),
                            'Unit': 'Count',
                            'Value': row[col_name] if row[col_name] else 0}} for col_name in agg_cols]
        return metrics

    def build_metric_dimensions(self, row, col_name):
        """ Takes a row from pyspark df query result and a column name of a agregate field as an input and
        returns a dict composed of a MetricName and Dimensions associated to the metric.

        Args:
            row: is expected to be a pyspark row
            col_name (str): is expected to be in format <col_name>|<alias_filter>|<agg_operation>. The expected
            separator is "|".

        Spaces, quotes and special characters are removed from metricname, agg_column, agg_column_value, alias_filter
        and agg_operation.

        Example:
        row = Row('RFM7_CLUSTER'='actifs', 'user_id|"Beauté & Bien-être"|count'=5)
        col_name = 'user_id|"Beauté & Bien-être"|count'
        self.group_by = 'RFM7_CLUSTER'
        self.schema = 'DW'
        self.table = 'FACT_AUDIENCE
        self.filter_name = None
        >>> build_metric_dimensions(row, col_name)
        >>> {'MetricName': 'user_id',
             'Dimensions': {'Name': 'version', 'Value': '3.0'},
                           {'Name': 'schema', 'Value': 'DW'},
                           {'Name': 'table', 'Value': 'FACT_AUDIENCE'},
                           {'Name': 'database', 'Value': 'snowflake'},
                           {'Name': 'agg_column', 'Value': 'rfm7_cluster'},
                           {'Name': 'agg_column_value', 'Value': 'actifs'},
                           {'Name': 'alias_filter', 'Value': 'beaute_&_bien-etre'},
                           {'Name': 'filter_name', 'Value': 'none'},
                           {'Name': 'agg_operation', 'Value': 'count'}}
        """

        dimensions = []

        # Build metric_name, agg_operation and alias_filter from col_name
        col_info = col_name.split('|')
        metric_name = self.format_name(col_info[0])
        alias_filter = self.format_name(col_info[1])
        agg_operation = self.format_name(col_info[2])

        # Add filter_name 
        filter_name = self.filter_name if self.filter_name else 'none'

        dimension_name = ['version', 'schema', 'table', 'database', 'agg_operation', 'alias_filter', 'filter_name']
        dimension_value = ['3.0', self.schema, self.table, self.db_name, agg_operation, alias_filter, filter_name]

        for name, value in zip(dimension_name, dimension_value):
            dimensions.append(self.add_dimension(name, value))

        # Add group by dimensions
        group_by_dim_name = ['agg_column', 'agg_column_value']
        if self.group_by:
            group_by_dim_value = ['_and_'.join([x for x in self.group_by]),
                                  '_and_'.join([str(row[x]) for x in self.group_by])]
            dimensions += [self.add_dimension(n, self.format_name(v)) for n, v in
                           zip(group_by_dim_name, group_by_dim_value)]
        else:
            dimensions += [self.add_dimension(n, 'none') for n in group_by_dim_name]

        return {'MetricName': metric_name, 'Dimensions': dimensions}

    def add_dimension(self, name, value):
        return {
            'Name': name,
            'Value': value
        }

    def format_name(self, name):
        """Remove spaces, quotes and special characters from string"""
        return unidecode(name.lower().replace(' ', '_').replace('"', '').replace('&', 'et'))


class CloudWatchMetric(NamedTuple):
    time_ref: datetime.date
    value: float


class CloudWatchLog(NamedTuple):
    historic_metrics: List[CloudWatchMetric]

    @classmethod
    def get_metrics_stat(cls, cloudwatch_client, metrics, statistic, end_date, nb_days=7):
        """Return a sorted list of tuple (timestamp, value) of all cloudwatch datapoint for a metric
        in the last nb_days.
        Nb : The StartTime and EndTime has to be datetime.datetime.
        """
        response = cloudwatch_client.get_metric_statistics(
            Namespace=NAMESPACE,
            Period=900,
            StartTime=end_date - timedelta(days=nb_days + 1),
            EndTime=end_date - timedelta(days=1),
            MetricName=metrics['MetricName'],
            Statistics=[statistic],
            Dimensions=metrics['Dimensions'])

        datapoints = response['Datapoints']
        historic_metrics = sorted([CloudWatchMetric(datapoint["Timestamp"].date(), datapoint[statistic])
                                   for datapoint in datapoints], key=itemgetter(1))
        return cls(historic_metrics)

    def get_missing_dates(self, end_date):
        """Check if missing datapoint in past 7 days"""
        missing_dates = []
        start_time = end_date - timedelta(days=7)
        end_time = end_date - timedelta(days=1)
        if self.historic_metrics:
            d = sorted([cw.time_ref for cw in self.historic_metrics])
            date_set = set(start_time + timedelta(x) for x in range((end_time - start_time).days))
            missing_dates = [datetime.datetime.strftime(date, "%Y-%m-%d") for date in sorted(date_set - set(d))]
        return missing_dates

    @property
    def get_multiple_values(self):
        """Check if multiple values sent on same date"""
        d = Counter(map(lambda x: datetime.datetime.strftime(x[0], "%Y-%m-%d"), self.historic_metrics))
        return [date for date in d if d[date] > 1]

    def get_value_7_days(self, end_date):
        """Get values at D-7 compared to end_date. We take the max here in case there is more than
        one datapoint on the same day."""
        if self.historic_metrics:
            values = [cw.value for cw in self.historic_metrics if cw.time_ref == end_date - timedelta(days=7)]
            return max(values) if values else None

    def calculate_variation_per(self, value_7d, new_datapoint):
        """Computes new datapoint % variation with average of historic datapoints"""
        return round(((float(new_datapoint) / float(value_7d)) - 1.0) * 100.0, 2)


class AlertLog(NamedTuple):
    alert_type: str
    alert_detail: str
    metric_text: dict
    value: Any
    text: str = None


class SlackBlockBuilder(NamedTuple):
    alogs: List[AlertLog]

    def add_new_alert_value(self, d, alog):
        """Add new alert value to dict alert metrics"""
        if alog.alert_type not in d:
            d.update(
                {alog.alert_type: {alog.alert_detail:
                                   {"metric_text": [agg_metric(alog.metric_text)],
                                    "value": [alog.value],
                                    "text": [alog.text]}}})
        elif alog.alert_type in d:
            if alog.alert_detail not in d[alog.alert_type]:
                d[alog.alert_type].update({alog.alert_detail:
                                           {"metric_text": [agg_metric(alog.metric_text)],
                                            "value": [alog.value],
                                            "text": [alog.text]}
                                           })
            elif alog.alert_detail in d[alog.alert_type]:
                d[alog.alert_type][alog.alert_detail]["metric_text"].append(agg_metric(alog.metric_text))
                d[alog.alert_type][alog.alert_detail]["value"].append(alog.value)
                d[alog.alert_type][alog.alert_detail]["text"].append(alog.text)
        return d

    def sort_alert_to_dict(self):
        """Transform alog list into dictionary for easier process afterwards 
        to form slack blocks. Expected output format dict:
            {alert_type : 
                {alert_detail1 :
                    {metric_text1 :[metric1, metric2, .. ],
                    text1 : [text1, text2, .. ],
                    value1 : [value1, value2, ..]
                },
               {alert_detail2 : 
                    {metric_text2 :[metric1, metric2, .. ],
                    text2 : [text1, text2, .. ],
                    value2 : [value1, value2, ..]
                },
                ..
            }"""
        d = {}
        for alog in self.alogs:
            d = self.add_new_alert_value(d, alog)
        return d

    def add_block_header(self, alert_type, emoji):
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji}{alert_type}"
            }
        }

    def add_block_title(self, alert_detail):
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{alert_detail}*"
            }
        }

    def add_block_section(self, metric_text, text, value, grafana_link):
        text = "" if text is None else text
        metric_text = " " if metric_text is None else metric_text
        return {
            "type": "section",
            "fields": [
                {
                    "type": "plain_text",
                    "text": f"{value}{text}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"{metric_text}\n{grafana_link}"
                }
            ]
        }

    def alert_to_blocks(self, d):
        """From the well formatted dict, creates the slack block formating message.
        The expected message blocks are in this format:

        alert_type as header
        alert_detail as title
        a combination of metric_text, text and value as block sections"""
        blocks = []

        for alert_type, d_detail in sorted(d.items()):
            alert_emoji = ":red_circle:" if 'ALERT' in alert_type else ":large_orange_circle:"
            blocks.append(self.add_block_header(f" *{alert_type}*", alert_emoji))
            for alert_detail, d_values in d_detail.items():
                blocks.append(self.add_block_title(alert_detail))
                for metric_text, value, text in zip(d_values["metric_text"], d_values["value"], d_values["text"]):
                    grafana_link = build_grafana_dash_link(metric_text)
                    blocks.append(self.add_block_section(metric_text, text, value, grafana_link))

        return blocks


def build_grafana_dash_link(metric_text, time_window_day=90):
    """Example value of metric_text:
    {'MetricName': 'user_id',
      'Dimensions': 
            {'version': '3.0',
             'schema': 'DW',
             'table': 'FACT_REGISTERED',
             'database': 'snowflake',
             'agg_operation': 'count',
             'alias_filter': 'none',
             'filter_name': 'none',
             'agg_column': 'none',
             'agg_column_value':
             'none'},
       'Value': 8}
    """
    root_url = "https://grafana.molotov.net/d/AUXHaBI7k/data-quality?orgId=6&"
    url = ""
    if metric_text:
        params = [f"var-metricname={metric_text['MetricName']}"] + \
            [f"var-{k.lower()}={v}" for k, v in metric_text['Dimensions'].items()] + \
            [f"from=now-{time_window_day}d&to=now"]
        url = f"<{root_url}{'&'.join(params)}|Grafana dashboard>"
    return url


def common_agg_dict_expr(col_name):
    """Count / count distinct agg_dict expr on provided col_name"""
    return agg_dict_to_expr(
        {col_name.upper(): {f'{col_name}|none|count': F.count, f'{col_name}|none|countDistinct': F.countDistinct}})


def agg_dict_to_expr(agg_dict):
    """
    Reads the agg_dict and returns a list of pyspark sql functions

    Args:
        agg_dict : {<col_name>: {<alias_name>: <pyspark sql function>}}. The expected
        format for the <alias_name> is <metricname>|<alias_filter≥|<agg_operation>. The expected seprator
        is "|". If no alias_filter, put 'none' as alias_filter value

    Example:
    >>> agg_dict = {'USER_ID': {'user_id|my_filter|count': F.count}}
    >>> agg_dict_to_expr(agg_dict)
    >>> [F.count(F.col('USER_ID')).alias('user_id|my_filter|count')]
    """
    return [to_expr(k, n, f) for k in agg_dict for n, f in agg_dict[k].items()]


def to_expr(col_name, alias_name, agg_func):
    if len(alias_name.split('|')) == 3:
        return agg_func(F.col(col_name)).alias(alias_name)
    else:
        raise Exception(f'Alias name for {alias_name} is not properly formatted')


def send_metrics(cloudwatch_client, metrics):
    """Send metrics to cloudwatch. We chunk the metric list in chunks of
    20 metrics max as it is the limit for the put_metric_data method"""
    chunked_metrics = [metrics[i:i + 20] for i in range(0, len(metrics), 20)]
    for chunk in chunked_metrics:
        cloudwatch_client.put_metric_data(Namespace=NAMESPACE, MetricData=chunk)


def flatten_dimension(d):
    return {dim['Name']: dim['Value'] for dim in d}


def agg_metric(metric):
    """Formats metric output"""
    return {'MetricName': metric['MetricName'],
            'Dimensions': flatten_dimension(metric['Dimensions']),
            'Value': metric['Value']
            } if metric else None


def is_ml_dim(dim):
    """Check if the dimensions are the same between a snowflake table and mysql table. The 
    checks are done on the count operation."""
    copy_dim = copy.deepcopy(dim)
    copy_dim.pop('table', None)
    copy_dim.pop('database', None)
    copy_dim.pop('filter_name', None)
    ml_dim = {'schema': 'ML',
              'agg_column': 'none',
              'agg_operation': 'count',
              'agg_column_value': 'none',
              'version': '3.0',
              'alias_filter': 'none'}

    return True if ml_dim == copy_dim else False


def extract_ml_metrics(metrics):
    """Exrtract snowflake/mysql metrics with same dimensions in a dictionary. Example expected output format is:
    d_ml = {'RECO_USER': {'snowflake': 8, 'mysql': 12}"""
    d_ml = defaultdict(dict)
    for metric in metrics:
        dim = flatten_dimension(metric['Dimensions'])
        if is_ml_dim(dim):
            d_ml[dim['table']].update({dim['database']: metric['Value']})
    return d_ml


def compare_ml_value(values):
    """For a table, compare if same value between snowflake and mysql.
    Example values : {'snowflake': 8, 'mysql': 12}"""
    try:
        return (True, "") if values["snowflake"] == values["jdbc"] else (
            False, f"snowflake: {values['snowflake']} vs jdbc: {values['jdbc']}")
    except KeyError as e:
        return (False, f"{e} missing")


def check_sent_metrics_vs_historic(cloudwatch_client, sent_metrics, date, job_name, threshold_conf=None):
    """Quality checks on new datapoints vs historic datapoints
    """
    alogs = []

    threshold_up = threshold_conf["threshold_up"] if threshold_conf else 50
    threshold_down = threshold_conf["threshold_down"] if threshold_conf else 30

    for metric in sent_metrics:

        # Fetch cloudwatch metrics on the last 7 days
        cwlog = CloudWatchLog.get_metrics_stat(cloudwatch_client, metric, "Maximum",
                                               datetime.datetime.combine(date, datetime.datetime.min.time()), nb_days=7)

        # Get multiple values on same date
        multiple_values = cwlog.get_multiple_values
        if multiple_values:
            alogs.append(AlertLog(f'WARNING - {job_name}', 'Mulitple values:', metric, multiple_values))

        # Get metric at D-7
        value_7d = cwlog.get_value_7_days(date)
        if value_7d is not None and value_7d > 0:
            # Caluculate variation. If +50% increase or 30% decrease in variation, then sends an alert.
            variation_per = cwlog.calculate_variation_per(value_7d, metric['Value'])
            if value_7d > 200 and (variation_per > threshold_up or variation_per < -threshold_down):
                alogs.append(AlertLog(f'ALERT - {job_name}', 'High variation:', metric, variation_per, "%"))

    return alogs


def send_metrics_from_qlog(cloudwatch_client, spark, qlog_list, job_name):
    """Send metrics to cloudwatch from list of QualityLog conf and perform alerting checks.
    Return:
        sent_metrics (list): list of all metrics sent. It can be used for further checks on sent metrics.
        alogs (list[AlertLog]): List of AlertLog from the metrics sent.
    """

    alogs = []
    sent_metrics = []

    # Calculate and send each metric to cloudwatch. Quality check on historic datapoints.
    for qlog in qlog_list:

        # Build and execute quality query
        df = qlog.query_df(spark)

        if df:
            # Build cloudwatch metrics from query result
            metrics = qlog.df_to_cloudwatch_metrics(df)

            # Send metrics to cloudwatch
            send_metrics(cloudwatch_client, metrics)
            sent_metrics += metrics

            # Quality checks on new datapoints vs historic datapoints
            if qlog.alert:
                alogs += check_sent_metrics_vs_historic(cloudwatch_client, metrics, qlog.date, job_name, qlog.threshold_conf)

        else:
            # Missing tables
            alogs.append(
                AlertLog(f'WARNING - {job_name}', 'Missing tables:', None, f"{qlog.table_name}", f" {qlog.db_name}"))

    return sent_metrics, alogs


def send_slack_notif_from_alogs(alogs, slack_token, alert_on=True):
    """Converts alogs to slakc block and sends message to slack if alerts/warning."""
    blockbuilder = SlackBlockBuilder(alogs)

    # Transform alerts to formatted dict for easier process
    alert_dict = blockbuilder.sort_alert_to_dict()
    # Build blocks message ready to send for slack
    blocks = blockbuilder.alert_to_blocks(alert_dict)

    if alert_on:
        if blocks:
            send_slack_message(blocks, slack_token)
        else:
            print("No alerts/warning, everything went well.. phew!")
