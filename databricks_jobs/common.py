import json
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger
import datetime
from typing import Dict, Any
from distutils.util import strtobool

from pyspark.sql import SparkSession
from databricks_jobs.jobs.utils.local_dbfs import LocalDbutils


# abstract class for jobs
class Job(ABC):

    @abstractmethod
    def init_adapter(self):
        """
        Init adapter is an abstract method to perform some particular settings in the Job subclass.
        Method is called after creation of the SparkSession.
        :return:
        """
        pass

    def __init__(self, spark=None, init_conf=None):
        self.spark = self._prepare_spark(spark)
        self.logger = self._prepare_logger()
        self.parsed_args = self.get_arg_parse()
        self.dbutils = self.get_dbutils()
        if init_conf:
            self.conf = init_conf
        else:
            self.conf = self._provide_config(self.parsed_args)
        self.init_adapter()
        self._log_conf()

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        else:
            return spark

    @staticmethod
    def _get_dbutils(spark: SparkSession):
        try:
            from pyspark.dbutils import DBUtils # noqa
            if "dbutils" not in locals():
                utils = DBUtils(spark)
                return utils
            else:
                return locals().get("dbutils")
        except ImportError:
            return LocalDbutils()

    def get_dbutils(self):
        utils = self._get_dbutils(self.spark)

        if not utils:
            self.logger.warn("No DBUtils defined in the runtime")
        else:
            self.logger.info("DBUtils class initialized")

        return utils

    def _provide_config(self, namespace):
        self.logger.info("Reading configuration from --conf-file job option")
        conf_file = namespace.conf_file
        conf_dict = self._read_config(conf_file) if conf_file else {}

        if len(conf_dict) == 0:
            self.logger.info('No conf file was provided, setting configuration to empty dict.'
                             'Please override configuration in subclass init method')
        else:
            self.logger.info(f"Conf file was provided, reading configuration from {conf_file}")

        # Add optional date info in the config
        if namespace.date is not None:
            conf_dict["date"] = namespace.date

        # Add optional alert info in the config
        conf_dict["alert_on"] = namespace.alert_on
        return conf_dict

    @classmethod
    def get_arg_parse(cls):
        p = ArgumentParser()
        p.add_argument("--conf-file", required=False, type=str)
        p.add_argument("--date", default=None, type=str)
        p.add_argument("--alert_on", default="True", type=str)
        cls.add_more_args(p)
        return p.parse_known_args()[0]

    @staticmethod
    def add_more_args(arg_parser):
        pass

    def parse_date_args(self):
        # Handle the date parameter to get a date object
        default_date = datetime.datetime.now().date()
        job_date = self.conf.get("date")

        date = None
        if job_date is not None:
            # Airflow sends the date D-1 when schedule a D + 5 minutes
            # We add a day to keep the same format as before
            date = (datetime.datetime.strptime(job_date, "%Y-%m-%d") + datetime.timedelta(days=1)).date()
            # To avoid mistakes when scheduling manually, we cannot have a date larger than today
            if date > default_date:
                date = default_date

        return date or default_date

    def parse_alert_on_args(self):
        """Expected value for --alert_on: "True", "Yes" or "On" for positive values. "False", "No" or "Off" for 
        negative values"
        """
        try:
            alert_on = self.conf.get("alert_on")
            return bool(strtobool(self.conf.get("alert_on"))) if alert_on else True
        except ValueError:
            raise ValueError("Wrong input parameter for --alert_on")

    def _read_config(self, conf_file) -> Dict[str, Any]:
        raw_content = "".join(self.spark.read.format("text").load(conf_file).toPandas()["value"].tolist())
        config = json.loads(raw_content)
        return config

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    def _log_conf(self):
        # log parameters
        self.logger.info("Launching job with configuration parameters:")
        for key, item in self.conf.items():
            self.logger.info("\t Parameter: %-30s with value => %-30s" % (key, item))

    @abstractmethod
    def launch(self):
        """
        Main method of the job.
        :return:
        """
        pass
