from unittest import TestCase
from unittest import mock
import os
import datetime

from databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint import UserAndProgFeatureLogJob
from databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.helpers import DictFormatter, SocioDemoBuilder, \
    SocioDemoAggregator
from tests.unit.utils.mocks import multiplex_mock, create_spark_df_from_data
from tests.unit.utils.test_utils import find_user_rows
from databricks_jobs.jobs.misc_jobs.sample.entrypoint import SampleJob
from pyspark.sql import SparkSession


class TestUserProgFeatureJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = UserAndProgFeatureLogJob()

    @classmethod
    def prepare_mocks(cls):
        now = datetime.datetime.now()
        data = {
            "USER_ID": [1, 1, 1, 1, 1],
            "PROGRAM_ID": [0, 5, 2, 4, 1],
            "DURATION": [1345, 13450, 245, 23456, 120],
            "CHANNEL_ID": [1, 2, 1, 2, 1],
            "CATEGORY_ID": [1, 2, 1, 4, 10],
            "ACTION_DEVICE_TYPE": ["phone", "desktop", "tv", "tablet", "tv"],
            'REAL_START_AT': [now,
                              now - datetime.timedelta(days=1),
                              now - datetime.timedelta(days=2),
                              datetime.datetime(1990, 1, 1),
                              now - datetime.timedelta(days=1)]
        }
        fact_watch_mock = create_spark_df_from_data(cls.job.spark, data)

        ids = list(range(100))
        data = {
            "PROGRAM_ID": ids,
            "PERSON_ID": [1 if i % 2 == 0 else 2 for i in ids]
        }
        top_pers_df = create_spark_df_from_data(cls.job.spark, data)
        return fact_watch_mock, top_pers_df

    def test_user_feat_log(self):
        fact_watch_mock, top_pers_df = self.prepare_mocks()

        with mock.patch("databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: fact_watch_mock
                        if "watch" in table_name.lower() else multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint.load_snowflake_query_df",
                            new=lambda spark, options, query: top_pers_df):
                fact_watch_aug, user_df, top_pers_df = self.job.build_preliminary_tables()
                df = self.job.build_user_feature_log(fact_watch_aug, user_df, top_pers_df)
                df.collect()

    def test_program_feat_log(self):
        ids = list(range(100))
        data = {
            "PROGRAM_ID": ids,
            "PERSON_ID": [1 if i % 2 == 0 else 2 for i in ids]
        }
        top_pers_df = create_spark_df_from_data(self.job.spark, data)

        with mock.patch("databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name),):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     load_snowflake_query_df=lambda spark, options, query:
                                     create_spark_df_from_data(spark, {"PROGRAM_ID": [1, 2, 3],
                                                                       "total_celeb_points": [1, 10, 100]})):
                df = self.job.build_program_feature_log(top_pers_df)
                df.collect()

    def test_channel_feat_log(self):
        fact_watch_mock, top_pers_df = self.prepare_mocks()

        with mock.patch("databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: fact_watch_mock
                        if "watch" in table_name.lower() else multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.ml_models_jobs.user_and_prog_feature_log_job.entrypoint.load_snowflake_query_df",
                            new=lambda spark, options, query: top_pers_df):
                fact_watch_mock, user_df, top_pers_df = self.job.build_preliminary_tables()
                df = self.job.build_channel_feature_log(fact_watch_mock, user_df)
                self.assertIn("channel_DURATION_socio=F_15", df.columns)
                df.collect()


class TestHelpers(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.job = SampleJob()

    def test_formatter_for_pivot_renaming(self):
        f = DictFormatter("test", "NAME", ["JEAN", "PIERRE"])
        renamed_cols = f.final_col_name
        self.assertSetEqual({renamed_cols}, {"test_NAME"})

    def test_formatter_for_pivot(self):
        data = {
            "HOUR": [1, 1, 2],
            "NAME": ["JEAN", "PIERRE", "JEAN"],
            "DURATION": [33, 77, 12]
        }
        df = create_spark_df_from_data(self.job.spark, data)

        f = DictFormatter("DURATION", "NAME", ["JEAN", "PIERRE"])
        final_df = f.build_sum_dict(df, "HOUR")

        self.assertSetEqual(set(final_df.columns), {"HOUR", "DURATION_NAME"})
        rows = final_df.collect()
        r1 = find_user_rows(rows, 1, "HOUR")[0]
        self.assertEqual(r1.asDict()["DURATION_NAME"], {"PIERRE": 77, "JEAN": 33})

    def test_socio_demo_aggregator(self):
        sd = SocioDemoBuilder(1, 33, "M")
        sda = SocioDemoAggregator(sd, "DURATION", prefix="test")
        data = {
            "GENDER": ["M"],
            "AGE": [23],
            "DURATION": [10]
        }
        df = create_spark_df_from_data(self.job.spark, data)

        rez = df.select(sda.op_fn())

        self.assertSetEqual(set(rez.columns), {sda.formatted_col_name})
        self.assertSetEqual(set(rez.columns), {"test_DURATION_socio=M_2_33"})
