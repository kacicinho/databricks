import os
from unittest import TestCase
from unittest import mock
import datetime

from databricks_jobs.jobs.personalised_reco_jobs.merge_reco_job.entrypoint import MergeRecoJob
from tests.unit.utils.mocks import multiplex_mock, mock_raw_user, mock_raw_program, create_spark_df_from_data
from databricks_jobs.jobs.utils.merge_reco_utils import MergeInfos

from pyspark.sql import SparkSession


class TestMergeRecoJobUserID(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.merge_reco_job.entrypoint.MergeRecoJob",
                                 parse_table_infos_args=lambda *args, **kwargs:
                                 MergeInfos(merge_reco_table="RECO_USER_PROGS_THIS_WEEK_VAR",
                                            latest_reco_table="RECO_USER_PROGS_THIS_WEEK_LATEST",
                                            key_name="USER_ID"),
                                 add_more_args=mock.DEFAULT):

            cls.job = MergeRecoJob()

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.merge_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                 load_snowflake_query_df=lambda spark, options, query:
                                 mock_raw_user(spark)):
            self.job.launch()

    def test_merge_latest_reco(self):
        pids = list(range(5))
        now = datetime.datetime.now().date()

        all_split_users_mock = {
            "AB_TEST_ID": [1, 1, 1, 1, 1],
            "USER_ID": [0, 1, 4, 2, 3],
            "VARIATIONS": ["A", "B", "A", "B", "B"]
        }

        reco_df_mock = {
            "USER_ID": 2 * [i for i in pids],
            "RECOMMENDATIONS": 2 * [
                """[{
                    "AFFINITY": "Drames & Sentiments",
                    "PROGRAM_ID": 46802,
                    "ranking": 8.734767,
                    "rating": 0.1,
                    "reco_origin": "total_celeb_points"
                }]""" for _ in pids],
            "UPDATE_DATE": 2 * [now for _ in pids],
            "VARIATIONS": ["A" for _ in pids] + ["B" for _ in pids]
        }

        all_split_users_mock = create_spark_df_from_data(self.job.spark, all_split_users_mock)
        reco_df_mock = create_spark_df_from_data(self.job.spark, reco_df_mock)

        rows = self.job.merge_latest_reco(all_split_users_mock, reco_df_mock).collect()
        # The _var table has 10 rows made up of 5 user id with both variation A & B.
        # In the final _latest table there should be only 5 user ids. 
        self.assertEqual(len(rows), 5)


class TestMergeRecoJobProgID(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.merge_reco_job.entrypoint.MergeRecoJob",
                                 parse_table_infos_args=lambda *args, **kwargs:
                                 MergeInfos(merge_reco_table="RECO_PROG_PROGS_META_VAR",
                                            latest_reco_table="RECO_PROG_PROGS_META_LATEST",
                                            key_name="PROGRAM_ID"),
                                 add_more_args=mock.DEFAULT):

            cls.job = MergeRecoJob()

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.merge_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                 load_snowflake_query_df=lambda spark, options, query:
                                 mock_raw_program(spark)):
            self.job.launch()
