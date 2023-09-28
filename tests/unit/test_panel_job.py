from unittest import TestCase
from unittest import mock
import os
import math
import datetime
import random

from databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint import PanelJob
from tests.unit.utils.mocks import multiplex_mock, create_spark_df_from_data, multiplex_query_mock
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


def build_consent_watch_mock(now):
    # Build the mock for consent fact watch
    hour_diff = datetime.timedelta(hours=1)
    data = {"USER_ID": [0, 0, 0], "WATCH_START": [now - 3 * hour_diff, now - 2 * hour_diff, now - 1 * hour_diff],
            "WATCH_END": [now - 2 * hour_diff, now - 1 * hour_diff, now], "CHANNEL_ID": [0, 1, 2]}
    return data


def create_nightly_mock(spark):
    data = {
        "USER_ID": [10]
    }
    return create_spark_df_from_data(spark, data)


class TestPanelJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = PanelJob()

    def setUp(self) -> None:
        self.job.MIN_USER_PER_SEGMENT = 1

    def test_user_presence_timestamp(self):
        """
        In this test, we need to mock get_watch_consent_df to control the data returned by this function,
        thus we can check that we do the right kind of timestamp rounding
        """
        today = datetime.date.today()
        now = datetime.datetime(today.year, today.month, today.day)
        data = build_consent_watch_mock(now)
        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name),
                                 load_snowflake_query_df=lambda spark, options, table_name:
                                 create_nightly_mock(spark)):
            with mock.patch("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint.get_watch_consent_df",
                            new=lambda spark, options, start, end=None, with_consent=None:
                                create_spark_df_from_data(spark, data)):
                df = self.job.build_user_presence_df()
                rows = df.collect()

                # One watch session is transformed into 2 events
                self.assertEqual(len(rows), 6)

                # We should have the same timestamp of events
                timestamps = [r.timestamp.timestamp() for r in rows]
                expected = sorted([math.ceil(t.timestamp() / self.job.TIME_GRANULARITY) * self.job.TIME_GRANULARITY
                                   for t in data["WATCH_START"] + data["WATCH_END"]])
                self.assertListEqual(sorted(timestamps), expected)

    def test_user_presence_count(self):
        with mock.patch("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.utils.onboarding_utils.load_snowflake_query_df",
                            new=multiplex_query_mock):
                df = self.job.build_user_presence_df()
                rows = df.collect()

                # One watch session is transformed into 2 events
                self.assertEqual(len(rows), 12)

    def test_user_enrichment_df(self):
        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name),
                                 load_snowflake_query_df=lambda spark, options, table_name:
                                 create_nightly_mock(spark)):
            with mock.patch("databricks_jobs.jobs.utils.utils.load_snowflake_table",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                df = self.job.build_user_info_base()
                rows = df.collect()
                self.assertSetEqual({r.category for r in rows}, {"H_4", "F_4"})

                # Let's see the normalisation factor,
                # it should be the sum of the weights of the 4 h_4 users
                user_1_row = find_user_rows(rows, 1, "ID")[0]
                # Max weight = 1 because we have ID 1, 2, 4 with weights 0.5, 0.4, 0.1
                # Note affinities filter out 3 and all users have all affinities
                self.assertEqual(round(user_1_row.max_pop_per_cat, 3), round(sum([0.5, 0.1, 1.7]), 3))

    def test_min_user_per_segment(self):
        self.job.MIN_USER_PER_SEGMENT = 5

        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name),
                                 load_snowflake_query_df=lambda spark, options, table_name:
                                 create_nightly_mock(spark)):
            with mock.patch("databricks_jobs.jobs.utils.utils.load_snowflake_table",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                df = self.job.build_user_info_base()
                rows = df.collect()
                # We require to have at least 5 users in all subgroups, we have only 4,
                # so everything should be filtered out
                self.assertEqual(len(rows), 0)

    def test_cumsum_ops(self):
        today = datetime.date.today()
        now = datetime.datetime(today.year, today.month, today.day)

        def create_watch_pattern_mock(spark):
            uids = list(range(7))
            true_start = now - datetime.timedelta(days=1.99)
            starts = [true_start + datetime.timedelta(minutes=int(random.random() * 12 * 60)) for _ in uids]
            ends = [start + datetime.timedelta(days=1) for start in starts]
            data = {
                "USER_ID": uids + uids,
                "CHANNEL_ID": [0 for _ in uids] + [0 for _ in uids],
                "timestamp": starts + ends,
                "watch": [1 for _ in uids] + [-1 for _ in uids],
            }
            return create_spark_df_from_data(spark, data)
        with mock.patch.multiple("databricks_jobs.jobs.third_party_export_jobs.panel_job.entrypoint",
                                 load_snowflake_table=lambda spark, options, table_name:
                                 multiplex_mock(spark, options, table_name),
                                 load_snowflake_query_df=lambda spark, options, table_name:
                                 create_nightly_mock(spark)):
            with mock.patch("databricks_jobs.jobs.utils.utils.load_snowflake_table",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                user_base_df = self.job.build_user_info_base()
                user_presence_df = create_watch_pattern_mock(self.job.spark)

                df = self.job.build_cumsum_over_segments(user_base_df, user_presence_df)

                rows = df.collect()
                cat_h4_rows = find_user_rows(rows, "H_4", "category")
                # All affinities are available for all users except number 3, choose 1
                cat_h4_rows = find_user_rows(cat_h4_rows, "Mode", "affinity")

                # Cum sum >= 0 all the time
                self.assertTrue(all(r.total_watchers >= 0 for r in cat_h4_rows))
                # There should be a +1 and a -1 per user and user 3 has no affinity (nb = 3)
                # All -1 are in the futur, so should not produce a line
                self.assertEqual(len(cat_h4_rows), 3)
                self.assertEqual(cat_h4_rows[0].total_users_in_group, int(2.3))
                # Channel info is the same for all rows
                self.assertSetEqual({r.CHANNEL_ID for r in rows}, {0})
