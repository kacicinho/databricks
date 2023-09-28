import datetime
import os
from unittest import TestCase
from unittest import mock

from pyspark.sql import SparkSession

from databricks_jobs.jobs.personalised_reco_jobs.reco_replay_job.entrypoint import ReplayRecoJob
from tests.unit.utils.mocks import multiplex_mock, create_spark_df_from_data, mock_product_to_tvbundle
from tests.unit.utils.test_utils import find_user_rows


class TestVodRecoJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        cls.job = ReplayRecoJob()

    def test_compute_episode_live_stats(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.reco_replay_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_query_df=lambda spark, options, query:
                                     mock_product_to_tvbundle(spark),
                                     load_snowflake_table=multiplex_mock):
                df = self.job.compute_episode_live_stats()
                rows = df.collect()
                # In the mock, only two episode exists, watched separately by 2 user 1 and 33
                for r in rows:
                    self.assertEqual(r.engaged_watchers, 0)  # Not engaged enough as it is not >70% or 20 minutes
                    self.assertGreater(r.total_duration, 1)

    def test_build_first_broadcast_time_df(self):
        episode_data = {
            "EPISODE_ID": [1, 2, 3, 4, 5]
        }
        episode_df = create_spark_df_from_data(self.job.spark, episode_data)
        now = datetime.datetime.now()
        broadcast_data = {
            "EPISODE_ID": [1, 2, 3, 4, 5],
            "PROGRAM_ID": [1, 2, 3, 4, 5],
            "START_AT": [now - datetime.timedelta(days=1), now - datetime.timedelta(days=1),
                         now - datetime.timedelta(days=1), now - datetime.timedelta(days=1),
                         now - datetime.timedelta(days=30)],
            "END_AT": [now, now, now, now, now],
            "TVBUNDLE_ID": [25, 25, 25, 25, 25]
        }
        broadcast_mock = create_spark_df_from_data(self.job.spark, broadcast_data)
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.reco_replay_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 build_broadcast_df_with_episode_info=lambda *args, **kwargs: broadcast_mock):
            df = self.job.build_first_broadcast_time_df(episode_df)
            rows = df.collect()

            self.assertSetEqual({r.aired_on_free for r in rows}, {True})
            self.assertEqual(len(find_user_rows(rows, 30, "age")), 1)
            self.assertEqual(len(find_user_rows(rows, 1, "age")), 4)

    def test_select_reco(self):
        ids = list(range(10))
        data = {
            'EPISODE_ID': ids,
            'CHANNEL_ID': [46 for _ in ids],
            'PROGRAM_ID': ids,
            'best_rank': ids,
            'reco_origin': ["my_metric" for _ in ids]
        }
        top_k_df = create_spark_df_from_data(self.job.spark, data)
        with mock.patch.multiple("databricks_jobs.db_common",
                                 load_snowflake_query_df=lambda spark, options, query:
                                 mock_product_to_tvbundle(spark),
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.reco_replay_job.entrypoint",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_table=multiplex_mock):
                    rez = self.job.select_recos(top_k_df)
        rows = rez.collect()

        user_0_rows = find_user_rows(rows, 0)
        self.assertEqual(len(user_0_rows), 1)

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1}),
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.reco_replay_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     write_df_to_snowflake=lambda df, *args, **kwargs: None):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_query_df=lambda spark, options, query:
                                         mock_product_to_tvbundle(spark),
                                         load_snowflake_table=multiplex_mock):
                    self.job.launch()
