from unittest import TestCase
from unittest import mock
import os

from databricks_jobs.jobs.personalised_reco_jobs.next_episode_watch_job.entrypoint import NextEpisodeWatchJob
from databricks_jobs.db_common import build_broadcast_df_with_episode_info
from tests.unit.utils.mocks import multiplex_mock, mock_broadcast_for_episode_match, create_detailed_episode_df, \
    create_spark_df_from_data
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


def build_test_df(job, category=2, kind=1):
    data = {
        "USER_ID": [0],
        "PROGRAM_ID": [123],
        "last_seen_episode": [3],
        "last_seen_season": [1],
        "total_duration": [1000],
        "seen_episodes": [[111, 222, 333]],
    }
    user_watch_df = create_spark_df_from_data(job.spark, data)

    data = {
        "PROGRAM_ID": [123, 123],
        "START_AT": [job.now, job.now],
        "END_AT": [job.now, job.now],
        "SEASON_NUMBER": [1, 1],
        "EPISODE_NUMBER": [1, 4],
        "CHANNEL_ID": [1, 1],
        "DURATION": [10000, 10000],
        "REF_PROGRAM_KIND_ID": [kind, kind],
        "REF_PROGRAM_CATEGORY_ID": [category, category],  # TV show kind
        "EPISODE_ID": [111, 9]
    }
    broadcast_with_episode_df = create_spark_df_from_data(job.spark, data)
    df = job.filter_programs_with_next_watch(user_watch_df, broadcast_with_episode_df)
    return df


class TestNextWatchJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = NextEpisodeWatchJob()

    def test_last_watch_table(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.next_episode_watch_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            df = self.job.build_user_last_watch_df(create_detailed_episode_df(self.job.spark))
            rows = df.collect()

            # Matching infos are deduced from the fact_watch mock and episode_id mock
            for user_id, prog_id in zip([33, 1], [0, 1]):
                user_x_rows = find_user_rows(rows, user_id)
                self.assertEqual(user_x_rows[0].PROGRAM_ID, prog_id)
                self.assertEqual(user_x_rows[0].last_seen_episode, 1)

    def test_episode_filtering_tvshow(self):
        """
        This function tests that we do not return episodes when they are not following what the user has already seen
        """
        df = build_test_df(self.job)
        rows = df.collect()
        # Only one reco should remain, and it should be the episode 4
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].EPISODE_ID, 9)

    def test_episode_filtering_film(self):
        df = build_test_df(self.job, category=1)
        rows = df.collect()
        # Film are assumed to have no follow up, so they are filtered out
        self.assertEqual(len(rows), 0)

    def test_episode_filtering_docu(self):
        df = build_test_df(self.job, category=8)
        rows = df.collect()
        # For broadcast, nothing is filtered but the highest episode is taken
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].EPISODE_ID, 9)

    def test_filter_for_next_watch(self):
        """
        Based on how the mocks are set up
        We expect to have suggestion that are either next_episode or next_season
        """
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.next_episode_watch_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):

                episode_df = create_detailed_episode_df(self.job.spark)
                new_broadcast_df = build_broadcast_df_with_episode_info(
                    self.job.spark, self.job.options, episode_df, self.job.now, self.job.now + self.job.delta,
                    min_duration_in_mins=15)
                new_episode_mock, new_broadcast_mock = mock_broadcast_for_episode_match(
                    self.job.spark,
                    episode_df,
                    broadcast_df=new_broadcast_df
                )

                user_watch_df = self.job.build_user_last_watch_df(new_episode_mock)
                df = self.job.filter_programs_with_next_watch(user_watch_df, new_broadcast_mock)
                rows = df.collect()

                for user_id, prog_id in zip([33, 1], [0, 1]):
                    user_x_rows = find_user_rows(rows, user_id)
                    self.assertEqual(user_x_rows[0].PROGRAM_ID, prog_id)

                    # Check basic rules defined by the mock construction
                    self.assertEqual(user_x_rows[0].is_following_episode_for_user, prog_id == 1)
                    self.assertEqual(user_x_rows[0].is_next_season_for_user, prog_id == 0)

    def test_join_sources(self):
        data = {
            "USER_ID": [0, 0],
            "PROGRAM_ID": [1, 2],
        }
        df1 = create_spark_df_from_data(self.job.spark, data)

        data = {
            "USER_ID": [0, 0],
            "PROGRAM_ID": [3, 2],
        }
        df2 = create_spark_df_from_data(self.job.spark, data)
        df = self.job.join_sources(df1, df2)
        rows = df.collect()

        self.assertSetEqual({r.PROGRAM_ID for r in rows}, {1, 2, 3})
        r = find_user_rows(rows, 2, "PROGRAM_ID")[0]
        self.assertEqual(r.source, "vod")

    def test_full_run(self):
        """
        Similar to an integration test
        As two sources of data go through the job, this test is useful to check
        both sources are ok with the processing functions
        """
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.next_episode_watch_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.next_episode_watch_job.entrypoint.write_df_to_snowflake",
                                lambda df, opts, name, *args, **kwargs: df.show()):
                    self.job.launch()
