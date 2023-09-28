from unittest import TestCase
from unittest import mock
import os

from databricks_jobs.jobs.personalised_reco_jobs.simple_vod_reco_job.entrypoint import VodRecoJob
from databricks_jobs.jobs.utils.popular_reco_utils import keep_top_X_programs
from tests.unit.utils.mocks import multiplex_mock, create_spark_df_from_data, mock_product_to_tvbundle
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


class TestVodRecoJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        cls.job = VodRecoJob()

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1}),
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.simple_vod_reco_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     write_df_to_snowflake=lambda df, *args, **kwargs: None):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_query_df=lambda spark, options, query:
                                         mock_product_to_tvbundle(spark),
                                         load_snowflake_table=multiplex_mock):
                    self.job.launch()

    def test_enriched_info_generation(self):
        ids = list(range(10))
        data = {
            "PROGRAM_ID": ids,
            "SEASON_NUMBER": ids,
            "EPISODE_NUMBER": ids,
            "CHANNEL_ID": ids,
            "DURATION": [1000 for _ in ids],
            "REF_PROGRAM_KIND_ID": [1 for _ in ids],
            "REF_PROGRAM_CATEGORY_ID": [1 for _ in ids],
            "EPISODE_ID": ids,
            "PRODUCTION_YEAR": [1999 for _ in ids]
        }
        vod_df = create_spark_df_from_data(self.job.spark, data=data)
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1}),
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.simple_vod_reco_job.entrypoint",
                                     load_snowflake_table=multiplex_mock):

                prog_with_info_df = self.job.join_pop_info_to_programs(vod_df)

                rows = prog_with_info_df.collect()
                prog1_row = find_user_rows(rows, 1, "PROGRAM_ID")
                self.assertEqual(prog1_row[0].total_celeb_points, 1)

    def test_reco_selection(self):
        ids = list(range(10))
        data = {
            "PROGRAM_ID": ids,
            "SEASON_NUMBER": ids,
            "EPISODE_NUMBER": ids,
            "CHANNEL_ID": [45 for _ in ids],
            "DURATION": [1000 for _ in ids],
            "REF_PROGRAM_KIND_ID": [1 for _ in ids],
            "REF_PROGRAM_CATEGORY_ID": [1 for _ in ids],
            "EPISODE_ID": ids,
            "PRODUCTION_YEAR": [1999 for _ in ids],
            "AFFINITY": ["Action & Aventure" for _ in ids],
            "TVBUNDLE_ID": [25 for _ in ids]
        }
        vod_df = create_spark_df_from_data(self.job.spark, data=data)
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [4], "total_celeb_points": 1000}),
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.simple_vod_reco_job.entrypoint",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_table=multiplex_mock,
                                         load_snowflake_query_df=lambda spark, options, query:
                                         mock_product_to_tvbundle(spark)):

                    prog_with_info_df = self.job.join_pop_info_to_programs(vod_df)

                    top_k_progs_per_aff_df = keep_top_X_programs(
                        prog_with_info_df, use_affinity=False,
                        scoring_methods=("distinct_replay_watch", "avg_replay_duration", "external_rating",
                                         "nb_likes", "total_celeb_points")
                    )
                    rez_df = self.job.select_recos(top_k_progs_per_aff_df)

                rows = rez_df.collect()
                user0_row = rows[0]
                # 2 program are kept because of watch duration, another because of celeb_pts
                self.assertSetEqual({r.PROGRAM_ID for r in user0_row.recommendations}, {0, 1, 4})

    def test_reco_selection_rules(self):
        ids = list(range(15))
        data = {
            "PROGRAM_ID": ids,
            "best_rank": ids,
            "CHANNEL_ID": [45 for _ in ids],
            "reco_origin": "reco",
            "AFFINITY": ["Divertissement" for _ in ids],
            "EPISODE_ID": ids
        }
        top_k_df = create_spark_df_from_data(self.job.spark, data)

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [4], "total_celeb_points": 1000}),
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.simple_vod_reco_job.entrypoint",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_table=multiplex_mock,
                                         load_snowflake_query_df=lambda spark, options, query:
                                         mock_product_to_tvbundle(spark)):
                    rez_df = self.job.select_recos(top_k_df)

        rows = rez_df.collect()
        user1_rows = find_user_rows(rows, 0)
        self.assertEqual(len(user1_rows), 1)
        self.assertLessEqual(len(user1_rows[0].recommendations), self.job.MAX_RECOS)
        self.assertLessEqual(len(set(user1_rows[0].recommendations)), self.job.MAX_RECOS)
