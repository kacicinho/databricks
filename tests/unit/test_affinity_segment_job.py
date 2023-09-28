import os
from unittest import TestCase
from unittest import mock
import subprocess
import datetime

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs.affinity_segment_job.entrypoint import \
    AffinitySegmentJob
from tests.unit.utils.test_utils import find_user_rows
from tests.unit.utils.mocks_affinity import create_audience_prog_type_affinity_df_mock, create_program_df_mock, \
    create_affinities_df_mock, multiplex_mock, create_spark_df_from_data


class TestAffinitySegmentJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        subprocess.run(["python3", "-m", "spacy", "download", "fr_core_news_lg"])
        cls.job = AffinitySegmentJob(spark=spark)

    def test_audience_prog_type_affinity_df(self):
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_segment_job.entrypoint.load_snowflake_query_df",
                        new=lambda spark, options, query: create_audience_prog_type_affinity_df_mock(spark)):
            aff_df = self.job.audience_prog_type_affinity_df_job()
            rows = aff_df.collect()
            # Check expected number of rows
            self.assertEqual(len(rows), 5)
            # Check exepected results
            for r in rows:
                if r.USER_ID == 1:
                    self.assertEqual(r.movie, 0)
                    self.assertEqual(r.serie, 0)
                    self.assertEqual(r.broadcast, 1)
                    self.assertEqual(r.documentary, 1)
                    break

    def test_program_df(self):
        expected_columns = ['PROGRAM_ID', 'PROGRAM', 'CATEGORY', 'KIND', 'SUMMARY', 'info', 'subinfo', 'score']
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_segment_job.entrypoint.load_snowflake_query_df",
                        new=lambda spark, options, query: create_program_df_mock(spark)):
            program_df = self.job.program_df_job()
            rows = program_df.collect()
            columns = program_df.columns
            # Check expected columns
            self.assertListEqual(columns, expected_columns)
            # Check expected rows / 'Religion' Kind should be removed
            self.assertEqual(len(rows), 3)
            # Check example of expected output for info / subinfo
            for r in rows:
                if r.PROGRAM_ID == 0:
                    self.assertEqual(r.info, 'enfant enfant series')
                    self.assertEqual(r.subinfo, '')
                elif r.PROGRAM_ID == 1:
                    self.assertEqual(r.info, 'information')
                    self.assertEqual(r.subinfo, 'info journal télévisé')

    def test_affinities_df(self):
        expected_columns = ['segment_name_tokens', 'segment_algorithm', 'segment_table', 'profil', 'subcategories']
        with mock.patch("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                        "affinity_segment_job.entrypoint.load_affinities_df",
                        new=lambda spark: create_affinities_df_mock(spark)):
            affinities_df = self.job.affinities_df_job()
            rows = affinities_df.collect()
            columns = affinities_df.columns
            # Check expected columns
            self.assertListEqual(columns, expected_columns)
            # Check Subcategories < 3
            self.assertEqual(len(rows), 3)
            # Check example of expected output
            for r in rows:
                if r.segment_algorithm == 'Divertissement/Film/Sentimentales':
                    self.assertEqual(r.segment_name_tokens, 'divertissement film sentimental')
                    self.assertEqual(r.subcategories, 1)

    def test_grouped_df(self):
        expected_columns = ['PROGRAM_ID', 'PROGRAM', 'CATEGORY', 'KIND', 'Affinity',
                            'profil', 'info', 'subinfo', 'segment_name_tokens', 'score']
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_query_df=lambda spark, options, query: create_program_df_mock(spark),
                                 load_affinities_df=lambda spark: create_affinities_df_mock(spark)):
            program_df = self.job.program_df_job()
            affinities_df = self.job.affinities_df_job()
            grouped_df, grouped_df_subinfo = self.job.grouped_df_job(program_df, affinities_df)
            rows_info = grouped_df.collect()
            rows_subinfo = grouped_df.collect()
            columns_info = grouped_df.columns
            columns_subinfo = grouped_df_subinfo.columns
            # Check expected columns
            self.assertListEqual(columns_info, expected_columns)
            self.assertListEqual(columns_subinfo, expected_columns)
            # Check len row
            self.assertEqual(len(rows_info), 4)
            self.assertEqual(len(rows_subinfo), 4)

    def test_result_df(self):
        expected_columns = ['PROGRAM_ID', 'PROGRAM', 'CATEGORY', 'KIND', 'AFFINITY', 'PROFIL', 'SCORE']
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_query_df=lambda spark, options, query: create_program_df_mock(spark),
                                 load_affinities_df=lambda spark: create_affinities_df_mock(spark)):
            program_df = self.job.program_df_job()
            affinities_df = self.job.affinities_df_job()
            grouped_df, grouped_df_subinfo = self.job.grouped_df_job(program_df, affinities_df)
            result_df = self.job.result_info_job(grouped_df, grouped_df_subinfo)
            rows = result_df.collect()
            columns = result_df.columns
            # Check expected columns
            self.assertListEqual(columns, expected_columns)
            # Check len row
            print(rows)
            self.assertEqual(len(rows), 2)
            for r in rows:
                if r.PROGRAM_ID == 2:
                    self.assertEqual(round(r.SCORE, 3), 0.952)

    def test_fact_audience_aff_only_job(self):
        expected_columns = {'USER_ID', 'date', 'Action & Aventure', 'Banque & Finance', 'Téléréalité'}
        data = {
            "USER_ID": [0, 1, 2, 3],
            "AFFINITY": ["Action & Aventure"] * 4,
            "AFFINITY_RATE": [0.1, 0.9, 0.8, 0.5]
        }
        user_aff_df = create_spark_df_from_data(self.job.spark, data)
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_query_df=lambda spark, options, query: user_aff_df,
                                 load_affinities_df=lambda spark: create_affinities_df_mock(spark)):
            affinities_df = self.job.affinities_df_job()
            df_joined = self.job.fact_audience_aff_only_job(affinities_df)

            df_joined = df_joined. \
                where(F.col("Action & Aventure") > 0.2)

            rows = df_joined.collect()
            columns = df_joined.columns
            # Check expected columns
            self.assertSetEqual(set(columns), expected_columns)
            # Check len row
            self.assertEqual(len(rows), 3)

    def test_fact_audience_aff_only_flag_job(self):
        expected_columns = {'USER_ID', 'date', 'Action & Aventure', 'Cuisine', 'Banque & Finance', "small_watcher"}
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            global_agg_fact_affinity_df = self.job.fact_audience_aff_only_flag_job()
            rows = global_agg_fact_affinity_df.collect()
            columns = global_agg_fact_affinity_df.columns
            # Check expected columns
            self.assertSetEqual(set(columns), expected_columns)
            # Check len row
            self.assertEqual(len(rows), 4)

    def test_fact_audience_aff_regie_job(self):
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            global_agg_fact_affinity_df = self.job.fact_audience_aff_only_flag_job()
            fact_audience_aff_regie_df = self.job.fact_audience_aff_regie_job(global_agg_fact_affinity_df)

            rows = fact_audience_aff_regie_df.collect()
            users = []
            for r in rows:
                if r["Cuisine_F_18+_49-_R45"] == 1:
                    users.append(r.USER_ID)

            # Only 1 user is selected in the new segment
            self.assertEqual(len(users), 1)

            # User 2 is in the new segment
            self.assertEqual(users[0], 2)

    def test_ad_segment_no(self):
        now = datetime.date.today()
        minus_one = datetime.timedelta(days=1)
        aff_watch_table_data = {
            "USER_ID": [1, 1, 1, 1],
            "DATE_DAY": [now, now - minus_one, now - 2 * minus_one, now - 3 * minus_one],
            "watch_duration": [3600, 7200, 3600, 3600]
        }
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_table=lambda *args, **kwargs:
                                 create_spark_df_from_data(self.job.spark, aff_watch_table_data)):
            df = self.job.compute_ad_affinities()
            rows = df.collect()
            user_1 = find_user_rows(rows, 1)[0]
            self.assertEqual(user_1.small_watcher, 0)

    def test_ad_segment_yes(self):
        now = datetime.date.today()
        minus_one = datetime.timedelta(days=1)
        aff_watch_table_data = {
            "USER_ID": [1, 1, 1, 1],
            "DATE_DAY": [now, now - minus_one, now - 2 * minus_one, now - 3 * minus_one],
            "watch_duration": [3600, 3600, 3600, 3600]
        }
        with mock.patch.multiple("databricks_jobs.jobs.fact_table_build_jobs.fact_audience_jobs."
                                 "affinity_segment_job.entrypoint",
                                 load_snowflake_table=lambda *args, **kwargs:
                                 create_spark_df_from_data(self.job.spark, aff_watch_table_data)):
            df = self.job.compute_ad_affinities()
            rows = df.collect()
            user_1 = find_user_rows(rows, 1)[0]
            self.assertEqual(user_1.small_watcher, 1)
