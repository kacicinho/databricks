from unittest import TestCase
from unittest import mock
import os

from databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint import PopularPersonalityJob
from tests.unit.utils.mocks import multiplex_mock, mock_keep_person_id_query
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


class TestPopularPersonalityJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        # Only 1 partition to speed up test
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        cls.job = PopularPersonalityJob(spark=spark)

    def test_freely_available_programs(self):
        """
        Test the pipeline where programs are extracted and filtered based on information like duration and other
        """
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):

                rows = self.job.prepare_freely_available_programs_df().collect()
                prog_ids = [r.PROGRAM_ID for r in rows]
                # Program 3 and 4 should be filtered by condition on duration and program kind
                self.assertSetEqual(set(prog_ids), {0, 1, 2, 5})

    def test_personality_available(self):
        """
        Test the pipeline where personality are extracted and filtered from available programs
        """
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock,
                                     load_snowflake_query_df=lambda spark, options, query: mock_keep_person_id_query(spark)):

                progs_df = self.job.prepare_freely_available_programs_df()

                rows = self.job.personality_available(progs_df).collect()
                person_ids = [r.PERSON_ID for r in rows]
                self.assertSetEqual(set(person_ids), {1, 2, 3})

    def test_most_followed_person(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            rows = self.job.prepare_most_followed_person().collect()
            follows_person_1 = find_user_rows(rows, user_id=1, field="PERSON_ID")[0].tot_fols
            follows_person_4 = find_user_rows(rows, user_id=4, field="PERSON_ID")[0].tot_fols
            # Personality 1 and 4 have respectively 2 and 1 followers
            self.assertEqual([follows_person_1, follows_person_4], [2, 1])

    def test_next_week_duration(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                progs_df = self.job.prepare_freely_available_programs_df()

                rows = self.job.next_week_duration(progs_df).collect()
                next_week_person_1 = find_user_rows(rows, user_id=1, field="PERSON_ID")[0].nxt_wk_dur
                # Personality 1 will appear 4 times in program 0 and 1 time in program 1 next week
                self.assertEqual(next_week_person_1, (10 * 60 + 1) + (10 * 60 + 1))

    def test_general_perso(self):

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock,
                                     load_snowflake_query_df=lambda spark, options, query: mock_keep_person_id_query(spark)):

                freely_available_programs_df = self.job.prepare_freely_available_programs_df()
                personality_available_df = self.job.personality_available(freely_available_programs_df)
                future_personality_with_infos_df = self.job.join_info_to_person(freely_available_programs_df,
                                                                                personality_available_df)
                rows = self.job.build_non_personalized_reco(future_personality_with_infos_df).collect()
                person_ids = [r.PERSON_ID for r in rows]
                self.assertEqual(set(person_ids), {1, 2, 3})

    def test_penalization_score(self):

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            user_penalized_df = self.job.build_user_reco_history()
            rows = user_penalized_df.collect()

            # 4 person ids for user 0
            self.assertEqual(len(rows), 4)

            # The most viewed person_id for user id 0 is the person_id 3. It's score must then be equal to 1.
            pen_score = [r.penalization_score_norm for r in rows if r.USER_ID == 0 and r.PERSON_ID == 3]
            self.assertEqual(len(pen_score), 1)  # No duplicates are allowed
            self.assertEqual(pen_score[0], 1.0)

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_personality_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     write_df_to_snowflake=lambda df, *args, **kwargs: df.show()):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_table=multiplex_mock,
                                         load_snowflake_query_df=lambda spark, options, query: mock_keep_person_id_query(spark)):
                    self.job.launch()
