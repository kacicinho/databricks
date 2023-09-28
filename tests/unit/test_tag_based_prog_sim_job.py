from unittest import TestCase
from unittest import mock
import os
import subprocess
import datetime

from tests.unit.utils.mocks_tag import multiplex_mock
from databricks_jobs.jobs.program_similarity_jobs.tag_based_prog_sim_job.entrypoint import TagBasedProgramSimilarityJob
from tests.unit.utils.mocks import create_spark_df_from_data
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


class TestTagBasedProgramSimilarityJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        subprocess.run(["python3", "-m", "spacy", "download", "fr_core_news_lg"])
        cls.job = TagBasedProgramSimilarityJob()

        now = datetime.datetime.now()
        broadcast_data = {
            "PROGRAM_ID": [0, 1, 1, 0, 2],
            "EPISODE_ID": [1, 2, 3, 4, 5],
            "START_AT": [now, now, now, now, now]
        }
        tag_data = {
            "EPISODE_ID": [1, 2, 3, 4, 5],
            "REF_TAG_ID": [0, 1, 2, 0, 0]
        }
        cls.broadcast_df = create_spark_df_from_data(cls.job.spark, broadcast_data)
        cls.tag_df = create_spark_df_from_data(cls.job.spark, tag_data)

    def test_program_sim_matrix_gen(self):
        with mock.patch.multiple("databricks_jobs.db_common",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.jobs.program_similarity_jobs.tag_based_prog_sim_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     write_df_to_snowflake=lambda df, *args, **kwargs: None):

                rez = self.job.build_program_sim_matrix(self.broadcast_df, self.tag_df)
                rows = rez.collect()
                # We have 3 programs so 6 combination of similarity
                self.assertEqual(len(rows), 6)

                # Most similar to program 0 is program 2 with 1 common tag and same kind
                example = rows[0]
                self.assertEqual(example.RECOMMENDED_PROGRAM_ID, 2)

                # Test latest reco output
                latest = self.job.write_recos_to_snowflake(rez, "test")
                rows = latest.collect()
                self.assertEqual(len(rows), 3)
                row = find_user_rows(rows, 1, "PROGRAM_ID")
                recommended_program = find_user_rows(row[0].recommendations, 0, "program_id")
                self.assertEqual(round(recommended_program[0].rating, 2), 0)

    def test_program_sim_matrix_gen_with_summary(self):
        tag_data = {
            "EPISODE_ID": [1000, 1001, 1002, 1003, 1004],
            "REF_TAG_ID": [0, 1, 2, 0, 0]
        }
        tag_df = create_spark_df_from_data(self.job.spark, tag_data)
        with mock.patch.multiple("databricks_jobs.db_common",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.jobs.program_similarity_jobs.tag_based_prog_sim_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     write_df_to_snowflake=lambda df, *args, **kwargs: None):
                rez = self.job.build_program_sim_matrix(self.broadcast_df, tag_df, description_similarity=True)

        rows = rez.collect()
        self.assertEqual(len(rows), 6)

        prog_0_rows = find_user_rows(rows, 0, "PROGRAM_ID")
        prog_0_rank_1 = find_user_rows(prog_0_rows, 1, "rank")[0]
        # Given the summary of the program, we expect
        # "Le feu fait ravage" and "Une tour en feu" to be closer to each other rather than "Amour et rire"
        self.assertEqual(prog_0_rank_1.RECOMMENDED_PROGRAM_ID, 1)
