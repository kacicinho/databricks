from unittest import TestCase
from unittest import mock
import os
import numpy as np
from databricks_jobs.jobs.metadata_enrichment_jobs.translation_job.entrypoint import TranslationJob, \
    select_longest_summary
from tests.unit.utils.mocks_translation import multiplex_mock, program_summary, edito_summary, episode_summary
from pyspark.sql import SparkSession


class TestTranslationJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = TranslationJob()

    def test_select_longest_summary(self):
        with mock.patch("databricks_jobs.jobs.metadata_enrichment_jobs.translation_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            program_id_to_test = 1

            # mocked data loading
            program, edito, episode = self.job.load_data(self.job.spark, self.job.options,
                                                         self.job.TRANSLATED_SUMMARIES_TABLE)

            # true values given mocked data
            program_pandas = program.toPandas()
            summaries = program_pandas[program_pandas['PROGRAM_ID'] == program_id_to_test]['SUMMARY'].values
            string_lengths = np.vectorize(len)
            summaries_length = string_lengths(summaries)
            longest_summary_index = np.argmax(summaries_length)
            true_longest_summary = summaries[longest_summary_index]

            # tested function results
            computed_longest_summary_df = select_longest_summary(df=program, output_prefix='program').toPandas()
            computed_longest_summary = \
                computed_longest_summary_df[computed_longest_summary_df['PROGRAM_ID'] == program_id_to_test][
                    'program_SUMMARY'].values[0]

            # equality test
            self.assertEqual(true_longest_summary, computed_longest_summary)

    def test_concat_summaries(self):
        with mock.patch("databricks_jobs.jobs.metadata_enrichment_jobs.translation_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            # mocked data loading
            program, edito, episode = self.job.load_data(self.job.spark, self.job.options,
                                                         self.job.TRANSLATED_SUMMARIES_TABLE)
            # longest summary selection
            program_longest_summary = select_longest_summary(df=program, output_prefix='program')
            edito_longest_summary = select_longest_summary(df=edito, output_prefix='edito')
            episode_longest_summary = select_longest_summary(df=episode, output_prefix='episode')

            # joined data
            summaries = program_longest_summary \
                .join(edito_longest_summary, "PROGRAM_ID", how='left') \
                .join(episode_longest_summary, "PROGRAM_ID", how='left') \
                .fillna(0, subset=['edito_summary_length', 'episode_summary_length']) \
                .fillna('', subset=['program_SUMMARY', 'edito_SUMMARY', 'episode_SUMMARY'])

            # true value given mocked data
            true_value = program_summary + ' ' + edito_summary + ' ' + episode_summary

            # test function result
            concatenated_summaries = self.job.concat_summaries(summaries, self.job.SERIES_LIKE_PROGRAMS,
                                                               self.job.MAX_STRING_LENGTH)
            computed_value = concatenated_summaries.toPandas()['FR_SUMMARY'].values[0]

            # equality test
            self.assertEqual(computed_value, true_value)

    def test_translated_summaries(self):
        with mock.patch("databricks_jobs.jobs.metadata_enrichment_jobs.translation_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            # mocked data loading
            program, edito, episode = self.job.load_data(self.job.spark, self.job.options,
                                                         self.job.TRANSLATED_SUMMARIES_TABLE)
            # longest summary selection
            program_longest_summary = select_longest_summary(df=program, output_prefix='program')
            edito_longest_summary = select_longest_summary(df=edito, output_prefix='edito')
            episode_longest_summary = select_longest_summary(df=episode, output_prefix='episode')

            # joined data
            summaries = program_longest_summary \
                .join(edito_longest_summary, "PROGRAM_ID", how='left') \
                .join(episode_longest_summary, "PROGRAM_ID", how='left') \
                .fillna(0, subset=['edito_summary_length', 'episode_summary_length']) \
                .fillna('', subset=['program_SUMMARY', 'edito_SUMMARY', 'episode_SUMMARY'])

            # concatenation
            concatenated_summaries = self.job.concat_summaries(summaries, self.job.SERIES_LIKE_PROGRAMS,
                                                               self.job.MAX_STRING_LENGTH)

            # true translation given mocked databricks_jobs
            true_translation = 'dog and cat and pig. rat and mouse. horse.'

            # test function result
            translated_summaries = self.job.translate_summaries(self.job.spark, concatenated_summaries)
            computed_translation = translated_summaries.toPandas().at[0, 'EN_SUMMARY']

            # equality test
            self.assertEqual(computed_translation.lower(), true_translation)
