from unittest import TestCase
from unittest import mock
import os
import numpy as np
from databricks_jobs.jobs.metadata_enrichment_jobs.keywords_embeddings_job.entrypoint import KeywordsEmbeddingsJob
from tests.unit.utils.mocks_keywords_embeddings import multiplex_mock
from pyspark.sql import SparkSession


class TestKeywordsEmbeddingsJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = KeywordsEmbeddingsJob()

    def test_make_pandas_result_df(self):
        global mock_keywords_ids
        keywords_ids = np.array([1, 2])
        vec1 = np.arange(768)
        vec2 = np.arange(1, 769)
        embeddings = [vec1, vec2]
        res = self.job.make_pandas_result_df(keywords_ids, embeddings)
        self.assertEqual(res.at[0, 'EMBEDDING_BYTES'], vec1.dumps())
        self.assertEqual(res.at[1, 'EMBEDDING_BYTES'], vec2.dumps())
        self.assertEqual(list(res.columns), ['ID', 'EMBEDDING', 'EMBEDDING_BYTES', 'UPDATE_AT'])

    def test_get_already_embeded_keywords_ids(self):
        with mock.patch("databricks_jobs.jobs.metadata_enrichment_jobs.keywords_embeddings_job."
                        "entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            res = self.job.get_already_embeded_keywords_ids().toPandas()
            self.assertEqual(res['ID'].values.tolist(), [1, 2])

    def test_get_keywords_ids_and_keywords(self):
        with mock.patch("databricks_jobs.jobs.metadata_enrichment_jobs.keywords_embeddings_job."
                        "entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            ids, kws = self.job.get_keywords_ids_and_keywords()
            self.assertTrue((ids == np.array([3, 4])).all())
            self.assertTrue((kws == np.array(['chien', 'poule'])).all())
