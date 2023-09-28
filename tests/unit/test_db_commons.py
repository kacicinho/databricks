from unittest import mock
import os
from unittest import TestCase
from datetime import datetime
import pyspark.sql.functions as F
from tests.unit.utils.mocks import mock_product_to_tvbundle, multiplex_mock

from databricks_jobs.db_common import build_user_to_paying_tv_bundle, build_broadcast_df_with_episode_info, \
    BANNED_KIND_IDS
from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from tests.unit.utils.mocks import mock_backend_broadcast, create_spark_df_from_data, create_detailed_episode_df, \
    mock_rel_channel_table

from pyspark.sql import SparkSession


class TestDBCommon(TestCase):

    @classmethod
    def setUp(self) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        self.job = PopularRecoJob()

    def test_build_user_to_tv_bundle(self):
        with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            df = build_user_to_paying_tv_bundle(
                self.job.spark, self.job.options, mock_product_to_tvbundle(self.job.spark), datetime.now()
            )

        rows = df.collect()
        for r in rows:
            self.assertEqual(r.TVBUNDLE_ID, 60)

    def test_broadcast_df_ban(self):
        """
        Test that we filter out a program with an invalid
        :return:
        """

        def run(cat_id=2, kind_id=34):
            mock_episode_df = create_detailed_episode_df(self.job.spark)
            mock_episode_df.\
                withColumn("PROGRAM_ID", F.lit(0))

            def mock_backend_program(spark):
                data = {
                    "ID": [0],
                    "REF_PROGRAM_CATEGORY_ID": [cat_id],
                    "REF_PROGRAM_KIND_ID": [kind_id],
                    "PRODUCTION_YEAR": [1999],
                    "DURATION": [30 * 60]
                }
                return create_spark_df_from_data(spark, data)

            def local_mock(spark, options, table_name, *args, **kwargs):
                if table_name == "backend.program":
                    return mock_backend_program(spark)
                elif table_name == "backend.broadcast":
                    return mock_backend_broadcast(spark)
                elif table_name == "backend.rel_tvbundle_channel":
                    return mock_rel_channel_table(spark)
                else:
                    raise Exception

            with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                            new=lambda spark, options, table_name: local_mock(spark, options, table_name)):
                df = build_broadcast_df_with_episode_info(self.job.spark, self.job.options, mock_episode_df,
                                                          self.job.now, self.job.now + self.job.delta, 1)
                rows = df.collect()
            return rows

        for k in BANNED_KIND_IDS:
            rows = run(kind_id=k)
            self.assertEqual(len(rows), 0)

        rows = run(kind_id=1)
        self.assertGreater(len(rows), 0)
