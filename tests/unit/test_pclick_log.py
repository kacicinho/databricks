from unittest import TestCase, mock
import datetime
import os

import pyspark.sql.functions as F

from databricks_jobs.jobs.ml_models_jobs.pclick.pclick_learning_log_job.entrypoint import PClickLearningLogJob
from databricks_jobs.jobs.utils.date_utils import build_datetime_ceiler

from tests.unit.utils.mocks import create_spark_df_from_data, multiplex_mock
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


class TestPClickLearningLogJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.job = PClickLearningLogJob()
        cls.job.LOG_SAMPLING_RATIO = 1.0
        cls.job.N_NEGATIVE_SAMPLING = 1
        cls.job.RANDOM_NEGATIVE_SAMPLING_RATIO = 0.0
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    @classmethod
    def create_fake_epg(cls):
        end_2 = datetime.datetime.now()
        start_2 = end_2 - datetime.timedelta(hours=1)
        end_1 = start_2
        start_1 = end_1 - datetime.timedelta(hours=1)
        data = {
            "PROGRAM_ID": [1, 2, 3, 4, 5],
            "CHANNEL_ID": [1, 1, 2, 2, 3],
            "START_AT": [start_1, start_2, start_1, start_2, start_1],
            "END_AT": [end_1, end_2, end_1, end_2, end_2]
        }
        return create_spark_df_from_data(cls.job.spark, data)

    def test_epg_creation(self):
        df = self.create_fake_epg()

        with mock.patch.multiple("databricks_jobs.db_common",
                                 load_snowflake_table=multiplex_mock):
            rez = self.job.build_full_epg_table(df, self.job.TIME_SAMPLING_FACTOR).orderBy("TIMESTAMP")
            rows = rez.collect()

        self.assertSetEqual(set(rez.columns), {"TIMESTAMP", "CURRENT_EPG"})

        first_row = rows[0]
        epg = first_row.asDict()["CURRENT_EPG"]
        pids = {r.PROGRAM_ID for r in epg}
        self.assertSetEqual(pids, {1, 3, 5})

        first_row = rows[-1]
        epg = first_row.asDict()["CURRENT_EPG"]
        pids = {r.PROGRAM_ID for r in epg}
        self.assertSetEqual(pids, {2, 4, 5})

    def test_combination(self):

        tc = build_datetime_ceiler(self.job.TIME_SAMPLING_FACTOR)
        now = datetime.datetime.now()
        today = now.date()
        tt = tc(now - datetime.timedelta(hours=0.5))

        def create_tnt_segemnt_for_tt(tt):
            data = {
                "TIMESTAMP": [tt] * 25,
                "PROGRAM_ID": list(range(1, 26)),
                "RAIL_ORDER": list(range(1, 26)),
                "CHANNEL_ID": list(range(1, 26)),
            }
            return create_spark_df_from_data(self.job.spark, data). \
                groupBy("TIMESTAMP"). \
                agg(F.collect_set(F.struct("PROGRAM_ID", "RAIL_ORDER", "CHANNEL_ID")).alias("CURRENT_EPG")). \
                withColumn("TIMESTAMP", F.to_timestamp("TIMESTAMP"))

        full_epg_df = create_tnt_segemnt_for_tt(tt)
        data = {
            "TIMESTAMP": [tt, tt, tt, tt],
            "USER_ID": [1, 2, 3, 12],
            "PROGRAM_ID": [1, 1, 3, 4],
            "CLICK_RANK": [1, 1, 14, 145],
            "CHANNEL_ID": [1, 1, 3, 2],
            "EVENT_DATE": [today, today, today, today]
        }
        click_df = create_spark_df_from_data(self.job.spark, data)

        data = {
            "USER_ID": [1, 2],
            "STATUS_DAY_DATE": [today, today],
            "is_premium": [True, True]
        }
        premium_df = create_spark_df_from_data(self.job.spark, data)

        rez = self.job.combine_and_sample(click_df, full_epg_df, premium_df)
        rows = rez.collect()

        # When click rank > 25, drop
        u_rows = find_user_rows(rows, 12)
        self.assertEqual(0, len(u_rows), "Click_rank > 25 should be filtered out")

        # Counting 2 rows per click event
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r.LABEL == 1 for r in rows), 3)
        self.assertEqual(sum(r.LABEL == 0 for r in rows), 3)

        # When click rank > 0, neg_rank < pos_rank
        u_rows = find_user_rows(rows, 3)
        pos_row = find_user_rows(u_rows, 1, "LABEL")
        neg_row = find_user_rows(u_rows, 0, "LABEL")
        self.assertGreater(pos_row[0].RANK, neg_row[0].RANK, "Error when click_rank > top 6")

        # When click rank == 0, neg_rank > pos_rank
        u_rows = find_user_rows(rows, 1)
        pos_row = find_user_rows(u_rows, 1, "LABEL")
        neg_row = find_user_rows(u_rows, 0, "LABEL")
        self.assertGreater(neg_row[0].RANK, pos_row[0].RANK, "Error when click_rank==0")
