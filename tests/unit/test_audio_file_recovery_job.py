from unittest import TestCase
from unittest import mock
import os
from tests.unit.utils.mock_transcribe import multiplex_mock
from databricks_jobs.jobs.metadata_enrichment_jobs.audio_file_recovery_job.entrypoint import AudioFileRecoveryJob
from tests.unit.utils.test_utils import find_user_rows
from pyspark.sql import SparkSession


class TestAudioFileRecoveryJOB(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        cls.job = AudioFileRecoveryJob()

    def test_build_audio_with_info(self):
        with mock.patch.multiple("databricks_jobs.jobs.metadata_enrichment_jobs.audio_file_recovery_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            audio_with_info_df = self.job.build_audio_with_info()
            rez = self.job.build_audio_path(audio_with_info_df)

            rows = rez.collect()
            # Only 2 files are kept, the most recent file for episode 5 and unique available file for episode 15
            self.assertEqual(len(rows), 2)

            file_path = find_user_rows(rows, user_id=5, field="EPISODE_ID")[0].PATH
            self.assertSetEqual(set(file_path),
                                set(
                                    "mtv-prod-assets-vod/vod/usp/v2/e5/59/30/6c4c7734d45f1ee0772a7eeb9eb7677bc0c3059e5/b43d3a99ca11eb2ef9b654b69f8d5564cb4df2e35.isma"))
