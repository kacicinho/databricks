from pyspark.sql import functions as F
from databricks_jobs.common import Job
from pyspark.sql.window import Window
from databricks_jobs.jobs.utils.utils import write_df_to_snowflake, get_snowflake_options, load_snowflake_table
from pyspark.sql.types import StringType
from py4j.protocol import Py4JJavaError


class AudioFileRecoveryJob(Job):

    def __init__(self, *args, **kwargs):
        super(AudioFileRecoveryJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD", "EXTERNAL_SOURCES")
        self.path = "mtv-prod-assets-vod/vod/usp/"

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        audio_with_info_df = self.build_audio_with_info()
        audio_with_new_info_df = self.build_audio_path(audio_with_info_df)
        self.update_snowflake_table(audio_with_new_info_df)

    def build_audio_with_info(self):
        # File is available if still present on S3
        is_available_file = (F.col("S3") == 1) & (F.col("DELETED_AT").isNull())

        # We load tables :
        # - asset_manifest (contain all the information of stored program)
        # - asset_audio (give specific information of audio files)
        # - broadcast_df and replay_df (give specific information of episode)
        asset_manifest_df = load_snowflake_table(self.spark, self.options, "backend.asset_manifest"). \
            where(is_available_file). \
            withColumnRenamed("CREATED_AT", "MANIFEST_CREATED_AT").replace(float('nan'), None)
        asset_audio_df = load_snowflake_table(self.spark, self.options, "backend.asset_audio"). \
            where((F.col("LANGUAGE") == 'fre') & (F.col("EXTENSION") == "isma")). \
            withColumnRenamed("BASENAME", "FILE_BASENAME")
        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        episode_df = load_snowflake_table(self.spark, self.options, "backend.episode")
        broadcast_df = load_snowflake_table(self.spark, self.options, "backend.broadcast")
        vod_df = load_snowflake_table(self.spark, self.options, "backend.vod")

        # Gathering information from tables to build the path and find the associated episode_id :
        # -with broadcast
        # -with replay
        audio_with_broadcast_info_df = asset_manifest_df. \
            join(asset_audio_df, asset_audio_df.ASSET_MANIFEST_ID == asset_manifest_df.ID). \
            join(broadcast_df, broadcast_df.ID == asset_manifest_df.BROADCAST_ID). \
            join(program_df, program_df.ID == broadcast_df.PROGRAM_ID). \
            select("S3", "VERSION", "BASENAME", "FILE_BASENAME", "LANGUAGE",
                   asset_audio_df.EXTENSION.alias("FILE_EXTENSION"), asset_audio_df.DURATION,
                   "EPISODE_ID", "REF_PROGRAM_CATEGORY_ID", "BROADCAST_ID", "REPLAY_ID", "MANIFEST_CREATED_AT")

        audio_with_replay_info_df = asset_manifest_df. \
            join(asset_audio_df, asset_audio_df.ASSET_MANIFEST_ID == asset_manifest_df.ID). \
            join(vod_df, vod_df.ASSET_MANIFEST_ID == asset_manifest_df.ID). \
            join(episode_df, episode_df.ID == vod_df.EPISODE_ID). \
            join(program_df, episode_df.PROGRAM_ID == program_df.ID). \
            where(asset_manifest_df.BROADCAST_ID.isNull()). \
            select("S3", "VERSION", "BASENAME", "FILE_BASENAME", "LANGUAGE",
                   asset_audio_df.EXTENSION.alias("FILE_EXTENSION"), asset_audio_df.DURATION,
                   "EPISODE_ID", "REF_PROGRAM_CATEGORY_ID", asset_manifest_df.BROADCAST_ID, "REPLAY_ID",
                   "MANIFEST_CREATED_AT")

        return audio_with_broadcast_info_df. \
            union(audio_with_replay_info_df). \
            withColumn("recency", F.row_number().
                       over(Window.partitionBy("EPISODE_ID").orderBy(F.desc("MANIFEST_CREATED_AT"))))

    def build_audio_path(self, audio_with_info_df):

        def process_basename(x):
            # this function extract the name of S3 folder based on the 6 last characters of manifest basename
            folder_composed = "/".join([x[-2:], x[-4:-2], x[-6:-4], ""])
            return folder_composed

        # We build the path of S3 files using :
        # -version
        # -manifest basename
        # -audio file basename
        udf_folder = F.udf(lambda x: process_basename(x), StringType())
        audio_with_new_info_df = audio_with_info_df. \
            where(F.col("recency") == 1). \
            withColumn("FOLDER", udf_folder(F.col("BASENAME"))). \
            withColumn("PATH", F.concat(F.lit(self.path), F.lit('v'), F.col("VERSION"), F.lit('/'), F.col("FOLDER"),
                                        F.col("BASENAME"), F.lit('/'), F.col("FILE_BASENAME"), F.lit('.'),
                                        F.col("FILE_EXTENSION"))). \
            select("S3", "VERSION", "BASENAME", "FILE_BASENAME", "LANGUAGE",
                   "FILE_EXTENSION", "DURATION",
                   "EPISODE_ID", "REF_PROGRAM_CATEGORY_ID", "BROADCAST_ID", "REPLAY_ID", "MANIFEST_CREATED_AT", "PATH")

        return audio_with_new_info_df

    def update_snowflake_table(self, audio_with_new_info_df):

        try:
            audio_with_old_info_df = load_snowflake_table(self.spark, self.options,
                                                          "EXTERNAL_SOURCES.AUDIO_FILES_S3_PATH"). \
                select("S3", "VERSION", "BASENAME", "FILE_BASENAME", "LANGUAGE",
                       "FILE_EXTENSION", "DURATION",
                       "EPISODE_ID", "REF_PROGRAM_CATEGORY_ID", "BROADCAST_ID", "REPLAY_ID", "MANIFEST_CREATED_AT", "PATH")
        except Py4JJavaError:
            schema = audio_with_new_info_df.schema
            audio_with_old_info_df = self.spark.createDataFrame([], schema)

        # Updating files list with new available
        update_info_df = audio_with_new_info_df.subtract(audio_with_old_info_df). \
            withColumn('UPDATED_AT', F.lit(self.now))

        write_df_to_snowflake(update_info_df, self.options, "AUDIO_FILES_S3_PATH", "append")


if __name__ == "__main__":
    job = AudioFileRecoveryJob()
    job.launch()
