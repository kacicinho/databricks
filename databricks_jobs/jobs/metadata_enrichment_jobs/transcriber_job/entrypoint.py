import boto3
from botocore.exceptions import ClientError
import pyspark.sql.functions as F
import pandas as pd
from databricks_jobs.common import Job
from pyspark.sql.window import Window
import time
from databricks_jobs.jobs.utils.utils import get_snowflake_options, load_snowflake_table, get_snowflake_connection, \
    write_df_to_snowflake


class TranscribeJob(Job):

    def __init__(self, *args, **kwargs):
        super(TranscribeJob, self).__init__(*args, **kwargs)

        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.region_name = self.conf.get("aws_region_name", "")

        self.options = get_snowflake_options(self.conf, 'PROD', 'EXTERNAL_SOURCES')
        self.conn = get_snowflake_connection(self.conf, 'PROD')

        self.now = self.parse_date_args()

        self.output_bucket = 'mtv-prod-data-audio-output-transcribed'

        self.env = self.conf.get("env")

        if self.env == 'DEV':
            raise Exception('Job can\'t be run in this environment: {}'.format(self.env))

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 Create transcribe client
        transcribe = boto3.client('transcribe', aws_access_key_id=self.aws_access_key,
                                  aws_secret_access_key=self.aws_secret_key,
                                  region_name=self.region_name)

        self.jobs_cleaning(transcribe)

        all_paths = self.files_collection()
        transcribed_files_df, completed_jobs, failed_jobs = self.transcription_files(transcribe, all_paths)

        self.update_transcribed_table(transcribed_files_df, completed_jobs)

        if len(failed_jobs) > 0:
            raise Exception('Jobs Transcription failed : {}'.format(failed_jobs))

    @staticmethod
    def jobs_cleaning(transcribe):
        # Clean remaining jobs
        cleaning = transcribe.list_transcription_jobs()
        list_job_cleaning = cleaning['TranscriptionJobSummaries']
        while len(list_job_cleaning) > 0:
            for elt in list_job_cleaning:
                name = elt['TranscriptionJobName']
                transcribe.delete_transcription_job(TranscriptionJobName=name)
            cleaning = transcribe.list_transcription_jobs()
            list_job_cleaning = cleaning['TranscriptionJobSummaries']

    def files_collection(self):
        # 2 load table with audio files path and program information
        file_df = load_snowflake_table(self.spark, self.options, "EXTERNAL_SOURCES.AUDIO_FILES_S3_PATH"). \
            withColumnRenamed("REF_PROGRAM_CATEGORY_ID", "AUDIO_CATEGORY_ID"). \
            withColumnRenamed("DURATION", "AUDIO_DURATION"). \
            withColumnRenamed("UPDATED_AT", "AUDIO_UPDATED")
        transcribed_df = load_snowflake_table(self.spark, self.options, "EXTERNAL_SOURCES.AUDIO_FILES_TRANSCRIBED")
        broadcast_df = load_snowflake_table(self.spark, self.options, "BACKEND.BROADCAST")
        ref_tvbundle_df = load_snowflake_table(self.spark, self.options, "BACKEND.REL_TVBUNDLE_CHANNEL")
        program_df = load_snowflake_table(self.spark, self.options, "BACKEND.PROGRAM")

        # 3 select a set of episodes :
        # - documentaries
        # - Free, TF1 and M6 bundles
        # - between 20 and 60 min
        # - more recent episode for each program
        # - not transcribed previously
        episode_with_path_df = file_df. \
            join(broadcast_df, broadcast_df.ID == file_df.BROADCAST_ID). \
            drop(broadcast_df.EPISODE_ID). \
            join(ref_tvbundle_df, ref_tvbundle_df.CHANNEL_ID == broadcast_df.CHANNEL_ID). \
            join(program_df, program_df.ID == broadcast_df.PROGRAM_ID). \
            where((F.col("AUDIO_CATEGORY_ID") == 8) & (F.col("TVBUNDLE_ID").isin(*[25, 144, 146])) & (
                F.col("AUDIO_DURATION").between(1200, 3600))). \
            withColumn("num", F.row_number().
                       over(Window.partitionBy(program_df.TITLE).
                            orderBy(F.desc("AUDIO_UPDATED")))). \
            where(F.col("num") == 1)

        # select episodes_id already transcribed
        already_transcribed_df = transcribed_df. \
            select("EPISODE_ID"). \
            toPandas()

        already_transcribed = already_transcribed_df['EPISODE_ID'].unique()

        # drop program already transcribed
        all_paths = episode_with_path_df. \
            select("PATH", "EPISODE_ID"). \
            where(~F.col("EPISODE_ID").isin(*already_transcribed)). \
            limit(50). \
            toPandas()

        return all_paths

    def transcription_files(self, transcribe, all_paths):
        # 4 Start a transcription job for each episode file, the result is stored in S3 bucket
        all_res, job_name_list, failed_jobs = self.execute_transcriber_calls(transcribe, all_paths)
        transcribed_files_df = self.spark.createDataFrame(data=all_res,
                                                          schema=["JOB_NAME", "EPISODE_ID", "INPUT_PATH",
                                                                  "TRANSCRIBED_PATH"])
        status_list = ["IN_PROGRESS"] * len(job_name_list)

        # 5 Wait the end of each jobs (Completed or failed)
        continuing = True
        completed_jobs = []
        while continuing:
            response = transcribe.list_transcription_jobs()
            job_summaries = response['TranscriptionJobSummaries']
            if len(job_summaries) == 0:
                break
            for elt in job_summaries:
                name = elt['TranscriptionJobName']
                status = elt['TranscriptionJobStatus']
                if status == "COMPLETED" or status == "FAILED":
                    transcribe.delete_transcription_job(TranscriptionJobName=name)
                    job_index = job_name_list.index(name)
                    status_list[job_index] = status
            time.sleep(100)

        # 6 Update table with completed transcriptions
        for job_k, status in zip(job_name_list, status_list):
            if status == "COMPLETED":
                completed_jobs = completed_jobs + [job_k]

            if status == "FAILED":
                failed_jobs = failed_jobs + [job_k]

        return transcribed_files_df, completed_jobs, failed_jobs

    def execute_transcriber_calls(self, transcribe, all_paths: pd.DataFrame):
        """
        Schedule all the transcribe calls and handle unvalid S3 paths
        """
        all_res = []
        job_name_list = []
        failed_jobs = []
        for k, x in all_paths.iterrows():
            path = x['PATH']
            episode = x['EPISODE_ID']
            job_name = "transcribe-episode-{}".format(episode)
            output_name = '{}.json'.format(episode)
            try:
                transcribe.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={
                        'MediaFileUri': "s3://" + path
                    },
                    MediaFormat='mp4',
                    OutputBucketName=self.output_bucket,
                    OutputKey=output_name,
                    LanguageCode='fr-FR',
                    JobExecutionSettings={
                        'AllowDeferredExecution': True,
                        'DataAccessRoleArn': 'arn:aws:iam::999326574752:role/mtv-prod-transcriber-role'

                    }
                )
                out_path = '/'.join([self.output_bucket, output_name])
                all_res = all_res + [[job_name, episode, path, out_path]]
                job_name_list = job_name_list + [job_name]

            except ClientError as error:
                if error.response['Error']['Code'] == 'BadRequestException':
                    print("The path {} can't be accessed".format(path))
                    failed_jobs = failed_jobs + [job_name]
                if error.response['Error']['Code'] == 'ConflictException':
                    print("The job {} already_exist".format(job_name))
                else:
                    raise error

        return all_res, job_name_list, failed_jobs

    def update_transcribed_table(self, transcribed_files_df, completed_jobs):
        updating_transcribe_df = transcribed_files_df. \
            where(F.col("JOB_NAME").isin(*completed_jobs)). \
            select("EPISODE_ID", "TRANSCRIBED_PATH")

        write_df_to_snowflake(updating_transcribe_df, self.options, "audio_files_transcribed", "append")


if __name__ == "__main__":
    job = TranscribeJob()
    job.launch()
