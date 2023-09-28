from typing import Any, Tuple
import pandas as pd
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.translation import fr_en_translate
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql import DataFrame as SparkDataFrame, Column as SparkColumn, Window
from pyspark.sql.functions import struct, lit, when, rank, col, monotonically_increasing_id, concat_ws, substring, \
    greatest, length, desc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from databricks_jobs.jobs.utils.slack_plugin import send_msg_to_slack_channel


class TranslationJob(Job):
    TRANSLATED_SUMMARIES_TABLE = "ML.TRANSLATED_SUMMARIES_FR_TO_EN"
    MAX_NB_SUMMARIES_TO_TRANSLATE = 10000
    MAX_STRING_LENGTH = 6000  # 6 letters per word * 1000 is enough
    SERIES_LIKE_PROGRAMS = [2]  # REF_PROGRAM_CATEGORY_ID = 2 -> Série

    def __init__(self, *args, **kwargs):
        super(TranslationJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")
        self.alerting_channel_id = 'C03C5D66E5U'
        self.slack_bot_token = self.dbutils.secrets.get("slack", "scraping_token")
        # the bot used to post on slack is visible at
        # https://molotovtv.slack.com/apps/A02L3KJQJ3W-just-post-bot?settings=1&next_id=0
        # its corresponding token is called "scraping_token"

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 loading of already translated programs table, the ref_cat table and the 3 tables containing summaries:
        program, edito, episode = self.load_data(self.spark, self.options, self.TRANSLATED_SUMMARIES_TABLE)

        # 2a longest summary of each table per PROGRAM_ID
        program_longest_summary = select_longest_summary(df=program, output_prefix='program')
        edito_longest_summary = select_longest_summary(df=edito, output_prefix='edito')
        episode_longest_summary = select_longest_summary(df=episode, output_prefix='episode')

        # 2b and then join it
        summaries = program_longest_summary \
            .join(edito_longest_summary, "PROGRAM_ID", how='left') \
            .join(episode_longest_summary, "PROGRAM_ID", how='left') \
            .fillna(0, subset=['edito_summary_length', 'episode_summary_length']) \
            .fillna('', subset=['program_SUMMARY', 'edito_SUMMARY', 'episode_SUMMARY'])

        # 3 longest summary per PROGRAM_ID of the 3 previous tables and concatenate the 2 other summaries to it.
        concatenated_summaries = self.concat_summaries(summaries, self.SERIES_LIKE_PROGRAMS, self.MAX_STRING_LENGTH) \
            .limit(self.MAX_NB_SUMMARIES_TO_TRANSLATE) \
            .select('PROGRAM_ID', 'FR_SUMMARY')

        # 4 translation
        translated_summaries = self.translate_summaries(self.spark, concatenated_summaries)

        # 5 add date column and write
        write_df_to_snowflake(translated_summaries.withColumn('UPDATE_AT', lit(self.now)),
                              self.write_options,
                              table_name=self.TRANSLATED_SUMMARIES_TABLE,
                              mode="append")

    def translate_summaries(self, spark, concatenated_summaries: SparkDataFrame) -> SparkDataFrame:
        concatenated_summaries = concatenated_summaries \
            .withColumn("PROGRAM_ID", col("PROGRAM_ID").cast(IntegerType()))
        pandas_df = concatenated_summaries.toPandas()

        translation_output = pandas_df.apply(fr_en_translate_series, axis=1, result_type='expand').astype(
            {'ERROR': bool})
        translated_summaries = pd.concat([pandas_df, translation_output], axis='columns')

        translated_summaries_ok = translated_summaries[~translated_summaries['ERROR']]
        translated_summaries_ok = translated_summaries_ok[['PROGRAM_ID', 'EN_SUMMARY']]
        schema = StructType([
            StructField("PROGRAM_ID", IntegerType(), True),
            StructField("EN_SUMMARY", StringType(), True)
        ])
        translated_summaries_df = spark.createDataFrame(data=translated_summaries_ok, schema=schema)

        translated_summaries_error = translated_summaries[translated_summaries['ERROR']]
        error_dict = translated_summaries_error[['PROGRAM_ID', 'FR_SUMMARY']].to_dict('records')
        if error_dict:
            error_msg = 'errors while translating summaries are :\n' + str(error_dict)
            send_msg_to_slack_channel(msg=error_msg, channel_id=self.alerting_channel_id, token=self.slack_bot_token)

        return translated_summaries_df

    @staticmethod
    def concat_summaries(summaries: SparkDataFrame, ref_cat_list_to_exclude: list,
                         max_string_length: int) -> SparkDataFrame:
        """
        * concatenates summaries per PROGRAM_ID, putting the longest one before the others
        * PROGRAM_ID having a category id in ref_cat_list_to_exclude are exclude
        * final summary string is truncated at max_string_length
        """
        concatenated_summaries = summaries \
            .where(~col('REF_PROGRAM_CATEGORY_ID').isin(ref_cat_list_to_exclude)) \
            .withColumn('greatest_summary_length_with_value',
                        row_max_with_name("edito_summary_length", "program_summary_length", "episode_summary_length")) \
            .withColumn('greatest_summary_length', col('greatest_summary_length_with_value')['col']) \
            .withColumn('concatenated_summaries',
                        when(
                            col('greatest_summary_length') == 'edito_summary_length',
                            concat_ws(' ',
                                      col('EDITO_SUMMARY'), col('PROGRAM_SUMMARY'), col('EPISODE_SUMMARY')))
                        .when(
                            col('greatest_summary_length') == 'program_summary_length',
                            concat_ws(' ',
                                      col('PROGRAM_SUMMARY'), col('EDITO_SUMMARY'), col('EPISODE_SUMMARY')))
                        .when(
                            col('greatest_summary_length') == 'episode_summary_length',
                            concat_ws(' ',
                                      col('EPISODE_SUMMARY'), col('EDITO_SUMMARY'), col('PROGRAM_SUMMARY')))) \
            .withColumn('FR_SUMMARY', substring('concatenated_summaries', 0, max_string_length))

        return concatenated_summaries

    @staticmethod
    def load_data(spark, options, already_translated_table) -> Tuple[Any, Any, Any]:
        """
        returns 3 SparkDataFrames corresponding to "massaged" tables :
        backend.program, backend.edito_program, and backend.episode
        """
        already_translated = load_snowflake_table(spark, options, already_translated_table)

        # REF_PROGRAM_CATEGORY (so we don't keep programs that are SERIES_LIKE_PROGRAMS)
        ref_cat = load_snowflake_table(spark, options, "BACKEND.REF_PROGRAM_CATEGORY") \
            .withColumnRenamed('ID', 'REF_PROGRAM_CATEGORY_ID')

        # BACKEND.PROGRAM (we don't keep the ids of already translated programs)
        program = load_snowflake_table(spark, options, "BACKEND.PROGRAM") \
            .withColumnRenamed('ID', 'PROGRAM_ID') \
            .join(already_translated, 'PROGRAM_ID', how='left_anti') \
            .join(ref_cat, 'REF_PROGRAM_CATEGORY_ID') \
            .select('PROGRAM_ID', 'TITLE', 'SUMMARY', 'REF_PROGRAM_CATEGORY_ID', 'NAME')

        # BACKEND.EDITO_PROGRAM
        edito = load_snowflake_table(spark, options, "BACKEND.EDITO_PROGRAM") \
            .select('PROGRAM_ID', 'SUMMARY')

        # BACKEND.EPISODE
        episode = load_snowflake_table(spark, options, "BACKEND.EPISODE") \
            .withColumnRenamed('TEXT_LONG', 'SUMMARY') \
            .select('PROGRAM_ID', 'SUMMARY')

        return program, edito, episode


def fr_en_translate_series(row: pd.Series):
    try:
        translation = fr_en_translate(row['FR_SUMMARY'])
        error = False
    except IndexError:
        translation = ''
        error = True
    return {'EN_SUMMARY': translation, 'ERROR': error}


def row_max_with_name(*cols: str) -> SparkColumn:
    """
    returns a row with a struct
    {'value': max value , 'col': name of the column of max value amongst cols}
    """
    cols_ = [struct(col(c).alias("value"), lit(c).alias("col")) for c in cols]
    return greatest(*cols_)


def select_longest_summary(df: SparkDataFrame, output_prefix: str) -> SparkDataFrame:
    """
    (df must contain a column named PROGRAM_ID and a column named SUMMARY)
    groups by PROGRAM_ID and keep the longest corresponding SUMMARY
    (same PROGRAM_ID can have different summaries depending on multiple factors like channel of broadcasting, etc)
    the resulting column is named "output_prefix_SUMMARY" and the length of this summary is stored in a column named
    "output_prefix_summary_length"
    """
    window = Window.partitionBy("PROGRAM_ID").orderBy(desc("summary_length"), 'tiebreak')
    result = (df.fillna('')
              .withColumn('summary_length', length('SUMMARY'))
              .withColumn('tiebreak', monotonically_increasing_id())
              .withColumn('rank', rank().over(window))
              .filter(col('rank') == 1)
              .drop('rank', 'tiebreak')
              .withColumnRenamed('SUMMARY', output_prefix + '_SUMMARY')
              .withColumnRenamed('summary_length', output_prefix + '_summary_length'))

    return result


if __name__ == "__main__":
    job = TranslationJob()
    job.launch()
