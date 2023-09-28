import datetime
from typing import List, Tuple
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
import pandas as pd
import numpy as np

AUTHORIZED_TVBUNDLES = (25, 90, 26, 60, 31, 142, 146)


def next_week_info(spark: SparkSession) -> Tuple[int, datetime.date]:
    """
    returns next week iso week number, and next monday date
    """
    next_week_info_df = (
        spark.range(1)
        .withColumn('next_monday', F.next_day(F.current_date(), 'monday'))
        .withColumn('week_number', F.weekofyear(F.col('next_monday')))
    )
    row = next_week_info_df.collect()
    next_monday_date = row[0][1]
    next_week_number = row[0][2]
    return next_week_number, next_monday_date


def available_broadcast(broadcast: SparkDataFrame, rel_tvbundle_channel: SparkDataFrame, weeks_lookback: int = 52,
                        bundles: tuple = AUTHORIZED_TVBUNDLES) -> SparkDataFrame:
    """
    - next_week starts on next monday and ends on the following sunday (this sundays is START_AT_MAX date)
    - from this sunday we go back 'weeks_lookback' weeks, and that gives us START_AT_MIN date
    - we keep all programs of broadcast table between those two dates, and convert their START_AT to a iso week number.
    - filtering on authorised bundles
    """
    # a channel may belong to multiple bundles : we only keep one pair channel / bundle
    bundle_window = Window.partitionBy('PROGRAM_ID', 'WEEK_START_AT').orderBy(F.col("CHANNEL_ID"), F.col("TVBUNDLE_ID"))
    broadcast_with_bundle = (
        broadcast
        .join(rel_tvbundle_channel, broadcast.CHANNEL_ID == rel_tvbundle_channel.CHANNEL_ID)
        .drop(rel_tvbundle_channel.CHANNEL_ID)
        .select(['PROGRAM_ID', 'START_AT', 'CHANNEL_ID', 'TVBUNDLE_ID'])
    )
    weekly_program_broadcast = (
        broadcast_with_bundle
        .withColumn('next_monday', F.next_day(F.current_date(), 'monday'))
        .withColumn('the_sunday_after', F.next_day(F.current_date(), 'monday') + 6)
        .withColumn('START_AT_MAX', F.col('the_sunday_after'))
        .withColumn('START_AT_MIN', F.col('the_sunday_after') - 7 * weeks_lookback + 1)
        .where(F.to_date(F.col('START_AT')).between(F.col('START_AT_MIN'), F.col('START_AT_MAX')))
        .withColumn('WEEK_START_AT', F.weekofyear(F.col('START_AT')))
        .dropDuplicates(["PROGRAM_ID", "WEEK_START_AT", "CHANNEL_ID", "TVBUNDLE_ID"])
        .where(F.col('TVBUNDLE_ID').isin(*bundles))
        .select(['PROGRAM_ID', 'WEEK_START_AT', 'CHANNEL_ID', 'TVBUNDLE_ID'])
        .withColumn("KIND", F.lit("BROADCAST"))
        .withColumn("chnl_rank_per_bundle", F.row_number().over(bundle_window))
        .where(F.col("chnl_rank_per_bundle") <= 1)
        .select('PROGRAM_ID', 'CHANNEL_ID', 'KIND', 'WEEK_START_AT')
    )
    return weekly_program_broadcast.dropDuplicates()


def available_vod(vod: SparkDataFrame, bundles: tuple = AUTHORIZED_TVBUNDLES) -> SparkDataFrame:
    """
    - filtering on authorised bundles
    - keeping only one occurrence at most for each program
    """
    bundle_window = Window.partitionBy('PROGRAM_ID').orderBy(F.col("CHANNEL_ID"), F.col("TVBUNDLE_ID"))
    res = (
        vod
        .where(F.col('TVBUNDLE_ID').isin(*bundles))
        .select(['PROGRAM_ID', 'CHANNEL_ID', 'TVBUNDLE_ID'])
        .withColumn("chnl_rank_per_bundle", F.row_number().over(bundle_window))
        .where(F.col("chnl_rank_per_bundle") <= 1)
        .withColumn("kind", F.lit("VOD"))
        .select('PROGRAM_ID', 'CHANNEL_ID', 'KIND')
        .dropDuplicates()
    )
    return res


def program_with_tag_info(available_programs: SparkDataFrame, similarities: SparkDataFrame, ref_tag: SparkDataFrame,
                          score_min: float = 0.3) -> SparkDataFrame:
    """
    we join tags info when similarity between program and tag is above 'score_min'
    """
    ref_tag = ref_tag.select(['ID', 'NAME']).withColumnRenamed('NAME', 'KEYWORD')
    similarities = (
        similarities
        .filter(F.col("SIMILARITY_SCORE") >= score_min)
        .orderBy(F.col('PROGRAM_ID'))
        .drop("UPDATE_AT")
    )
    similarities_with_name = similarities.join(ref_tag, ref_tag.ID == similarities.KEYWORD_ID).drop('ID', 'KEYWORD_ID')
    available_programs_with_tags = (
        available_programs
        .join(similarities_with_name, available_programs.PROGRAM_ID == similarities_with_name.PROGRAM_ID)
        .drop(similarities_with_name.PROGRAM_ID)
        .dropDuplicates()
    )
    return available_programs_with_tags


def keep_only_best_programs(df: SparkDataFrame, n: int = 3) -> SparkDataFrame:
    """
    keeps a maximum of n programs per week per channel per keyword,
    so no more thant n programs per channel.is present on the final rail
    """
    channel_window = \
        Window.partitionBy("WEEK_START_AT", "CHANNEL_ID", "KEYWORD").orderBy(F.col("SIMILARITY_SCORE").desc())

    filtered = (
        df
        .withColumn("chnl_sim_rank", F.row_number().over(channel_window))
        .where(F.col("chnl_sim_rank") <= n)
        .orderBy("WEEK_START_AT", "CHANNEL_ID", "KEYWORD")
    )
    return filtered


def keyword_score_per_week(df: SparkDataFrame, rail_size: int = 10) -> pd.DataFrame:
    """
    sums scores of top "rail_size" scores keywords across channels and programs, grouping per week
    """
    score_window = Window.partitionBy("WEEK_START_AT", "KEYWORD").orderBy(F.col("SIMILARITY_SCORE").desc())
    scores = (
        df
        .withColumn("kw_week_sim_rnk", F.row_number().over(score_window))
        .where(F.col("kw_week_sim_rnk") <= rail_size)
        .select("WEEK_START_AT", "KEYWORD", "SIMILARITY_SCORE")
        .groupby("WEEK_START_AT", "KEYWORD")
        .agg(F.sum('SIMILARITY_SCORE').alias('KEYWORD_SCORE'))
        .select('WEEK_START_AT', 'KEYWORD', 'KEYWORD_SCORE')
        .dropDuplicates()
        .toPandas()
    )
    return scores


def normalize_data(array: np.array) -> np.array:
    mean = np.mean(array)
    std = np.std(array)
    centered = (array - mean) / std
    return centered


def ranking_per_week(centered_data: pd.DataFrame) -> pd.DataFrame:
    """
    for each week of centered_data index, we order keywords by their deviation from mean
    """
    week_index = centered_data.index
    ranking = pd.DataFrame(
        [centered_data.loc[i].sort_values(ascending=False).index for i in week_index]).transpose()
    ranking.columns = [f"week_{i}" for i in week_index]
    ranking.index.name = "ranking"
    ranking.index = [i + 1 for i in ranking.index]
    return ranking


def possible_tags(df: SparkDataFrame, rail_size: int, week: int):
    """
    a tag is possible candidate if it is present at least 'rail_size' times (across programs and channels) in a given week
    """
    count_window = Window.partitionBy("WEEK_START_AT", "KEYWORD")
    possible_tags_of_the_week = (
        df
        .withColumn("kw_week_occ", F.count(F.col('SIMILARITY_SCORE')).over(count_window))
        .where(F.col('WEEK_START_AT') == week)
        .where(F.col('kw_week_occ') >= rail_size)
        .withColumn('PROGRAM_ID', F.col('PROGRAM_ID').cast(IntegerType()))
        .withColumn('CHANNEL_ID', F.col('CHANNEL_ID').cast(IntegerType()))
        .toPandas()['KEYWORD'].unique()
    )
    return possible_tags_of_the_week


def programs_of_the_week(best_programs_df, tags: List[str], week: int) -> pd.DataFrame:
    """
    returns all programs present in best_programs_df available at 'week' and having a tag in 'tags'
    """
    res = (
        best_programs_df
        .where(F.col('WEEK_START_AT') == week)
        .where(F.col('KEYWORD').isin(*tags))
        .withColumn('PROGRAM_ID', F.col('PROGRAM_ID').cast(IntegerType()))
        .withColumn('CHANNEL_ID', F.col('CHANNEL_ID').cast(IntegerType()))
        .toPandas()
        .sort_values("SIMILARITY_SCORE", ascending=False)
        .reset_index(drop=True)
    )
    return res


# noinspection PyTypeChecker
def get_ref_table(spark, next_monday: datetime.date, tags: List[str]):
    """
    a reference table with week info (iso number, monday date and top tags)
    NB : next_monday is given as a parameter instead of being computed, to avoid
    incorrect result if job is ran around midnight on sunday
    """
    ref = (
        spark.range(1)
        .withColumn('FIRST_WEEK_DAY', F.lit(next_monday))
        .withColumn('week_number', F.weekofyear(F.col('FIRST_WEEK_DAY')))
        .withColumn('year', F.year(F.col('FIRST_WEEK_DAY')))
        .select('year', 'week_number', 'FIRST_WEEK_DAY')
        .withColumn('tags_of_the_week', F.array([F.lit(string) for string in tags]))
    )
    return ref
