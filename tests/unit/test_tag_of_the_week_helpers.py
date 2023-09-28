from databricks_jobs.jobs.algotorial.tag_of_the_week_job.helpers import \
    available_broadcast, available_vod, \
    program_with_tag_info, keep_only_best_programs, keyword_score_per_week, \
    normalize_data, ranking_per_week, possible_tags, programs_of_the_week, next_week_info
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
now = datetime.now()
one_week_from_now = now + timedelta(days=7)
two_weeks_from_now = now + timedelta(days=14)
four_weeks_ago = now - timedelta(days=28)
two_years_ago = now - timedelta(days=730)
today = now.date()


def test_next_week_info():
    next_week_number, next_monday_date = next_week_info(spark)
    desired_next_monday = today + timedelta(7 - today.weekday())
    desired_next_week_number = desired_next_monday.isocalendar()[1]
    assert next_monday_date == desired_next_monday
    assert next_week_number == desired_next_week_number
    return


def test_available_broadcast_filter_dates():
    """
    programs of ids 2 and 3 should be filtered out since they are out of time span of interest
    """
    broadcast = spark.createDataFrame(pd.DataFrame([
        {'PROGRAM_ID': 1, 'START_AT': one_week_from_now, 'CHANNEL_ID': 10},
        {'PROGRAM_ID': 2, 'START_AT': two_weeks_from_now, 'CHANNEL_ID': 10},
        {'PROGRAM_ID': 3, 'START_AT': two_years_ago, 'CHANNEL_ID': 10},
        {'PROGRAM_ID': 4, 'START_AT': four_weeks_ago, 'CHANNEL_ID': 10}
    ]))
    rel_tvbundle_channel = spark.createDataFrame(pd.DataFrame([{'TVBUNDLE_ID': 1, 'CHANNEL_ID': 10}]))
    res = available_broadcast(broadcast, rel_tvbundle_channel, weeks_lookback=52, bundles=(1,)).toPandas()
    assert set(res['PROGRAM_ID'].values) == {1, 4}
    return


def test_available_broadcast_drop_duplicates_programs():
    """
    program of id 1, on channel of id 10 is present 8 consecutive days ("crossing" exactly two weeks),
    so once START_AT is converted to iso week number, only two distinct lines should be kept
    """
    broadcast = spark.createDataFrame(pd.DataFrame(
        [{'PROGRAM_ID': 1, 'START_AT': now + timedelta(days=i), 'CHANNEL_ID': 10} for i in range(8)]
    ))
    rel_tvbundle_channel = spark.createDataFrame(pd.DataFrame([{'TVBUNDLE_ID': 1, 'CHANNEL_ID': 10}]))
    res = available_broadcast(broadcast, rel_tvbundle_channel, weeks_lookback=52, bundles=(1,)).toPandas()
    assert (len(res.index) == 2)
    return


def test_available_broadcast_drop_duplicates_channel():
    """
    for same program, same week:
    same channel on two different bundles : only one line is to be kept
    """
    broadcast = spark.createDataFrame(pd.DataFrame([
        {'PROGRAM_ID': 1, 'START_AT': one_week_from_now, 'CHANNEL_ID': 10},
        {'PROGRAM_ID': 1, 'START_AT': one_week_from_now, 'CHANNEL_ID': 10}
    ]))
    rel_tvbundle_channel = spark.createDataFrame(pd.DataFrame([
        {'TVBUNDLE_ID': 1, 'CHANNEL_ID': 10},
        {'TVBUNDLE_ID': 2, 'CHANNEL_ID': 10}
    ]))
    res = available_broadcast(broadcast, rel_tvbundle_channel, weeks_lookback=52, bundles=(1,)).toPandas()
    assert len(res.index) == 1
    return


def test_available_vod_prohibited_bundle():
    """
    one of the two bundles is not allowed
    """
    vod = spark.createDataFrame(pd.DataFrame([
        {'PROGRAM_ID': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 10},
        {'PROGRAM_ID': 2, 'TVBUNDLE_ID': 2, 'CHANNEL_ID': 12}
    ]))
    res = available_vod(vod, bundles=(1,)).toPandas()
    assert len(res.index) == 1
    return


def test_program_with_tag_info():
    available_programs = spark.createDataFrame(pd.DataFrame([
        {'PROGRAM_ID': 1, 'CHANNEL_ID': 10, 'KIND': 'VOD'},
        {'PROGRAM_ID': 2, 'CHANNEL_ID': 10, 'KIND': 'VOD'}
    ]))
    similarities = spark.createDataFrame(pd.DataFrame([

        {'PROGRAM_ID': 1, 'KEYWORD_ID': 2, 'SIMILARITY_SCORE': 0.1},
        {'PROGRAM_ID': 1, 'KEYWORD_ID': 3, 'SIMILARITY_SCORE': 0.4}
    ]))
    ref_tag = spark.createDataFrame(pd.DataFrame([
        {'ID': 2, 'NAME': 'cat'},
        {'ID': 3, 'NAME': 'dog'}
    ]))
    res = program_with_tag_info(available_programs, similarities, ref_tag, score_min=0.3).toPandas()
    desired_res = pd.DataFrame([
        {'PROGRAM_ID': 1, 'CHANNEL_ID': 10, 'KIND': 'VOD', 'SIMILARITY_SCORE': 0.4, 'KEYWORD': 'dog'}
    ])
    pd.testing.assert_frame_equal(res, desired_res)
    return


def test_keep_only_best_programs():
    df = spark.createDataFrame(pd.DataFrame([
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 1, 'PROGRAM_ID': 10, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.5},
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 1, 'PROGRAM_ID': 10, 'KEYWORD': 'dog', 'SIMILARITY_SCORE': 0.5}
    ]))
    res = keep_only_best_programs(df, n=1).toPandas()
    assert len(res.index) == 2

    df = spark.createDataFrame(pd.DataFrame([
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 1, 'PROGRAM_ID': 10, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.5},
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 1, 'PROGRAM_ID': 11, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.5}
    ]))
    res = keep_only_best_programs(df, n=1).toPandas()
    assert len(res.index) == 1
    return


def test_keyword_score_per_week():
    df = spark.createDataFrame(pd.DataFrame([
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 1, 'PROGRAM_ID': 10, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.6},
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 2, 'PROGRAM_ID': 11, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.5},
        {'WEEK_START_AT': 2, 'CHANNEL_ID': 3, 'PROGRAM_ID': 12, 'KEYWORD': 'cat', 'SIMILARITY_SCORE': 0.4}
    ]))
    res = keyword_score_per_week(df, rail_size=2)
    assert (res.at[0, 'KEYWORD_SCORE'] == 1.1)
    return


def test_normalize_data():
    df = pd.DataFrame([
        {'cat': 0, 'dog': 0},
        {'cat': 0, 'dog': 4},
        {'cat': 2, 'dog': 4},
        {'cat': 2, 'dog': 16},
    ])
    res = df.apply(normalize_data, axis=0)
    np.testing.assert_array_almost_equal(res['cat'].values,
                                         [-1, -1, 1, 1])
    np.testing.assert_array_almost_equal(res['dog'].values,
                                         [-1, -1 / 3, -1 / 3, 5 / 3])
    return


def test_get_ranking_per_week():
    centered_data = pd.DataFrame([
        {'WEEK_START_AT': 1, 'cat': 13.0, 'dog': 7.0},
        {'WEEK_START_AT': 2, 'cat': 3.0, 'dog': 6.0}
    ]).set_index('WEEK_START_AT')

    res = ranking_per_week(centered_data)
    desired_res = pd.DataFrame([
        {'week_1': 'cat', 'week_2': 'dog'},
        {'week_1': 'dog', 'week_2': 'cat'}
    ])
    desired_res.index = [1, 2]
    pd.testing.assert_frame_equal(res, desired_res)
    return


def test_possible_tags():
    df = spark.createDataFrame(pd.DataFrame([
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'cat',
         'SIMILARITY_SCORE': 0.4},
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 2, 'KEYWORD': 'cat',
         'SIMILARITY_SCORE': 0.5},
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 2, 'KEYWORD': 'dog',
         'SIMILARITY_SCORE': 0.5}
    ]))
    res = possible_tags(df, rail_size=2, week=1)
    assert set(res) == {'cat'}


def test_programs_of_week():
    progs = spark.createDataFrame(pd.DataFrame([
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'cat',
         'SIMILARITY_SCORE': 0.4},
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'cat',
         'SIMILARITY_SCORE': 0.4},
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'dog',
         'SIMILARITY_SCORE': 0.4},
        {'WEEK_START_AT': 1, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'pig',
         'SIMILARITY_SCORE': 0.4},
        {'WEEK_START_AT': 2, 'TVBUNDLE_ID': 1, 'CHANNEL_ID': 1, 'PROGRAM_ID': 1, 'KEYWORD': 'cat',
         'SIMILARITY_SCORE': 0.4}
    ]))
    res = programs_of_the_week(progs, tags=['dog', 'cat'], week=1)
    # week 2 info should be discarded, as well as pig keyword
    assert len(res.index) == 3
    assert set(res['KEYWORD'].values) == {'cat', 'dog'}
