import datetime
from datetime import timedelta

import pandas as pd


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    df = df.where(pd.notnull(df), None)
    return spark.createDataFrame(df)


def mock_fact_watch(spark):
    """
    Only program 0 and 1 will have watch history
    """
    now = datetime.datetime.now()
    uids = [1, 1, 2, 2, 2]
    data = {
        'USER_ID': uids,
        'RECEIVED_AT': [now - timedelta(days=i * 2) for i in range(len(uids))],
        'PROGRAM_ID': [1 for _ in uids],
        'CHANNEL_ID': [1 for _ in uids],
        'DURATION': [30 * 60 for _ in uids]
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_page(spark):
    now = datetime.datetime.now()
    uids = [1, 1, 1, 2, 2, 2]
    data = {
        "USER_ID": uids,
        "EVENT_DATE": [(now - timedelta(days=3)).date() for _ in uids],
        "TIMESTAMP": [now - timedelta(days=3) + timedelta(hours=i) for i in range(len(uids))],
        "PAGE_NAME": ["program", "program", "program", "person", "program", "program"],
        "ORIGIN_PAGE": ["home" for _ in uids],
        "ORIGIN_SECTION": ['this-week-recommendations' for _ in uids],
        "ORIGIN_COMPONENT_RANK": list(range(len(uids))),
        "PROGRAM_ID": [2, 1, 1, None, 1, 1],
        "CHANNEL_ID": [1, 1, 1, None, 1, 1],
        "PERSON_ID": [None, None, None, 1, None, None]
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_bookmark_follow(spark):
    uids = [1, 2, 2]
    now = datetime.datetime.now()
    data = {
        "USER_ID": uids,
        "EVENT_NAME": ['bookmark_added', 'follow_person', 'bookmark_added'],
        "TIMESTAMP": [now - timedelta(days=1) + timedelta(hours=i) for i in range(len(uids))],
        "PROGRAM_ID": [1, None, 1],
        "CHANNEL_ID": [1, None, 1],
        "PERSON_ID": [None, 1, None]
    }
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    if table_name == "dw.fact_watch":
        return mock_fact_watch(spark)
    elif table_name == "dw.fact_page":
        return mock_fact_page(spark)
    elif table_name == "dw.fact_bookmark_follow":
        return mock_fact_bookmark_follow(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
