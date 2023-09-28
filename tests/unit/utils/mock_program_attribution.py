import datetime
from datetime import timedelta

import pandas as pd


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    df = df.where(pd.notnull(df), None)
    return spark.createDataFrame(df)


def mock_fact_page(spark):
    now = datetime.datetime.now()
    uids = [1, 1, 1, 2, 2, 2]
    data = {
        "USER_ID": uids,
        "EVENT_DATE": [(now - timedelta(days=1)).date() for _ in uids],
        "TIMESTAMP": [now - timedelta(days=1) + timedelta(hours=i) for i in range(len(uids))],
        "PAGE_NAME": ["program" for _ in uids],
        "ORIGIN_PAGE": ["program" for _ in uids],
        "ORIGIN_SECTION": ["", "", "similar-programs", "", "similar-programs", "similar-programs"],
        "PROGRAM_ID": [12, 12, 8, 46, 47, 48],
        "CHANNEL_ID": [329 for _ in uids]
    }
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    if table_name == "dw.fact_page":
        return mock_fact_page(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
