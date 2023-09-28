import pandas as pd
import numpy as np


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_ref_keyword_table(spark):
    data = {"ID": [1,
                   2,
                   3,
                   4],
            "NAME": ['cheval',
                     'chat',
                     'chien',
                     'poule'],
            "UPDATE_AT": ["2022-01-01",
                          "2022-01-01",
                          "2022-01-01",
                          "2022-01-01"]}

    return create_spark_df_from_data(spark, data)


def mock_keywords_embeddings_table(spark):
    data = {"ID": [1,
                   2],
            "EMBEDDING_BYTES": [np.arange(768).dumps(),
                                np.arange(768).dumps()],
            "UPDATE_AT": ["2022-01-01",
                          "2022-01-01"]}

    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    table_name = table_name.lower()

    if table_name == "ml.keywords_embeddings":
        return mock_keywords_embeddings_table(spark)
    elif table_name == "ml.ref_keywords":
        return mock_ref_keyword_table(spark)

    else:
        raise Exception(f"table {table_name} is not mocked")
