import pandas as pd
import datetime
from datetime import timedelta


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_backend_program(spark):
    n_progs = 6
    pids = list(range(n_progs))
    data = {
        "ID": pids,
        "REF_PROGRAM_CATEGORY_ID": [1, 2, 8, 8, 8, 8]
    }
    return create_spark_df_from_data(spark, data)


def mock_backend_broadcast(spark):
    now = datetime.datetime.now()
    data = {
        "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "ASSET_MANIFEST_ID": [1, 2, 3, 4] + [i for i in range(5, 15)],
        "PROGRAM_ID": [0, 1, 3, 5] + [2 for i in range(5, 15)],
        "EPISODE_ID": [1, 2, 3, 4] + [5 for i in range(5, 15)]
    }
    return create_spark_df_from_data(spark, data)


def mock_episode_table(spark):
    id_range = list(range(6))
    data = {
        "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "PROGRAM_ID": [0, 1, 3, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4]
    }
    return create_spark_df_from_data(spark, data)


def mock_backend_vod(spark):
    data = {
        "ASSET_MANIFEST_ID": [15],
        "EPISODE_ID": [15]
    }
    return create_spark_df_from_data(spark, data)


def mock_asset_manifest(spark):
    now = datetime.datetime.now()
    basename = ["b015b3b144471b49fc7db4757da85f38bb30a21d", "9a8dc6e699161b9a63e08c02a9f347376bddcda5",
                "32166c674dd0aff3fe802aa5480bc1a919f4af09", "12bbc0906c831e8ed064ec3b552469cd893975e9"]
    data = {
        "ID": [*[1, 2, 3, 4], *[i for i in range(5, 15)], *[15]],
        "S3": [*[1, 0, 1, 1], *[1 for i in range(5, 15)], *[1]],
        "VERSION": [*[2, 2, 2, 2], *[2 for i in range(5, 15)], *[2]],
        "BASENAME": [*basename, *["6c4c7734d45f1ee0772a7eeb9eb7677bc0c3059e{}".format(i) for i in range(5, 15)], *[
            "3f88cb97085d67add75e7da27766d5ccb45578ee"]],
        "BROADCAST_ID": [*[1, 2, 3, 4], *[i for i in range(5, 15)], *[None]],
        "REPLAY_ID": [*[None, 1, None, None], *[None for i in range(5, 15)], *[2]],
        "CREATED_AT": [*[now, now, now, now], *[now - timedelta(minutes=i) for i in range(5, 15)], *[now]],
        "DELETED_AT": [*[now, None, None, None], *[None for i in range(5, 15)], *[None]]
    }
    return create_spark_df_from_data(spark, data)


def mock_asset_audio(spark):
    basename = ["37da2e18ec263b9f7eb6c35b85e8488333046a23", "aec62a39f546c86e70086c293733d01df770ec7e",
                "6385a3b307dbd0a84443978f9d190306ecfefcae", "aaeee451bc105c1a67e9e64f341a3fa95257451a"]
    data = {
        "ASSET_MANIFEST_ID": [*[1, 2, 3, 4], *[i for i in range(5, 15)], *[15]],
        "BASENAME": [*basename, *["b43d3a99ca11eb2ef9b654b69f8d5564cb4df2e3{}".format(i) for i in range(5, 15)], *[
            "99f0244505b2cbf2ec2021f05ad3bbf8142a67f5"]],
        "LANGUAGE": [*["fre", "fre", "en", "fre"], *["fre" for i in range(5, 15)], *["fre"]],
        "EXTENSION": [*["isma", "isma", "isma", "mp4"], *["isma" for i in range(5, 15)], *["isma"]],
        "DURATION": [1000 for i in range(15)]
    }
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    table_name = table_name.lower()
    if table_name == "backend.program":
        return mock_backend_program(spark)
    elif table_name == "backend.episode":
        return mock_episode_table(spark)
    elif table_name == "backend.broadcast":
        return mock_backend_broadcast(spark)
    elif table_name == "backend.vod":
        return mock_backend_vod(spark)
    elif table_name == "backend.asset_manifest":
        return mock_asset_manifest(spark)
    elif table_name == "backend.asset_audio":
        return mock_asset_audio(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
