import pandas as pd
import datetime


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_backend_program(spark):
    n_progs = 6
    pids = list(range(n_progs))
    data = {"ID": pids,
            "TITLE": ["prog_{}".format(i) for i in range(n_progs)],
            "REF_PROGRAM_CATEGORY_ID": [1 for _ in range(n_progs)],
            "REF_PROGRAM_KIND_ID": [150, 150, 1, 1, 1, 2],
            "PRODUCTION_YEAR": [1990 for _ in range(n_progs)],
            "SUMMARY": ["Le feu fait ravage", "Une tour en feu", "Amour et rire", "Comédie décalé, fou rire garanti",
                        "Météo",
                        "Mickael Jordan, son histoire"],
            "DURATION": [30 * 60 for _ in range(n_progs)]}
    # Program 4 will be filtered by program kind filter
    return create_spark_df_from_data(spark, data)


def mock_dw_dim_channel_group(spark):
    data = {
        "CHANNEL_ID": [4, 4, 3, 1, 1],
        "CHANNEL_GROUP_ID": [2, 2, 2, 1, 1]}
    return create_spark_df_from_data(spark, data)


def mock_backend_broadcast(spark):
    now = datetime.datetime.now()
    data = {
        "PROGRAM_ID": [0, 1, 1, 0, 2],
        "EPISODE_ID": [1, 2, 3, 4, 5],
        "START_AT": [now, now, now, now, now]}
    return create_spark_df_from_data(spark, data)


def mock_backend_edito_program(spark):
    n_progs = 6
    pids = list(range(n_progs))
    data = {"PROGRAM_ID": pids,
            "TITLE": ["prog_{}".format(i) for i in range(n_progs)],
            "REF_PROGRAM_CATEGORY_ID": [1 for _ in range(n_progs)],
            "REF_PROGRAM_KIND_ID": [1, 1, 1, 1, 52, 2],
            "SUMMARY": ["Le feu fait ravage", "Une tour en feu", "Amour et rire", "Comédie décalé, fou rire garanti",
                        "Météo",
                        "Mickael Jordan, son histoire"]}
    return create_spark_df_from_data(spark, data)


def mock_backend_ref_program_category(spark):
    data = {"ID": [1],
            "NAME": ["Films"]}
    return create_spark_df_from_data(spark, data)


def mock_backend_ref_program_kind(spark):
    data = {"ID": [1, 2, 52, 150],
            "NAME": ["Comédie", "Basket-ball", "Météo", "Catastrophe"]}
    return create_spark_df_from_data(spark, data)


def mock_episode_table(spark):
    id_range = list(range(8))
    data = {
        "ID": id_range,
        "NUMBER": id_range,
        "SEASON_ID": id_range,
        "DURATION": [16 * 60 for _ in id_range],
        "TITLE_ORIGINAL": [f"PROGR_0_ep_{i}" for i in id_range],
        "PROGRAM_ID": [0, 0, 1, 1, 0, 2, 3, 4]
    }
    return create_spark_df_from_data(spark, data)


def mock_rel_episode_tag_table(spark):
    pids = list(range(6))
    data = {
        "EPISODE_ID": pids,
        "REF_TAG_ID": [1, 0, 1, 2, 0, 0]
    }
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    table_name = table_name.lower()
    if table_name == "backend.program":
        return mock_backend_program(spark)
    elif table_name == "backend.episode":
        return mock_episode_table(spark)
    elif table_name == "backend.rel_episode_tag":
        return mock_rel_episode_tag_table(spark)
    elif table_name == "backend.ref_program_category":
        return mock_backend_ref_program_category(spark)
    elif table_name == "backend.ref_program_kind":
        return mock_backend_ref_program_kind(spark)
    elif table_name == "backend.edito_program":
        return mock_backend_edito_program(spark)
    elif table_name == "backend.broadcast":
        return mock_backend_broadcast(spark)
    elif table_name == "dw.dim_channel_group":
        return mock_dw_dim_channel_group(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
