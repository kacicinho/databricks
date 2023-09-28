import pandas as pd


program_summary = "chien et chat et cochon."
edito_summary = "rat et souris."
episode_summary = "cheval"


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_translated_summaries_table(spark):
    data = {"UPDATE_AT": ['2022-01-01',
                          '2022-01-01',
                          '2022-01-01'],
            "PROGRAM_ID": [7,
                           10,
                           11],
            "EN_SUMMARY": ["summary 2",
                           "summary 10",
                           "summary 11"]}

    return create_spark_df_from_data(spark, data)


def mock_program_table(spark):
    data = {"ID": [1,
                   1,
                   2,
                   7,
                   8,
                   6],
            "TITLE": ["TITRE_1",
                      "TITRE_1",
                      "TITRE_2",
                      "TITRE_7",
                      "TITRE_8",
                      "TITRE_6"],
            "REF_PROGRAM_CATEGORY_ID": [1,
                                        1,
                                        1,
                                        1,
                                        2,
                                        1],
            "SUMMARY": ["poule",
                        program_summary,
                        "program summary 2",
                        "texte 7",
                        "texte 8",
                        "texte 6"]}

    return create_spark_df_from_data(spark, data)


def mock_edito_table(spark):
    data = {"PROGRAM_ID": [1,
                           2,
                           3],
            "TITLE": ["TITRE_1",
                      "TITRE_2",
                      "TITRE_3"],
            "REF_PROGRAM_CATEGORY_ID": [1,
                                        1,
                                        1],
            "SUMMARY": [edito_summary,
                        "edito un peu plus long 2",
                        "edito summary 3"]}

    return create_spark_df_from_data(spark, data)


def mock_episode_table(spark):
    data = {"PROGRAM_ID": [4,
                           2,
                           3,
                           1],
            "TITLE": ["TITRE_4",
                      "TITRE_2",
                      "TITRE_3",
                      "TITRE_1"],
            "REF_PROGRAM_CATEGORY_ID": [1,
                                        1,
                                        1,
                                        1],
            "TEXT_LONG": ["episode petit 4",
                          "episode et un peu plus long 2",
                          "episode et summary 3",
                          episode_summary]}

    return create_spark_df_from_data(spark, data)


def mock_backend_ref_program_category(spark):
    data = {"ID": [1, 2],
            "NAME": ["Films", "Série"]}
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    table_name = table_name.lower()
    if table_name == "backend.program":
        return mock_program_table(spark)
    elif table_name == "backend.episode":
        return mock_episode_table(spark)
    elif table_name == "backend.ref_program_category":
        return mock_backend_ref_program_category(spark)
    elif table_name == "backend.edito_program":
        return mock_edito_table(spark)
    elif table_name == "ml.translated_summaries_fr_to_en":
        return mock_translated_summaries_table(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
