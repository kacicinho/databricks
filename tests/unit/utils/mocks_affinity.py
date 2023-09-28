import datetime

from tests.unit.utils.mocks import create_spark_df_from_data

now = datetime.datetime.now()


def create_audience_prog_type_affinity_df_mock(spark):
    user_ids = [1, 4, 2, 0, 3]
    data = {"user_id": user_ids,
            "movie_duration": [3000 for _ in user_ids],
            "serie_duration": [3000 for _ in user_ids],
            "documentary_duration": [3000 for _ in user_ids],
            "broadcast_duration": [3000 for _ in user_ids],
            "movie_ntile": [0.1, 0.4, 0.3, 0.9, 0.8],
            "serie_ntile": [0.2, 0.1, 0.7, 0.5, 0.4],
            "documentary_ntile": [0.9, 0.9, 0.2, 0.1, 0.4],
            "broadcast_ntile": [0.7, 0.9, 0.7, 0.5, 0.1]
            }
    return create_spark_df_from_data(spark, data)


def create_program_df_mock(spark):
    data = {"PROGRAM_ID": [0, 1, 2, 3],
            "PROGRAM": ["Rock academy", "Le 14h-17h", "On Freddie Roach", "test"],
            "CATEGORY": ["Enfants", "Informations", "Séries", "test"],
            "KIND": ["+ de 10 ans - Séries", "Infos et journaux télévisés", "Téléréalité", "Religion"],
            "SUMMARY": ["", "", "", ""]}
    return create_spark_df_from_data(spark, data)


def create_affinities_df_mock(spark):
    pid = list(range(4))
    data = {"Segment algorithm": ["Divertissement/Séries/Téléréalité", "Banque & Finance",
                                  "Divertissement/Film/Action", "Banque/Banque/finance/finace//"],
            "Segment table": ["Téléréalité", "Banque & Finance", "Action & Aventure", "Banque & Finance"],
            "Profil": ["User" for i in pid]}
    return create_spark_df_from_data(spark, data)


def create_user_affinities_df_mock(spark):
    pid = list(range(4))
    data = {"USER_ID": pid,
            "AFFINITY": ["affinity_{}".format(i) for i in pid],
            "AFFINITY_RATE": [0.8, 0.63, 0.1, 0.5]}
    return create_spark_df_from_data(spark, data)


def create_fact_audience_aff_only_df_mock(spark):
    pid = list(range(4))
    previous_date = now.date() - datetime.timedelta(days=46)
    data = {"USER_ID": pid + pid,
            "date": [now.date(), now.date(), now.date(), now.date()] + [previous_date] * 4,
            "Action & Aventure": [1, 0, 0, 0.3, 0.5, 0, 0.2, 0],
            "Cuisine": [0, 0, 0.9, 0, 0.5, 0.4, 1, 0],
            "Banque & Finance": [0, 0.2, 0, 0.1, 0, 0, 0, 0]}
    return create_spark_df_from_data(spark, data)


def create_fact_audience_prog_type_affinity_df_mock(spark):
    pid = list(range(4))
    data = {"USER_ID": pid,
            "DATE_DAY": [now.date(), now.date(), now.date(), now.date()],
            "MOVIE": [1, 0, 0, 1],
            "SERIE": [1, 1, 0, 0],
            "BROADCAST": [1, 1, 0, 1],
            "DOCUMENTARY": [0, 1, 0, 1]}
    return create_spark_df_from_data(spark, data)


def build_user_mock(spark):
    bday = datetime.datetime.now() - datetime.timedelta(days=20 * 365)
    data = {'ID': [1, 3, 2, 0, 4],
            'GENDER': ['F', 'M', 'F', '', None],
            'BIRTHDAY': [bday, bday, bday, None, None],
            'FIRST_NAME': ["Jean", "Pierre", "Marie", "Gertrude", "Hans"],
            'LAST_NAME': ["Dupont", "Martin", "Beliard", "Pouet", "Zimmer"],
            'EMAIL': ["jd@gmail.com", "pm@gmail.com", "mb@gmail.com", "gp@gmail.com", "hz@gmail.com"]
            }
    return create_spark_df_from_data(spark, data)


def mock_prog_type_df(spark, dt=datetime.date.today()):
    data = {'USER_ID': [1, 3, 2, 0, 4],
            'DATE_DAY': [dt for _ in range(5)],
            'MOVIE_DURATION': [100, 0, 1234, 3456, 0],
            'SERIE_DURATION': [0, 3450, 1234, 3456, 0],
            'BROADCAST_DURATION': [0, 0, 1234, 13456, 0],
            'DOCUMENTARY_DURATION': [0, 0, 1234, 56, 0],
            'WATCH_DURATION': [1234, 345678, 23, 0, 12345]
            }
    return create_spark_df_from_data(spark, data)


def multiplex_mock(spark, options, table_name, *args):
    if table_name == "DW.FACT_AUDIENCE_AFF_ONLY":
        return create_fact_audience_aff_only_df_mock(spark)
    elif table_name == "DW.FACT_AUDIENCE_PROG_TYPE_AFFINITY":
        return create_fact_audience_prog_type_affinity_df_mock(spark)
    elif table_name == "DW.FACT_AUDIENCE_DAILY_USER_PROG_TYPE":
        return mock_prog_type_df(spark)
    elif table_name == "BACKEND.USER_RAW":
        return build_user_mock(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")
