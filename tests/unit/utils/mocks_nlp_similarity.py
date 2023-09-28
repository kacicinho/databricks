import pandas as pd


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_synopsys(spark):
    sharknado_1 = {
        'PROGRAM_ID': 10,
        'SYNOPSYS': 'Shark are awesome, but also very dangerous. Especially two headed sharks'
    }
    sharknado_2 = {
        'PROGRAM_ID': 20,
        'SYNOPSYS': 'Shark are amazing, but also very terrifying. Especially zombie sharks'
    }
    sport_movie_1 = {
        'PROGRAM_ID': 30,
        'SYNOPSYS': 'This is the story of the birth of soccer, one of the most popular sport ever.'
    }
    sport_movie_2 = {
        'PROGRAM_ID': 40,
        'SYNOPSYS': 'Alex loves rugby, and want to become the best rugby player of all times'
    }
    film_list = [sharknado_1, sharknado_2, sport_movie_1, sport_movie_2]
    films_dict = {
        'PROGRAM_ID': [],
        'SYNOPSYS': []
    }

    for d in film_list:
        for key, value in d.items():
            films_dict[key].append(value)

    df = pd.DataFrame(film_list)
    return create_spark_df_from_data(spark, df)
