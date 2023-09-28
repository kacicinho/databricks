from functools import partial
from pyspark.sql import SparkSession
import pandas as pd
import pyspark.sql.functions as F
from tests.unit.utils.mocks_user_reco_page import fact_watch_sdf_mock, procesed_pdf, \
    procesed_sdf, available_programs_sdf_mock, \
    similarity_table_sdf_mock
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType
from databricks_jobs.jobs.utils.user_item_reco_page import get_available_prog_similarity_table, \
    get_user_history, add_similar_programs, transpose_and_flatten_recos_table, collect_program_reco_list, \
    collect_user_reco_list_as_pdf, get_short_term_recos, limit_recos, \
    add_prog_rank, \
    collect_user_recos_table

spark = SparkSession.builder.getOrCreate()


def test_get_available_prog_similarity_table() -> None:
    available_prog_similarity_table_sdf = get_available_prog_similarity_table(similarity_table_sdf_mock,
                                                                              available_programs_sdf_mock,
                                                                              limit_sim_prog=50)
    all_recos_programs = set(
        available_prog_similarity_table_sdf.select("PROGRAM_ID_J").rdd.flatMap(lambda x: x).collect())
    available_programs = set(available_programs_sdf_mock.select("PROGRAM_ID").rdd.flatMap(lambda x: x).collect())
    assert all_recos_programs.issubset(available_programs)


def test_get_user_history():
    user_history = get_user_history(fact_watch_sdf_mock, limit_date_rank=10)
    # fact_watch_sdf_mock.show()
    date_rank_sizes = user_history \
        .groupBy('USER_ID').agg(F.collect_list('date_rank').alias('collected')) \
        .withColumn('length', F.size('collected')) \
        .select("length").rdd.flatMap(lambda x: x).collect()
    # print(date_rank_sizes)
    assert date_rank_sizes == [5, 5]


def test_add_similar_programs():
    user_history_sdf = spark.createDataFrame(pd.DataFrame([
        {'USER_ID': 1, 'PROGRAM_ID': 2, 'DATE': 0},
        {'USER_ID': 1, 'PROGRAM_ID': 10, 'DATE': 1},
    ]))

    similarity_table_sdf_mock = spark.createDataFrame(pd.DataFrame([
        {'PROGRAM_ID_I': 2, 'PROGRAM_ID_J': 3, 'SIMILARITY_SCORE': 0.8},
        {'PROGRAM_ID_I': 2, 'PROGRAM_ID_J': 4, 'SIMILARITY_SCORE': 0.8},
        {'PROGRAM_ID_I': 10, 'PROGRAM_ID_J': 4, 'SIMILARITY_SCORE': 0.8},
        {'PROGRAM_ID_I': 7, 'PROGRAM_ID_J': 15, 'SIMILARITY_SCORE': 0.8},
        {'PROGRAM_ID_I': 6, 'PROGRAM_ID_J': 17, 'SIMILARITY_SCORE': 0.8},
    ]))

    user_history_with_recos = add_similar_programs(user_history_sdf, similarity_table_sdf_mock)
    # user_history_with_recos.show()
    recos = user_history_with_recos.select("PROGRAM_ID_J").rdd.flatMap(lambda x: x).collect()
    assert recos == [3, 4, 4]


def test_limit_recos():
    # print('\n')
    # print('similarity_table_sdf_mock')
    # similarity_table_sdf_mock.show()

    # print('available_programs_sdf_mock')
    # available_programs_sdf_mock.show(100)

    available_similar_programs = get_available_prog_similarity_table(
        similarity_table_sdf_mock, available_programs_sdf_mock, limit_sim_prog=50)
    # print('available_similar_programs')
    # available_similar_programs.show(100)

    users_history = get_user_history(fact_watch_sdf_mock, limit_date_rank=10)
    # print('users_history')
    # users_history.show()

    users_history_with_recos = add_similar_programs(users_history, available_similar_programs)
    # print('users_history_with_recos')
    # users_history_with_recos.orderBy(F.col('USER_ID'), F.col('DATE_RANK')).show()

    # print('users_history_with_recos_with_prog_rank')
    users_history_with_recos_with_prog_rank = add_prog_rank(users_history_with_recos)
    # users_history_with_recos_with_prog_rank.orderBy(F.col('USER_ID'), F.col('DATE_RANK'), F.col('SIM_RANK')).show()

    # print('limited_recos')
    limited_recos = limit_recos(users_history_with_recos_with_prog_rank, nb_prog_to_make_recos_on=3,
                                nb_reco_per_prog=4).orderBy(F.col('USER_ID'), F.col('DATE_RANK'), F.col('SIM_RANK'))
    # limited_recos.show()
    user_1_recos = limited_recos.where(F.col('USER_ID') == 1).select("PROGRAM_ID_J").rdd.flatMap(lambda x: x).collect()
    # print(user_1_recos)
    assert user_1_recos == [8, 7, 22, 16]


def test_transpose_and_flatten_recos_table():
    res = transpose_and_flatten_recos_table([[1, 10, 100], [2, 20, 200], [3, 30], [4, 40, 400, 4000]])
    # print(res)
    assert res == [1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 400, 4000]


def test_collect_program_reco_list():
    df = procesed_pdf[(procesed_pdf['USER_ID'] == 17110149) & (procesed_pdf['PROGRAM_ID_I'] == 32163)]

    # print('\n', df)
    user_0_collected = collect_program_reco_list(df,
                                                 col_to_collect='PROGRAM_ID_J', rank_col='FINAL_SIM_RANK')
    # print(user_0_collected)
    assert set(user_0_collected) == {19973, 329451, 11582, 286980, 28471, 296851, 39738, 333422, 210060, 56736}


def test_collect_user_recos_table_complex_mock():
    df = procesed_pdf[procesed_pdf['USER_ID'] == 17110149]
    collected = collect_user_recos_table(df)
    # reminder : collected is a list of list of recos. each sub-list comes from a certain program_id, maped to a certain 'FINAL_DATE_RANK'
    nb_reco_origin = len(df['FINAL_DATE_RANK'].unique())
    assert (nb_reco_origin == len(collected))


def test_collect_user_recos_table_simple_mock():
    df = pd.DataFrame([
        {'USER_ID': 1, 'FINAL_DATE_RANK': 2, 'PROGRAM_ID_I': 12, 'PROGRAM_ID_J': 4, 'FINAL_SIM_RANK': 2},
        {'USER_ID': 1, 'FINAL_DATE_RANK': 1, 'PROGRAM_ID_I': 11, 'PROGRAM_ID_J': 2, 'FINAL_SIM_RANK': 2},
        {'USER_ID': 1, 'FINAL_DATE_RANK': 1, 'PROGRAM_ID_I': 11, 'PROGRAM_ID_J': 1, 'FINAL_SIM_RANK': 1},
        {'USER_ID': 1, 'FINAL_DATE_RANK': 2, 'PROGRAM_ID_I': 12, 'PROGRAM_ID_J': 3, 'FINAL_SIM_RANK': 1},
    ])
    collected = collect_user_recos_table(df)
    assert collected == [[1, 2], [3, 4]]


def test_collect_user_reco_list_as_pdf():
    reco_length = 10
    order_and_agg_recos_with_limit = partial(collect_user_reco_list_as_pdf, reco_length=reco_length)
    order_and_collect_recos_output_schema = StructType([
        StructField("USER_ID", IntegerType(), True),
        StructField("RECOS", ArrayType(IntegerType()), True)
    ])
    res = procesed_sdf \
        .groupby('USER_ID') \
        .applyInPandas(order_and_agg_recos_with_limit, order_and_collect_recos_output_schema)
    # res.show(truncate=False)
    user_0_recos = res.collect()[0].RECOS
    # print(user_0_recos)
    assert len(user_0_recos) == reco_length


def test_get_short_term_recos():
    factored_daily_recos = get_short_term_recos(procesed_sdf, reco_length=30)
    assert len(factored_daily_recos.toPandas()['USER_ID'].unique()) == 2
