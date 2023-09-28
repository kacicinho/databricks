from functools import partial
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType
import pandas as pd
import numpy as np
from random import randrange
from typing import List, Any, Tuple
from itertools import zip_longest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Window
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df
import pyspark.sql.functions as F

flat_reco_columns = ['USER_ID', 'FINAL_DATE_RANK', 'PROGRAM_ID_I', 'PROGRAM_ID_J', 'FINAL_SIM_RANK']


def get_available_prog_similarity_table(similarity_table: SparkDataFrame,
                                        available_programs: SparkDataFrame,
                                        limit_sim_prog: int) -> SparkDataFrame:
    """ Filters out unavailable programs from program recos of similarity_table (that is PROGRAM_ID_J column)
    Limit to limit_sim_prog PROGRAM_ID_J per PROGRAM_ID_I so as to limit is used to minimize later dataframe joins

    Args:
        similarity_table: a datafrme with columns 'PROGRAM_ID_I', 'PROGRAM_ID_J', 'SIMILARITY_SCORE'
        available_programs: a dataframe with column 'PROGRAM_ID
        limit_sim_prog: represent the maximum number of rows per with same PROGRAM_ID_I (best similarities are kept)

    Returns:
        A filtered version of similarity_table.
    """
    available_similar_programs = similarity_table \
        .join(available_programs, [similarity_table.PROGRAM_ID_J == available_programs.PROGRAM_ID], 'inner') \
        .select('PROGRAM_ID_I', 'PROGRAM_ID_J', 'SIMILARITY_SCORE', 'PROGRAM_J_CAT', 'PROGRAM_J_KIND') \
        .distinct() \
        .withColumn('RECO_RANK',
                    F.row_number().over(Window.partitionBy('PROGRAM_ID_I').orderBy(F.col('SIMILARITY_SCORE').desc()))) \
        .where(F.col('RECO_RANK') <= limit_sim_prog) \
        .drop('RECO_RANK')
    return available_similar_programs


def get_user_history(fact_watch: SparkDataFrame, limit_date_rank: int,
                     bookmarks: SparkDataFrame = None) -> SparkDataFrame:
    """Filters fact_watch to keep last events
    Args:
        fact_watch: a dataframe with columns 'USER_ID', 'PROGRAM_ID', 'DATE' with each row corresponding to the event
        'USER_ID watch PROGRAM_ID on DATE'
        bookmarks: a dataframe with columns 'USER_ID', 'PROGRAM_ID', 'DATE' with each row corresponding to the event
        'USER_ID bookmarked PROGRAM_ID on DATE'
        limit_date_rank: represent the number of row per USER_ID (last events are kept)

    Returns:
        a dataframe similar to fact_watch, with fewer events, and an added 'DATE_RANK' column
        (per user, there is a one to one correspondence wetween PROGRAM_ID_I and DATE_RANK)
    """
    date_rank_window = Window.partitionBy('USER_ID').orderBy(F.col('DATE').desc())
    if bookmarks is not None:
        user_history = fact_watch.union(bookmarks)
    else:
        user_history = fact_watch

    return user_history \
        .withColumn('DATE_RANK', F.row_number().over(date_rank_window)) \
        .where(F.col('DATE_RANK') <= limit_date_rank)


def add_similar_programs(user_history: SparkDataFrame, available_similar_programs: SparkDataFrame) -> SparkDataFrame:
    """ Add similar programs with their similarity scores to user last watch programs, and corresponding similarity rank

    Args:
        user_history: a dataframe with columns 'USER_ID', 'PROGRAM_ID', 'DATE_RANK'
        available_similar_programs: a dataframe with columns 'PROGRAM_ID_I', 'PROGRAM_ID_J', 'SIMILARITY_SCORE'

    Returns:
        a dataframe with columns 'USER_ID', 'PROGRAM_ID_I', 'DATE_RANK', 'PROGRAM_ID_J, 'SIM_RANK'
    """
    similarity_window = Window.partitionBy(F.col('USER_ID'), F.col('DATE')).orderBy(F.col('SIMILARITY_SCORE').desc())
    user_history_with_recos = \
        user_history \
        .join(available_similar_programs, [user_history.PROGRAM_ID == available_similar_programs.PROGRAM_ID_I], 'inner')\
        .drop(user_history.PROGRAM_ID) \
        .alias("u") \
        .join(user_history.alias("a"),
              on=[F.col("u.USER_ID") == F.col("a.USER_ID"), F.col("u.PROGRAM_ID_J") == F.col("a.PROGRAM_ID")],
              how='anti') \
        .withColumn('SIM_RANK', F.row_number().over(similarity_window))

    return user_history_with_recos


def add_prog_rank(user_history_with_recos):
    """
    add a rank that will ensure, when filtered, that a user don't get twice the same program in his recommendations of the day :
    since PROGRAM_ID_I == p and PROGRAM_ID_I == q can both recommended PROGRAM_ID_J == r
    we only want to keep a single occurrence of PROGRAM_ID_J == r

    Args:
        user_history_with_recos: a dataframe with columns 'USER_ID', 'PROGRAM_ID_I', 'DATE_RANK', 'PROGRAM_ID_J, 'SIM_RANK'
    Returns:
        same thing with added 'PROG_RANK' column
    """
    prog_rank_window = Window.partitionBy(F.col('USER_ID'), F.col('PROGRAM_ID_J')) \
        .orderBy(F.col('DATE_RANK'), F.col('SIM_RANK'))
    res = user_history_with_recos \
        .withColumn('PROG_RANK', F.row_number().over(prog_rank_window))
    return res


def limit_recos(user_history_with_prog_rank: SparkDataFrame, nb_prog_to_make_recos_on: int,
                nb_reco_per_prog: int) -> SparkDataFrame:
    """Limit the size of user_history_with_prog_rank to use only the last seen programs per user to make some recommendations
    make sure not to keep multiplke occurences of same program in recomendations, thank to PROG_RANK
    Since some filtering has previously been done, DATE_RANK and SIM_RANK are not contiguous,
    so FINAL_DATE_RANK and FINAL_SIM_RANK are created before limiting
    In the end, per user, there is a one to one correspondence between PROGRAM_ID_I and FINAL_DATE_RANK

    Args:
        user_history_with_prog_rank a dataframe with columns 'USER_ID', 'PROGRAM_ID_I', 'DATE_RANK', 'PROGRAM_ID_J, 'SIM_RANK', 'PROG_RANK'
        nb_prog_to_make_recos_on: maximum of program per user to use as source of recommendations
        nb_reco_per_prog: number of reco per user per source program

    Returns:
        a filtered version of user_history_with_prog_rank with added columns FINAL_DATE_RANK and FINAL_SIM_RANK
    """
    final_date_rank_window = Window.partitionBy(F.col('USER_ID')).orderBy(F.col('DATE_RANK'))
    final_sim_rank_window = Window.partitionBy(F.col('USER_ID'), F.col('DATE_RANK')).orderBy(F.col('SIM_RANK'))
    # .withColumn('sim_rank', F.row_number().over(similarity_window))
    return user_history_with_prog_rank \
        .where(F.col('PROG_RANK') == 1) \
        .withColumn('FINAL_DATE_RANK', F.dense_rank().over(final_date_rank_window)) \
        .withColumn('FINAL_SIM_RANK', F.dense_rank().over(final_sim_rank_window)) \
        .where(F.col('FINAL_DATE_RANK') <= nb_prog_to_make_recos_on) \
        .where(F.col('FINAL_SIM_RANK') <= nb_reco_per_prog) \
        .drop('PROG_RANK', 'DATE_RANK', 'SIM_RANK')


def transpose_and_flatten_recos_table(list_of_lists: List[List[Any]]) -> List[Any]:
    """Since the collected program ids for recommendations are of this form :
    [[a, aa, aaa],
     [b, bb, bbb],
     [c, cc, ccc]]

    (supposing that for a given user,
        - his program_1 leads to recos [a, aa, aaa]
        - his program_2 leads to recos [b, bb, bbb]
        - his program_3 leads to recos [c, cc, ccc])
    We need to transpose before flattening to obtain the list [a, b, c, aa, bb, cc, aaa, bbb, ccc] of recos
    The sublists may have different lengths, missing values will be filled with None, and then filtered out
    """
    #
    transposed_w_none = list(map(list, zip_longest(*list_of_lists, fillvalue=None)))
    return [x for sub_list in transposed_w_none for x in sub_list if x is not None]


def weighted_shuffle(ranks: List[int], alpha: float = 3) -> List[int]:
    """
    shuffle ordered rank list keeping low ranks in low position and high ranks in high positions (assuming alpha >= 1)
    ex list [0, 1, 2, 3, 4, 5, 6] could be shuffled to [1, 0, 3, 2, 4, 5, 6]
    but probably not to [5, 6, 3, 1, 2, 4]
    with a high alpha, original order is almost not changed

    Args:
        ranks: not necessarily order nor contiguous ranks
        alpha: order_factor. with a high alpha, small ranks are located in small indexes, and high ranks in high indexes
    Returns:
        sorted ranks with some shuffling
    """
    sorted_ranks = sorted(ranks)
    weights = [(len(sorted_ranks) - k) ** alpha for k in range(len(sorted_ranks))]
    rand_weighted_order = [randrange(int(w) + 1) for w in weights]
    return [sorted_ranks[i] for i in np.argsort(rand_weighted_order)[::-1]]


def collect_program_reco_list(df: pd.DataFrame, col_to_collect: str = 'PROGRAM_ID_J', rank_col: str = 'FINAL_SIM_RANK',
                              shuffle: bool = False) -> List[int]:
    """Collect column 'col_to_collect' of 'df', ordering (with some little shuffling if wanted) by 'rank_col'

    Args:
        df: a dataframe with columns 'flat_reco_columns'
            (as a reminder, there is a one to one correspondence between 'PROGRAM_ID_I'
        col_to_collect: the column that will be collected
        rank_col: the column used to rank before collection
        shuffle: if True, a light shuffling is done before collection
    Returns:
        the list of recommended items of column col_to_collect for unique 'PROGRAM_ID_I' value
    """
    df = df.set_index(rank_col)
    if shuffle:
        ranks = list(df.index)
        shuffled_ranks = weighted_shuffle(ranks)
        reco = df.loc[shuffled_ranks][col_to_collect].tolist()
    else:
        df = df.sort_index()
        reco = df[col_to_collect].tolist()
    return reco


def collect_user_recos_table(df: pd.DataFrame, group_col: str = 'FINAL_DATE_RANK') -> List[List[int]]:
    """Supposing that for a given user,
        - his program_1 leads to recos [a, aa, aaa]
        - his program_2 leads to recos [b, bb, bbb]
        - his program_3 leads to recos [c, cc, ccc]
    the returned value will be :
    [[a, aa, aaa],
     [b, bb, bbb],
     [c, cc, ccc]]

    (as a reminder, there is a one to one correspondence between 'PROGRAN_ID_I'
            and 'FINAL_DATE_RANK' columns per user)
    Args:
        df: dataframe with columns 'flat_reco_columns'
        group_col: the column to group on
    Returns:
        A list of list of recos per progrom_id_i

    NB : default behaviour of pandas.groupby is sort=True, explicit passing is meant to better readability
    """

    return df.groupby(group_col, sort=True).apply(collect_program_reco_list).tolist()


def collect_user_reco_list_as_pdf(df: pd.DataFrame, reco_length: int) -> pd.DataFrame:
    """Collect all recommendations for unique 'USER_ID' value of df

    Args:
        df: dataframe with columns 'flat_reco_columns' (with only one 'USER_ID')
        reco_length: max size of reco list
    Returns:
        user_recos flattended, encapsulated in a one line pandas dataframe (this format is needed to use a grouped map UDF)
    """
    assert len(set(df['USER_ID'])) == 1, "The column user_col must contain same value on all rows"
    user_id = df['USER_ID'][0]
    recos_table = collect_user_recos_table(df)
    recos = transpose_and_flatten_recos_table(recos_table)[:reco_length]
    return pd.DataFrame([{'USER_ID': user_id, 'RECOS': recos}])


def get_short_term_recos(limited_recos: SparkDataFrame, reco_length: int) -> SparkDataFrame:
    """Compute all daily recommendations for all users based on user watch history and program similarities previously computed

    Args:
        limited_recos: dataframe with columns 'flat_reco_columns'
        reco_length: the (max) size of reco list per user
    Returns:
        a dataframe with on line per user, with 'USER_ID' and 'RECOS' columns
    """

    order_and_collect_recos_output_schema = StructType([
        StructField("USER_ID", IntegerType(), True),
        StructField("RECOS", ArrayType(IntegerType()), True)
    ])

    collect_recos = partial(collect_user_reco_list_as_pdf, reco_length=reco_length)
    return limited_recos.groupby('USER_ID').applyInPandas(collect_recos, order_and_collect_recos_output_schema)


def get_similarities(spark, options, similarity_table, unwanted_categories: Tuple[int, ...],
                     unwanted_kinds: Tuple[int, ...]) -> SparkDataFrame:
    query = f"""
        with 
            prog_to_use_in_recos as
            (
            select * from backend.program
                where 
                    1=1
                    and REF_PROGRAM_CATEGORY_ID not in {unwanted_categories}
                    and REF_PROGRAM_KIND_ID not in {unwanted_kinds} 
            )
        select 
            sim.*,
            REF_PROGRAM_CATEGORY_ID as PROGRAM_J_CAT,
            REF_PROGRAM_KIND_ID as PROGRAM_J_KIND
        from    
            {similarity_table} as sim
        inner join
            prog_to_use_in_recos on prog_to_use_in_recos.ID = sim.PROGRAM_ID_J
        where 
            1=1
            and PROGRAM_ID_I in (select ID from prog_to_use_in_recos) 
        """
    return load_snowflake_query_df(spark, options, query)


def get_fact_watch(spark, options, days_look_back: int, unwanted_categories: Tuple[int, ...],
                   unwanted_kinds: Tuple[int, ...]) -> SparkDataFrame:
    query = f"""
    select 
        distinct
        fwagg.USER_ID,
        fwagg.EPISODE_ID,
        fwagg.PROGRAM_ID,
        fwagg.WATCH_FRAC_DURATION,
        fwagg.DATE_RECEIVED_AT,
        prog.REF_PROGRAM_CATEGORY_ID as PROGRAM_I_CAT,
        prog.REF_PROGRAM_KIND_ID as PROGRAM_I_KIND        
    from 
        DW.FACT_WATCH_7D_AGG as fwagg
    join 
        backend.program as prog on prog.ID = fwagg.PROGRAM_ID 
    where  
        prog.REF_PROGRAM_CATEGORY_ID not in {unwanted_categories} 
        and prog.REF_PROGRAM_KIND_ID not in {unwanted_kinds} 
        and fwagg.DATE_RECEIVED_AT >= current_date - {days_look_back}
    """
    res = load_snowflake_query_df(spark, options, query) \
        .where(F.col('WATCH_FRAC_DURATION') > 0.8) \
        .withColumnRenamed('DATE_RECEIVED_AT', 'DATE') \
        .select('USER_ID', 'PROGRAM_ID', 'DATE', 'PROGRAM_I_CAT', 'PROGRAM_I_KIND')
    return res
