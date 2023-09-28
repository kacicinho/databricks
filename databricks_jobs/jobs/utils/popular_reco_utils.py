from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql.types import IntegerType, StringType, ArrayType, StructField, StructType, FloatType
from pyspark.sql.window import Window

from databricks_jobs.jobs.utils.spark_utils import typed_udf, create_empty_df


def load_snowflake_table(ss, options, table_name):
    return ss.read \
        .format("snowflake") \
        .options(**options) \
        .option("dbtable", table_name) \
        .load()


def load_snowflake_query_df(ss, options, query):
    return ss.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", query) \
        .load()


def write_df_to_snowflake(df, write_options, table_name, mode="append"):
    df.write \
        .format("snowflake") \
        .options(**write_options) \
        .option("dbtable", table_name) \
        .mode(mode) \
        .save()


def unpivot_fact_audience(df, categories, other_cols):
    """
    Transform the affinity cols into new rows for each 1
    """
    all_dfs = list()
    for cat in categories:
        all_dfs.append(
            df.where(F.col(cat) > 0).select(*other_cols, F.col(cat).alias("score"), F.lit(cat).alias("category"))
        )

    new_df = all_dfs[0]
    for sub_df in all_dfs[1:]:
        new_df = new_df.unionAll(sub_df)
    return new_df. \
        withColumn("category", F.regexp_replace(F.col("category"), '"', ""))


def build_affinity_cooc(affinity_df):
    """
    Idea :
        User_1 [Affinity_1, Affinity_2]
        User_2 [Affinity_1, Affinity_3]
    We can deduce Affinity_1 may entail affinity Affinity_2 or Affinity_3 with 50% chance (called ratio)

    :param affinity_df:
    :return:
    """
    return affinity_df. \
        select("USER_ID", "category"). \
        withColumn("affinity_presence", F.count("USER_ID").over(Window.partitionBy("category"))). \
        alias("left"). \
        join(affinity_df.alias("right"), on="USER_ID"). \
        where(F.expr("left.category != right.category")). \
        groupBy("left.category", "right.category"). \
        agg(F.count("left.USER_ID").alias("cooc"),
            F.max("left.affinity_presence").alias("affinity_presence")). \
        selectExpr("left.category as affinity_1", "right.category as affinity_2", "cooc / affinity_presence as ratio")


struct_schema = StructType([
    StructField('PROGRAM_ID', IntegerType(), nullable=False),
    StructField('AFFINITY', StringType(), nullable=False),
    StructField('CHANNEL_ID', IntegerType(), nullable=False),
    StructField('EPISODE_ID', IntegerType(), nullable=False),
    StructField('reco_origin', StringType(), nullable=False),
    StructField('ranking', FloatType(), nullable=False),
    StructField('rating', FloatType(), nullable=False)
])


def filter_row_based_on_prog_id(rows, to_avoid):
    return filter(lambda x: x.PROGRAM_ID not in to_avoid, rows)


@typed_udf(ArrayType(struct_schema))
def merge_udf(reco_array_1, reco_array_2, already_seen, already_booked, n):
    already_booked = already_booked if already_booked else []
    reco_array_1 = list(filter_row_based_on_prog_id(reco_array_1, list(already_seen) + list(already_booked)))
    selected = len(reco_array_1)
    reco_array_2 = filter_row_based_on_prog_id(reco_array_2, list(already_seen) + list(already_booked))
    reco_array_3 = sorted(filter_row_based_on_prog_id(reco_array_2, {x.PROGRAM_ID for x in reco_array_1}),
                          key=lambda x: x.ranking)
    return reco_array_1 + reco_array_3[:n - selected]


def format_for_reco_output(df, alias_name, field_names=("PROGRAM_ID", "AFFINITY", "CHANNEL_ID", "EPISODE_ID",
                                                        "reco_origin", "ranking", "rating")):
    return df.groupBy("USER_ID"). \
        agg(F.collect_list(F.struct(*field_names)).alias(alias_name))


def complete_recos(selected_reco_df, global_reco_df, user_watch_history, user_bookmarks, user_id_to_hash, nb_recos):
    """
    Steps followed :
    - Group by USER_ID
    - join user_id -> bundle_hash, hash -> default_reco
    - Select default reco based on bundle_hash
    - Combine reco arrays

    :param selected_reco_df: personalised recos for all users with some history
    :param global_reco_df: used to complement recos from selected_reco_df
    :param user_watch_history: used to remove suggestions already seen
    :param user_bookmarks: used to remove suggestions already booked
    :param user_id_to_hash: used for the hash matching based default recos
    :param nb_recos: nb recos to seelct from selected_reco_df + global_reco_df - user_watch_history
    :return:
    """
    # Prepare global reco format
    global_reco_df = format_for_reco_output(global_reco_df, "generic_recos"). \
        alias("global")
    # Prepare user watch history and user bookmarks
    user_watch_history_grouped = user_watch_history. \
        groupBy("USER_ID"). \
        agg(F.collect_set("PROGRAM_ID").alias("already_seen"))

    user_bookmarks_grouped = user_bookmarks. \
        groupBy("USER_ID"). \
        agg(F.collect_set("PROGRAM_ID").alias("already_booked"))

    # Join addition info to user_reco
    user_recos = format_for_reco_output(selected_reco_df, "recos"). \
        withColumn("to_pick", F.lit(nb_recos)).\
        join(F.broadcast(user_id_to_hash.alias("hash")), user_id_to_hash.USER_ID == selected_reco_df.USER_ID, "left"). \
        select("default.USER_ID", "HASH_ID", "recos", "to_pick")
    # Default reco will be with user_id = 0
    user_recos = user_recos. \
        fillna(0, subset=["HASH_ID"])

    # Join default reco per user tvbundle and complete reco with UDF
    result = user_recos.alias("origin"). \
        join(user_watch_history_grouped, user_watch_history_grouped.USER_ID == user_recos.USER_ID). \
        join(user_bookmarks_grouped, user_bookmarks_grouped.USER_ID == user_recos.USER_ID, "left"). \
        join(F.broadcast(global_reco_df), F.col("origin.HASH_ID") == F.col("global.USER_ID")). \
        withColumn("recommendations", merge_udf("recos", "generic_recos", "already_seen", "already_booked",
                                                "to_pick")). \
        select("origin.USER_ID", "recommendations")

    return result


def certainty_fn(col_name, lambda_hours):
    """
    This function can be used to decrease the effect of niche programs when using an avg of watch
    """
    return F.lit(1) - F.exp(-F.sum(col_name) / F.lit(lambda_hours * 3600))


def keep_top_X_programs(future_program_with_infos, use_affinity=True,
                        scoring_methods=("total_distinct_bookmarks", "nb_likes",
                                         "external_rating", "total_celeb_points"),
                        top_k=200):
    """
    In this step, we collect the top X best programs according to the different kind of popularity.

    1 - Get rank of prog per affinity for each popularity
    2 - Filter progs to keep only the top k

    :param future_program_with_infos:
    :param use_affinity: Allow to do global top as well as local one
    :param scoring_methods: allows to reduce the pop_factors used
    :param top_k: Nb of programs to keep, note that higher nb are needed when use_affinity=False
    :return:
    """

    scoring_methods_top_names = ["top_{}".format(col_name) for col_name in scoring_methods]

    def extract_top_programs(prog_df, cols_to_top_over):
        """
        We do the following :

        prog_df gets a new column : top_{metric_name} in **scoring_methods**
        the top is computed either over an affinity or the whole set given **use_affinity**
        if metric_name == 0, we force the rank to be top_k, to be sure the program is not selected because of chance.
        """

        def window_fn(column):
            return Window.partitionBy("AFFINITY").orderBy(F.desc(column)) \
                if use_affinity else Window.orderBy(F.desc(column))

        for col_name in cols_to_top_over:
            rank_over_affinity_window = F.row_number(). \
                over(window_fn(col_name))
            prog_df = prog_df. \
                withColumn("top_{}".format(col_name),
                           F.when(F.col(col_name) > 0, rank_over_affinity_window).otherwise(top_k))
        return prog_df

    # 1 - Get rank of prog per affinity for each popularity
    top_prog_df = extract_top_programs(future_program_with_infos, scoring_methods)

    # 2 - Filter progs to keep only the top k
    # 2.1 - Create the filter expression with column names like
    #       "is_rank<k_over_pop1 or is_rank<k_over_pop2 or ..."
    filter_over_scoring_methods_string = " or ".join(
        ["{} <= {}".format(col_name, top_k)
         for col_name in scoring_methods_top_names])
    top_k_prog_df = top_prog_df. \
        where(filter_over_scoring_methods_string)

    # 2.2 - We retrieve the popularity type giving the highest rank to the program
    reco_origin_expression_string = " ".join(
        ["CASE WHEN ",
         " WHEN ".join([f"best_rank = {scoring_methods_top_name} THEN '{scoring_method}'"
                        for scoring_methods_top_name, scoring_method in
                        zip(scoring_methods_top_names, scoring_methods)]),
         "ELSE 'UNKNOWN' END as reco_origin"])
    future_program_with_top_infos = top_k_prog_df. \
        withColumn("best_rank", F.least(*scoring_methods_top_names)). \
        withColumn("reco_origin", F.expr(reco_origin_expression_string))

    return future_program_with_top_infos


def build_empty_user_csa_pref(spark):
    schema = T.StructType([
        StructField('USER_ID', T.IntegerType(), True),
        StructField('max_csa_id', T.IntegerType(), True)
    ])
    return create_empty_df(spark, schema)
