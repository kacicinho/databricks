import os
import time

import pandas as pd
from pyspark.sql.functions import regexp_replace
from pyspark.sql import functions as F
from pyspark.sql.functions import sha2, concat_ws
import snowflake.connector
from snowflake.connector import ProgrammingError


def get_snowflake_credentials(conf, db):
    """Returns snowflake credentials adapted to the databricks environnment used.
    Args:
        conf (dict): conf file with credentials
        db (str): The database to use (PROD or RAW). The function converts
        automatically the db to the appropriate env.
    Returns:
        cred (dict): Snowflake required credentials
    """
    cred = {}

    # Get current workspace
    env = conf.get("env", "DEV")

    # Get user/pw secrets
    snowflake_user = conf.get("user_name", "")
    snowflake_password = conf.get("password", "")

    cred["user"] = snowflake_user
    cred["password"] = snowflake_password

    # Dev if workspace is dev or stage
    if env == 'DEV':
        if db == "PROD":
            sf_database = "DEV"
        elif db == "RAW":
            sf_database = "RAW"
        else:
            raise Exception("db not available in dev")
        sf_role = "ANALYST_DEV"

    # Prod if workspace is prod
    elif env == 'PROD':
        if db in ('PROD', 'RAW'):
            sf_database = db
        else:
            raise Exception("db not available in prod")
        sf_role = "ANALYST_PROD"

    else:
        raise Exception("The env provided does not exist. Please try again with one of the following values: 'DEV' or 'PROD'")

    cred["database"] = sf_database
    cred["role"] = sf_role

    return cred


def get_snowflake_options(conf, db, sf_schema="", **additional_options):
    """Sets snowflake options for interacting with snowflake pyspark connector.
    Args:
        conf (dict): conf file with credentials
        db (str): The PROD database you want to use (PROD or RAW).
        sf_schema : Snowflake schema
        additional_options (dict): See Snowflake options :
        https://docs.snowflake.com/en/user-guide/spark-connector-use.html#additional-options
    Returns:
        options (dict): Snowflake options.
    """
    cred = get_snowflake_credentials(conf, db)

    options = {
        "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
        "sfSchema": sf_schema,
        "sfWarehouse": "DATABRICKS_XS_WH"
    }

    options_names = ["sfUser", "sfPassword", "sfDatabase", "sfRole"]
    cred_name = ["user", "password", "database", "role"]
    for option, val in zip(options_names, cred_name):
        options[option] = cred[val]

    options.update(additional_options)

    return options


def get_snowflake_connection(conf, db, **additional_options):
    """Returns snowflake connection for python snowflake connector.
    Args:
        db (str): The database to use (PROD or RAW).
        additional_options (str) : See Snowflake connections https://docs.snowflake.com/en/user-guide/python-connector-example.html#connecting-using-the-default-authenticator
    Returns:
        conn (class) : Snowflake connection
    """ 
    cred = get_snowflake_credentials(conf, db)

    options = {
        "account": 'molotovtv.eu-west-1',
        "warehouse": "DATABRICKS_XS_WH"
    }

    options_names = ["user", "password", "database", "role"]
    for option in options_names:
        options[option] = cred[option]

    options.update(additional_options)
    conn = snowflake.connector.connect(**options)

    return conn


def execute_snowflake_query(conn, query):
    """Excutes snowflake query with python snowflake connector
    Args:
        conn (class): Snowflake connection
        query (str) : SQL query 
    """
    cs = conn.cursor()
    try:
        cs.execute(query)
        query_id = cs.sfqid 
        while conn.is_still_running(conn.get_query_status_throw_if_error(query_id)):
            time.sleep(1)
    except ProgrammingError as err:
        print('Programming Error: {0}'.format(err))
    finally:
        cs.close()


def get_mysql_options(conf, **additional_options):

    db_port = 3306
    db_host = conf.get("db_host", "")
    db_name = conf.get("db_name", "")

    jdbc_url = "jdbc:mysql://{0}:{1}/{2}".format(db_host, db_port, db_name)

    db_username = conf.get("db_username", "")
    db_password = conf.get("db_password", "")

    options = {
        "url": jdbc_url,
        "driver": "com.mysql.jdbc.Driver",
        "user": db_username,
        "password": db_password
    }

    options.update(additional_options)

    return options


def load_snowflake_table(ss, options, table_name):
    return ss.read \
        .format("snowflake") \
        .options(**options) \
        .option("dbtable", table_name) \
        .load()


def load_mysql_table(ss, options, table_name):
    return ss.read \
        .format("jdbc") \
        .options(**options) \
        .option("dbtable", table_name) \
        .load()


def load_snowflake_query_df(ss, options, query):
    return ss.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", query) \
        .load()


def write_df_to_snowflake(df, write_options, table_name, mode):
    df.write \
        .format("snowflake") \
        .options(**write_options) \
        .option("dbtable", table_name) \
        .mode(mode) \
        .save()


def csv_to_dataframe(spark, path):
    return spark.read.csv(path, header="true", inferSchema="true")


def dump_dataframe_to_csv(dbutils, df, csv_path):
    """
    :param df:
    :param csv_path: dir_path where the csv can be dumped
    :return:
    """
    temp_path = os.path.join(csv_path, "dump_location")
    df.coalesce(1). \
        write.format('com.databricks.spark.csv'). \
        mode('overwrite'). \
        option('header', 'true').\
        csv(temp_path)

    temporary_csv_path = next(entry.path for entry in dbutils.fs.ls(temp_path) if entry.name.startswith('part-'))
    return temporary_csv_path


def unpivot_fact_audience(df, categories, other_cols):
    """
    Transform the affinity cols into new rows for each 1
    """
    all_dfs = list()
    for cat in categories:
        all_dfs.append(
            df.
            where(F.col(cat) > 0).
            select(*other_cols, F.col(cat).alias("score"), F.lit(cat).alias("category"))
        )

    new_df = all_dfs[0]
    for sub_df in all_dfs[1:]:
        new_df = new_df.unionAll(sub_df)
    return new_df. \
        withColumn("category", F.regexp_replace(F.col("category"), '"', ""))


def unpivot(df, categories, other_cols):
    """
    Used to recombine
    :param df:
    :param categories:
    :param other_cols:
    :return:
    """
    all_dfs = list()
    for cat in categories:
        all_dfs.append(
            df.where(F.col(cat) > 0).select(*other_cols, F.col(cat).alias("score"), F.lit(cat).alias("category")))
    df2 = all_dfs[0]
    for sub_df in all_dfs[1:]:
        df2 = df2.unionAll(sub_df)
    return df2.withColumn("category", regexp_replace(F.col("category"), '"', ""))


def get_pypsark_hash_function(col_name="ID"):
    return sha2(concat_ws("molotov-salt-", F.col(col_name)), 0)


def trunc_datetime(a_date):
    return a_date.replace(day=1)


def get_fresh_fa_unpivoted(spark, options):
    fact_audience_from_today_df = load_snowflake_table(spark, options, "dw.fact_audience")

    # 2.2 - We need to have a format like (USER_ID, AFFINITY) which is like unpivoting the table
    non_category_columns = {"USER_ID", "RFM7_CLUSTER", "RFM28_CLUSTER"}
    category_columns = set(fact_audience_from_today_df.columns).difference(non_category_columns)
    return unpivot_fact_audience(fact_audience_from_today_df, category_columns, non_category_columns). \
        select("USER_ID", "category")


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def format_tuple_for_sql(tup):
    if len(tup) == 1:
        return f"({tup[0]})"
    else:
        return str(tuple(tup))
