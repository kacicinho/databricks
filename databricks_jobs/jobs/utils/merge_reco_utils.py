from pyspark.sql.types import IntegerType, StructField, StructType, StringType, FloatType, DateType
from typing import Literal
from pydantic import BaseModel
from typing import NamedTuple


MERGE_RECO_TABLE_VALUES = Literal["RECO_USER_PROGS_THIS_WEEK_VAR", "USER_REPLAY_RECOMMENDATIONS_VAR",
                                  "RECO_PROG_PROGS_META_VAR"]
LATEST_RECO_TABLE_VALUES = Literal["RECO_USER_PROGS_THIS_WEEK_LATEST", "USER_REPLAY_RECOMMENDATIONS_LATEST",
                                   "RECO_PROG_PROGS_META_LATEST"]
KEY_NAME_VALUES = Literal["USER_ID", "PROGRAM_ID"]


class MergeInfos(BaseModel):
    """Records information about the merge"""
    merge_reco_table: MERGE_RECO_TABLE_VALUES
    latest_reco_table: LATEST_RECO_TABLE_VALUES
    key_name: KEY_NAME_VALUES


class QueryInfos(NamedTuple):
    """Records information about the ab test split table and the query to get raw data depending on the key_name"""
    query: str
    split_table_name: str
    split_table_schema: str

    @classmethod
    def from_key_name(cls, key_name):
        if key_name == "USER_ID":
            table = "backend.user_raw"
            condition = "WHERE PUBLIC.TRANSFORM_SUBSCRIPTION_EMAIL_MOLOTOV(email) = FALSE AND app = 'molotov'"
            split_table_name = "dw.user_split_ab_test"

        elif key_name == "PROGRAM_ID":
            table = "backend.program"
            split_table_name = "dw.program_split_ab_test"
            condition = ""

        split_table_schema = StructType([
            StructField("AB_TEST_ID", IntegerType(), nullable=False),
            StructField(key_name, IntegerType(), nullable=False),
            StructField("VARIATIONS", StringType(), nullable=False)
        ])

        return cls(query=f"""SELECT ID AS {key_name} FROM {table} {condition}""",
                   split_table_name=split_table_name,
                   split_table_schema=split_table_schema)


ab_test_conf_df_schema = StructType([
    StructField("AB_TEST_ID", IntegerType(), nullable=False),
    StructField("DESCRIPTION", IntegerType(), nullable=False),
    StructField("ALT_PROPORTION", FloatType(), nullable=False),
    StructField("MERGE_RECO_TABLE", StringType(), nullable=False),
    StructField("LATEST_RECO_TABLE", StringType(), nullable=False),
    StructField("START_AT", DateType(), nullable=False),
    StructField("END_AT", DateType(), nullable=False)
])
