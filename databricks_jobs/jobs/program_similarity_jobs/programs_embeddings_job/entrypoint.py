from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake, load_snowflake_query_df
from databricks_jobs.jobs.utils.embeddings import embed
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType, ArrayType, DoubleType, DateType
from pyspark.sql.functions import col
import datetime as dt
import pandas as pd
from datetime import timedelta
from typing import Tuple
import numpy as np


class ProgramsEmbeddingsJob(Job):
    embedding_bytes_table = "ML.PROGRAMS_EMBEDDINGS"
    embeddings_debug_table = "ML.PROGRAMS_EMBEDDINGS_DEBUG"
    batch_size = 40
    max_programs_to_embed = 10000
    re_compute_after_n_days = 720
    write_legible_debug_table = True

    def __init__(self, *args, **kwargs):
        super(ProgramsEmbeddingsJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.expiry_date = self.now - timedelta(days=self.re_compute_after_n_days)
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 get programs synopsys as pandas df
        program_ids, synopsys = self.get_programs_ids_and_synopsys(embeddings_table=self.embedding_bytes_table,
                                                                   expiry_date=self.expiry_date,
                                                                   limit=self.max_programs_to_embed)

        # 1 filter programs with empty summaries
        def is_empty_summary(s):
            return len(s.strip()) < 5

        empty_summary_indexes = list(map(is_empty_summary, synopsys))  # boolean array
        non_empty_summary_programs_synopsys = synopsys[np.logical_not(empty_summary_indexes)]
        non_empty_summary_programs_ids = program_ids[np.logical_not(empty_summary_indexes)]

        empty_summaries_program_ids = program_ids[empty_summary_indexes]

        # 2 compute embeddings numpy array
        embeddings = embed(non_empty_summary_programs_synopsys, self.batch_size) if synopsys.size > 0 else []

        # 3 create the pandas containing the data to be written on database
        non_empty_summary_data = pd.DataFrame(data={"PROGRAM_ID": non_empty_summary_programs_ids, 'EMBEDDING': embeddings})
        non_empty_summary_data['EMBEDDING_BYTES'] = non_empty_summary_data['EMBEDDING'].apply(lambda x: x.dumps())
        empty_summary_programs_data = pd.DataFrame(data={"PROGRAM_ID": empty_summaries_program_ids, 'EMBEDDING': None})
        data = non_empty_summary_data.append(empty_summary_programs_data)
        data['UPDATE_AT'] = self.now

        # 4 make the spark dataframe to be written in database
        schema = StructType([
            StructField("PROGRAM_ID", IntegerType(), True),
            StructField("EMBEDDING_BYTES", BinaryType(), True),
            StructField("UPDATE_AT", DateType(), True)
        ])
        embeddings_spark_df = self.spark.createDataFrame(data=data[['PROGRAM_ID', 'EMBEDDING_BYTES', 'UPDATE_AT']],
                                                         schema=schema)

        # 5 write tables
        write_df_to_snowflake(embeddings_spark_df, self.write_options,
                              table_name=self.embedding_bytes_table,
                              mode="append")

        # 6 debug table
        if self.write_legible_debug_table:
            debug_schema = StructType([
                StructField("PROGRAM_ID", IntegerType(), True),
                StructField("EMBEDDING", ArrayType(DoubleType()), True),
                StructField("UPDATE_AT", DateType(), True)
            ])
            embeddings_debug_spark_df = self.spark.createDataFrame(data=data[['PROGRAM_ID', 'EMBEDDING', 'UPDATE_AT']],
                                                                   schema=debug_schema)
            write_df_to_snowflake(embeddings_debug_spark_df, self.write_options,
                                  table_name=self.embeddings_debug_table,
                                  mode="append")

    def get_programs_ids_and_synopsys(self, embeddings_table: str,
                                      expiry_date: dt.datetime, limit: int = 100) -> Tuple[np.array, np.array]:
        """
        returns a dataframe with program ids and a synopsis that were found on imdb, wikipedia,
        or translated from french summary already present in molotov databases,
        for programs that have not been embed already, or that have been embed before expiry_date
        """
        scrape_info_table = "ML.SCRAPED_PROGRAMS"
        translated_summaries_table = "ML.TRANSLATED_SUMMARIES_FR_TO_EN"

        query = f"""
        with scraped_ranked_by_date as (
            select 
                *
                , row_number() over (
                  partition by 
                      TITLE,SCRAPE_SOURCE
                  order by
                      SCRAPED_AT desc
                ) as RANK 
            from 
                {scrape_info_table}       
        )
        , imdb as 
            (
            select PROGRAM_ID, TITLE, (CASE WHEN TRUSTED_RESULT=True THEN PLOT ELSE '' END) as PLOT
            from scraped_ranked_by_date
            where 
                RANK = 1
                and SCRAPE_SOURCE = 'imdb' 
            )
        , wikipedia as 
            (
            select PROGRAM_ID,TITLE, (CASE WHEN TRUSTED_RESULT=True THEN PLOT ELSE '' END) as PLOT
            from scraped_ranked_by_date
            where 
                RANK = 1
                and SCRAPE_SOURCE = 'wikipedia' 
            )
        , translated_summaries as 
            (
            select program.ID as PROGRAM_ID,program.TITLE_ORIGINAL as TITLE, translated.EN_SUMMARY as EN_SUMMARY
            from {translated_summaries_table} as translated
            join BACKEND.PROGRAM as program on program.ID = translated.PROGRAM_ID
            QUALIFY ROW_NUMBER() OVER (partition by program.ID order by UPDATE_AT desc) = 1
            )
        , all_sources as 
            (
            select 
                program.ID as PROGRAM_ID
                , imdb.PLOT as IMDB
                , wikipedia.PLOT as WIKIPEDIA
                , translated_summaries.EN_SUMMARY as EN_SUMMARY
            from 
                backend.PROGRAM
            left join 
                imdb on program.ID = imdb.PROGRAM_ID
            left join 
                wikipedia on program.ID = wikipedia.PROGRAM_ID
            left join 
                translated_summaries on program.ID = translated_summaries.PROGRAM_ID
            where 
                program.ID in ( select PROGRAM_ID from {translated_summaries_table} )
            )
        , not_to_recompute as 
            (
            select 
                * 
            from 
                {embeddings_table}
            where 
                DATE(UPDATE_AT) > DATE('{expiry_date.strftime("%Y-%m-%d")}')
            )
        select 
            PROGRAM_ID
            , concat(coalesce(IMDB,''),' ',coalesce(WIKIPEDIA,''),' ',coalesce(EN_SUMMARY,'')) as SYNOPSYS
        from 
            all_sources
        where 
            PROGRAM_ID not in (select PROGRAM_ID from not_to_recompute )
        order by 
            PROGRAM_ID
        limit {limit}
            """
        df = load_snowflake_query_df(self.spark, self.options, query) \
            .withColumn("PROGRAM_ID", col("PROGRAM_ID").cast("int")) \
            .toPandas()
        return df['PROGRAM_ID'].values, df['SYNOPSYS'].values


if __name__ == "__main__":
    job = ProgramsEmbeddingsJob()
    job.launch()
