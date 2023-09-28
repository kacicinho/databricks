from databricks_jobs.common import Job
from datetime import timedelta
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql.types import StructField, StructType, FloatType, IntegerType
from databricks_jobs.jobs.utils.similarity import get_n_nearest_embeddings, compute_index
from databricks_jobs.db_common import build_episode_df, build_vod_df_with_episode_infos
from databricks_jobs.jobs.utils.channels.ensemble import MANGO_CHANNELS
import faiss


class SynopsysBasedProgramSimilarityJob(Job):
    programs_embeddings_table = "ML.PROGRAMS_EMBEDDINGS"
    similarity_table = "ML.PROGRAM_PROGRAM_SIMILARITY"
    mango_similarity_table = "ML.MANGO_PROGRAM_PROGRAM_SIMILARITY"
    mango_reco_table = "ML.RECO_MANGO_TRANSFORMERS_PROGRAM_PROGRAM_SIMILARITY"
    dbfs_index_folder_path = '/dbfs/FileStore/faiss_indexes/'
    N_SIMILAR = 1000

    def __init__(self, *args, **kwargs):
        super(SynopsysBasedProgramSimilarityJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")
        self.delta = timedelta(days=3)

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 load all program embeddings
        embeddings_spark_df = load_snowflake_table(self.spark, self.options, self.programs_embeddings_table)
        # 1.1 all programs
        all_embeddings_pandas_df = (
            embeddings_spark_df
            .withColumn("ID", F.col("PROGRAM_ID").cast(IntegerType()))
            .toPandas().dropna()
        )
        # 1.2 only mango programs
        episode_info_df = build_episode_df(self.spark, self.options)
        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta)
        mango_vod_df = vod_df.where(F.col("CHANNEL_ID").isin(*MANGO_CHANNELS.chnl_ids())).select(F.col('PROGRAM_ID'))
        mango_vod_df = mango_vod_df.distinct()

        mango_embeddings_pandas_df = (
            embeddings_spark_df
            .join(mango_vod_df, embeddings_spark_df.PROGRAM_ID == mango_vod_df.PROGRAM_ID, "inner")
            .drop(mango_vod_df.PROGRAM_ID)
            .withColumn("ID", F.col("PROGRAM_ID").cast(IntegerType()))
            .toPandas().dropna()
        )

        # 2 compute similarities
        # 2.1 all programs vs all programs
        all_programs_index = compute_index(all_embeddings_pandas_df)
        all_similarities_pandas_df = get_n_nearest_embeddings(all_programs_index, all_embeddings_pandas_df,
                                                              self.N_SIMILAR)
        # 2.2 mango programs vs mangp programs
        mango_programs_index = compute_index(mango_embeddings_pandas_df)
        mango_similarities_pandas_df = get_n_nearest_embeddings(mango_programs_index, mango_embeddings_pandas_df,
                                                                self.N_SIMILAR)

        # 3 write search_indexes files to dbfs
        # 3.1 all programs
        faiss.write_index(all_programs_index, self.dbfs_index_folder_path + 'all_programs_index.faiss')
        # 3.2 mango programs index
        faiss.write_index(mango_programs_index, self.dbfs_index_folder_path + 'mango_programs_index.faiss')

        # 4 write similarities on DB
        schema = StructType([
            StructField('PROGRAM_ID', IntegerType(), True),
            StructField('SIMILAR_PROGRAM_ID', IntegerType(), True),
            StructField('SIMILARITY_SCORE', FloatType(), True)
        ])
        # 4.1 all programs
        all_similarities_spark_df = self.spark.createDataFrame(data=all_similarities_pandas_df, schema=schema) \
            .withColumn('UPDATE_AT', F.lit(self.now))
        write_df_to_snowflake(df=all_similarities_spark_df, write_options=self.write_options,
                              table_name=self.similarity_table,
                              mode='overwrite')

        # 4.2 mango programs
        mango_similarities_spark_df = self.spark.createDataFrame(data=mango_similarities_pandas_df, schema=schema) \
            .withColumn('UPDATE_AT', F.lit(self.now))
        write_df_to_snowflake(df=mango_similarities_spark_df, write_options=self.write_options,
                              table_name=self.mango_similarity_table,
                              mode='overwrite')

        # 4.3 mango reco table
        window = Window.partitionBy('PROGRAM_ID').orderBy(F.col('SIMILARITY_SCORE').desc())
        mango_recos = (
            mango_similarities_spark_df
            .where(F.col("PROGRAM_ID") != F.col("SIMILAR_PROGRAM_ID"))
            .withColumn("rank", F.row_number().over(window))
            .where(F.col('rank') <= 30)
            .groupby("PROGRAM_ID")
            .agg(
                F.collect_list(
                    F.struct(
                        F.col("SIMILAR_PROGRAM_ID").alias("program_id"),
                        F.col("SIMILARITY_SCORE").alias("rating")
                    )
                ).alias("recommendations")
            )
            .withColumn("method", F.lit("synopsis"))
            .withColumn('UPDATE_DATE', F.lit(self.now))
        )
        write_df_to_snowflake(df=mango_recos, write_options=self.write_options,
                              table_name=self.mango_reco_table,
                              mode='overwrite')


if __name__ == "__main__":
    job = SynopsysBasedProgramSimilarityJob()
    job.launch()
