from databricks_jobs.common import Job
from datetime import timedelta
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql.types import StructField, StructType, FloatType, IntegerType
from databricks_jobs.jobs.utils.similarity import get_n_nearest_embeddings, compute_index
from databricks_jobs.db_common import build_episode_df, build_vod_df_with_episode_infos, \
    build_broadcast_df_with_episode_info, MANGO_CHANNELS
import faiss


class TagBasedProgramSimilarityJobAlternate(Job):
    programs_embeddings_table = "ML.PROGRAMS_EMBEDDINGS"
    DAILY_RECO_TABLE = "RECO_PROG_PROGS_META_VAR"  # Needs to write in this table with variation B
    AB_TEST_VARIATION = 'B'
    dbfs_index_folder_path = '/dbfs/FileStore/faiss_indexes/'
    N_SIMILAR = 1000

    def __init__(self, *args, **kwargs):
        super(TagBasedProgramSimilarityJobAlternate, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")
        self.delta = timedelta(days=3)

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Synopsis based program sim.
        """

        # 1 load all program embeddings
        embeddings_spark_df = load_snowflake_table(self.spark, self.options, self.programs_embeddings_table)

        episode_info_df = build_episode_df(self.spark, self.options)
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False). \
            where("REF_PROGRAM_KIND_ID != 93")

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True). \
            where("REF_PROGRAM_KIND_ID != 93")

        mango_vod_df = broadcast_df.union(vod_df). \
            where(F.col("CHANNEL_ID").isin(*MANGO_CHANNELS)). \
            select(F.col('PROGRAM_ID'))
        mango_vod_df = mango_vod_df.distinct()

        mango_embeddings_pandas_df = (
            embeddings_spark_df
            .join(mango_vod_df, embeddings_spark_df.PROGRAM_ID == mango_vod_df.PROGRAM_ID, "inner")
            .drop(mango_vod_df.PROGRAM_ID)
            .withColumn("ID", F.col("PROGRAM_ID").cast(IntegerType()))
            .toPandas().dropna()
        )

        # 2 mango programs vs mango programs
        mango_programs_index = compute_index(mango_embeddings_pandas_df)
        mango_similarities_pandas_df = get_n_nearest_embeddings(mango_programs_index, mango_embeddings_pandas_df,
                                                                self.N_SIMILAR)

        # 3 write mango program search_indexes files to dbfs
        faiss.write_index(mango_programs_index, self.dbfs_index_folder_path + 'alternate_mango_programs_index.faiss')

        # 4 write similarities on DB
        schema = StructType([
            StructField('PROGRAM_ID', IntegerType(), True),
            StructField('SIMILAR_PROGRAM_ID', IntegerType(), True),
            StructField('SIMILARITY_SCORE', FloatType(), True)
        ])

        mango_similarities_spark_df = self.spark.createDataFrame(data=mango_similarities_pandas_df, schema=schema)

        # 4.3 mango reco table
        window = Window.partitionBy('PROGRAM_ID').orderBy(F.col('SIMILARITY_SCORE').desc())
        mango_recos = (
            mango_similarities_spark_df
            .where(F.col("PROGRAM_ID") != F.col("SIMILAR_PROGRAM_ID"))
            .withColumn("rank", F.row_number().over(window))
            .where(F.col('rank') <= 30)
            .withColumn("method", F.lit("synopsis"))
            .groupby("PROGRAM_ID", "method")
            .agg(
                F.collect_list(
                    F.struct(
                        F.col("SIMILAR_PROGRAM_ID").alias("program_id"),
                        F.col("SIMILARITY_SCORE").alias("rating")
                    )
                ).alias("recommendations")
            )
            .withColumn('UPDATE_DATE', F.lit(self.now))
            .withColumn("VARIATIONS", F.lit(self.AB_TEST_VARIATION))
        )
        write_df_to_snowflake(df=mango_recos, write_options=self.write_options,
                              table_name=self.DAILY_RECO_TABLE,
                              mode='append')


if __name__ == "__main__":
    job = TagBasedProgramSimilarityJobAlternate()
    job.launch()
