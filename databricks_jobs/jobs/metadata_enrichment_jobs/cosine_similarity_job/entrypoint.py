from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
from pyspark.sql.functions import col, lit
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructField, StructType, FloatType
from databricks_jobs.jobs.utils.similarity import get_n_nearest_embeddings, compute_index


class CosineSimilarityJob(Job):
    similarity_table = "ML.PROGRAM_KEYWORD_SIMILARITY"
    programs_embeddings_table = "ML.PROGRAMS_EMBEDDINGS"
    keywords_embeddings_table = "ML.KEYWORDS_EMBEDDINGS"
    N_SIMILAR = 100

    def __init__(self, *args, **kwargs):
        super(CosineSimilarityJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1.1 get keywords embeddings
        kw_embeddings_spark_df = load_snowflake_table(self.spark, self.options, self.keywords_embeddings_table)
        kw_embeddings_pandas_df = (
            kw_embeddings_spark_df
            .withColumn("ID", col("ID").cast(IntegerType()))
            .toPandas()
        )
        # 1.2 get program embeddings
        prg_embeddings_spark_df = load_snowflake_table(self.spark, self.options, self.programs_embeddings_table)
        prg_embeddings_pandas_df = (
            prg_embeddings_spark_df
            .withColumn("ID", col("PROGRAM_ID").cast(IntegerType()))
            .toPandas()
            .dropna()
        )

        # 2 compute similarities
        kw_index = compute_index(kw_embeddings_pandas_df)
        similarities_pandas_df = get_n_nearest_embeddings(kw_index, prg_embeddings_pandas_df, self.N_SIMILAR)

        # 3 write similarities on DB
        schema = StructType([
            StructField('PROGRAM_ID', IntegerType(), True),
            StructField('KEYWORD_ID', IntegerType(), True),
            StructField('SIMILARITY_SCORE', FloatType(), True)
        ])
        similarities_spark_df = self.spark.createDataFrame(data=similarities_pandas_df, schema=schema) \
            .withColumn('UPDATE_AT', lit(self.now))
        write_df_to_snowflake(df=similarities_spark_df, write_options=self.write_options,
                              table_name=self.similarity_table,
                              mode='overwrite')


if __name__ == "__main__":
    job = CosineSimilarityJob()
    job.launch()
