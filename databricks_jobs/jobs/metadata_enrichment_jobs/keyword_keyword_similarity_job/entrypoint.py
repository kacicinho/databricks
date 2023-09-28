from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
from pyspark.sql.functions import col, lit
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql.types import IntegerType
import faiss
from pyspark.sql.types import StructField, StructType, FloatType
from databricks_jobs.jobs.utils.similarity import get_n_nearest_embeddings, compute_index


class KeywordKeywordSimilarityJob(Job):
    keywords_embeddings_tables = ["ML.IMDB_KEYWORDS_EMBEDDINGS", "ML.KEYWORDS_EMBEDDINGS"]
    similarity_tables = ["ML.IMDB_KEYWORD_KEYWORD_SIMILARITY", "ML.KEYWORD_KEYWORD_SIMILARITY"]
    index_names = ["imdb_index.faiss", "in_house_kw_index.faiss"]
    N_SIMILAR = 100
    dbfs_index_folder_path = '/dbfs/FileStore/faiss_indexes/'

    def __init__(self, *args, **kwargs):
        super(KeywordKeywordSimilarityJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        for keywords_embeddings_table, similarity_table, index_name in zip(self.keywords_embeddings_tables,
                                                                           self.similarity_tables,
                                                                           self.index_names):
            # 1 get keywords embeddings
            embeddings_spark_df = load_snowflake_table(self.spark, self.options, keywords_embeddings_table)
            embeddings_pandas_df = (
                embeddings_spark_df
                .withColumn("ID", col("ID").cast(IntegerType()))
                .toPandas()
            )

            # 2 compute similarities
            index = compute_index(embeddings_pandas_df)
            similarities_pandas_df = get_n_nearest_embeddings(index, embeddings_pandas_df, self.N_SIMILAR)

            # 3 write search_indexes files to dbfs
            faiss.write_index(index, self.dbfs_index_folder_path + index_name)

            # 4 write similarities on DB
            schema = StructType([
                StructField('KEYWORD_ID', IntegerType(), True),
                StructField('SIMILAR_KEYWORDS_ID', IntegerType(), True),
                StructField('SIMILARITY_SCORE', FloatType(), True)
            ])
            all_similarities_spark_df = self.spark.createDataFrame(data=similarities_pandas_df, schema=schema) \
                .withColumn('UPDATE_AT', lit(self.now))
            write_df_to_snowflake(df=all_similarities_spark_df, write_options=self.write_options,
                                  table_name=similarity_table,
                                  mode='overwrite')


if __name__ == "__main__":
    job = KeywordKeywordSimilarityJob()
    job.launch()
