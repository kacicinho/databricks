from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
from databricks_jobs.jobs.utils.embeddings import embed
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType, DateType
from pyspark.sql.functions import col
from databricks_jobs.jobs.utils.utils import load_snowflake_table
import pandas as pd
from typing import Tuple
import numpy as np
from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame as SparkDataFrame


class KeywordsEmbeddingsJob(Job):
    embedding_bytes_table = "ML.KEYWORDS_EMBEDDINGS"
    ref_keywords_table = "ML.REF_KEYWORDS"

    def __init__(self, *args, **kwargs):
        super(KeywordsEmbeddingsJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 get programs synopsys as pandas df
        keywords_ids, keywords = self.get_keywords_ids_and_keywords()

        # 2 compute embeddings numpy array
        embeddings = embed(keywords, batch_size=40) if keywords.size > 0 else []

        # 3 create the pandas containing the data to be written on database
        data = self.make_pandas_result_df(keywords_ids, embeddings)

        # 4 make the spark dataframe to be written in database
        schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("EMBEDDING_BYTES", BinaryType(), True),
            StructField("UPDATE_AT", DateType(), True)
        ])
        embeddings_spark_df = self.spark.createDataFrame(data=data[['ID', 'EMBEDDING_BYTES', 'UPDATE_AT']],
                                                         schema=schema)

        # 5 write tables
        write_df_to_snowflake(embeddings_spark_df, self.write_options,
                              table_name=self.embedding_bytes_table,
                              mode="append")

    def get_already_embeded_keywords_ids(self) -> SparkDataFrame:
        """ returns the table self.embedding_bytes_table if it exists, and else, an empty dataframe """
        try:
            df = load_snowflake_table(self.spark, self.options, self.embedding_bytes_table) \
                .select('ID')
        except Py4JJavaError as e:
            if "doesn't exist" or "does not exist" in str(e.java_exception):
                empty_df = self.spark.createDataFrame(data=[], schema=StructType([StructField("ID", IntegerType(), True)]))
                return empty_df
            else:
                raise Exception(e)
        return df

    def get_keywords_ids_and_keywords(self) -> Tuple[np.array, np.array]:
        """
        returns keywords ids and keywords themselves
        for keywords that have not been embed already
        """
        already_embeded_keywords_df = self.get_already_embeded_keywords_ids()
        all_keywords_df = load_snowflake_table(self.spark, self.options, self.ref_keywords_table)
        keywords_spark_df = all_keywords_df.join(already_embeded_keywords_df, 'ID', how='left_anti')

        keywords_pandas_df = keywords_spark_df \
            .withColumn("ID", col("ID").cast("int")) \
            .toPandas()

        return keywords_pandas_df['ID'].values, keywords_pandas_df['NAME'].values

    def make_pandas_result_df(self, keywords_ids: np.array, embeddings: np.array):
        data = pd.DataFrame(data={"ID": keywords_ids, 'EMBEDDING': embeddings})
        data['EMBEDDING_BYTES'] = data['EMBEDDING'].apply(lambda x: x.dumps())
        data['UPDATE_AT'] = self.now
        return data


if __name__ == "__main__":
    job = KeywordsEmbeddingsJob()
    job.launch()
