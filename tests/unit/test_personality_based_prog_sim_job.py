from unittest import TestCase
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
import pandas as pd
import faiss
import numpy as np
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.jobs.program_similarity_jobs.personality_based_prog_sim_job.entrypoint import \
    PersonalityBasedProgramSimilarityJob
from tests.unit.utils.mocks import create_spark_df_from_data

from pyspark.sql import SparkSession


class TestPersonalityBasedProgramSimilarityJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        # Only 1 partition to speed up test
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        cls.job = PersonalityBasedProgramSimilarityJob(spark=spark)

    def test_person_similarity(self):

        data = {"PROGRAM_ID": [0, 0, 0, 1, 1, 1, 2, 2],
                "EPISODE_ID": [1, 2, 3, 4, 5, 6, 7, 8],
                "PERSON_ID": [1, 1, 2, 1, 4, 5, 2, 2],
                "tot_fols": [10, 10, 100.0, 10, 0, 0, 100.0, 100.0]
                }

        episode_with_person_df = create_spark_df_from_data(self.job.spark, data)

        program_with_person_df = episode_with_person_df. \
            withColumn("percent_presence",
                       100 * (F.count("EPISODE_ID").over(Window.partitionBy("PROGRAM_ID", "PERSON_ID")) /
                              F.count("EPISODE_ID").over(Window.partitionBy("PROGRAM_ID")))). \
            where(F.col("percent_presence") > 0.3). \
            withColumn("tot_fols", F.log10(F.col("tot_fols"))).\
            select("PROGRAM_ID", "PERSON_ID", "tot_fols"). \
            dropDuplicates(). \
            groupBy("PROGRAM_ID"). \
            pivot("PERSON_ID"). \
            sum("tot_fols").na.fill(0)

        input_cols = list(program_with_person_df.columns)[1:]

        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="features")

        output = assembler.transform(program_with_person_df)
        output_df = output.toPandas()
        output_df["features"] = output_df["features"].apply(lambda x: np.float32(x.toArray()))
        output_array = np.stack(output_df["features"].values)

        d = output_array.shape[1]
        ids = output_df["PROGRAM_ID"].values.astype(np.int64)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))  # IP stands for Inner Product
        index.add_with_ids(output_array, ids)

        similarities = index.search(output_array, len(ids))
        data = []
        for kw_id, similar_kw_ids, scores in zip(ids, similarities[1], similarities[0]):
            for s_id, value in zip(similar_kw_ids, scores):
                data.append([kw_id, s_id, value])

        schema = StructType([
            StructField('PROG_ID', IntegerType(), True),
            StructField('SIMILAR_PROG_ID', IntegerType(), True),
            StructField('SCORE', FloatType(), True)
        ])

        p_df = pd.DataFrame(data)
        s_df = self.job.spark.createDataFrame(data=p_df, schema=schema)
        s_df.show()

        prog_0_2_similarity = s_df.where((F.col("PROG_ID") == 0) & (F.col("SIMILAR_PROG_ID") == 2)).collect()

        # Similarity between prog 0 and 2 is log(100)^2
        self.assertEqual(prog_0_2_similarity[0]['SCORE'], 4)
