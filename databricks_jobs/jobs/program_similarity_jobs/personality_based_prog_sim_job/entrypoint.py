from datetime import timedelta
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
import faiss
from pyspark.sql.window import Window

from databricks_jobs.common import Job

from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos, MANGO_CHANNELS


class PersonalityBasedProgramSimilarityJob(Job):
    BAN_PERSON_ID = [169046]

    def __init__(self, *args, **kwargs):
        super(PersonalityBasedProgramSimilarityJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=14)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # Selection of program
        prog_df = self.prepare_program_df()

        # Build personalities similarity matrix
        person_person_df = self.compute_n_nearest_personalities(n=100)

        # Keep top personality for users
        top_person_per_prog_df = self.build_program_with_personality(prog_df, person_person_df)

        # Build prog similarity
        prog_sim_matrix = self.build_prog_sim(top_person_per_prog_df)

        # Write to reco
        self.write_recos_to_snowflake(prog_sim_matrix)

    def prepare_program_df(self):
        episode_info_df = build_episode_df(self.spark, self.options)

        # Import Broadcast and VOD program available
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False)

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True)

        # Filter only on Mango channels for the moment
        prog_df = broadcast_df.union(vod_df). \
            where(F.col("CHANNEL_ID").isin(*MANGO_CHANNELS)). \
            select("PROGRAM_ID", "EPISODE_ID"). \
            dropDuplicates()

        return prog_df

    def build_program_with_personality(self, prog_df, person_person_df):
        episode_person_df = load_snowflake_table(self.spark, self.options, "backend.rel_episode_person"). \
            where(~F.col("PERSON_ID").isin(*self.BAN_PERSON_ID))
        user_follow_df = load_snowflake_table(self.spark, self.options, "backend.user_follow_person")

        # Count number of follows per personality
        # we use the log to reduce the effect of too followed people and sqrt for inner product while computing score
        count_follows_person_df = user_follow_df. \
            where((user_follow_df.SOURCE == "molotov") & (user_follow_df.CREATED_AT <= self.now)). \
            groupby("PERSON_ID"). \
            agg(F.countDistinct("USER_ID").alias("tot_fols")). \
            withColumn("tot_fols", F.sqrt(F.log10(1 + F.col("tot_fols"))))

        # Select most representatives personalities for each program
        prog_with_personality_df = prog_df. \
            join(episode_person_df, prog_df.EPISODE_ID == episode_person_df.EPISODE_ID). \
            drop(episode_person_df.EPISODE_ID). \
            select("PROGRAM_ID", "EPISODE_ID", "PERSON_ID"). \
            withColumn("percent_presence",
                       100 * (F.count("EPISODE_ID").over(Window.partitionBy("PROGRAM_ID", "PERSON_ID")) /
                              F.count("EPISODE_ID").over(Window.partitionBy("PROGRAM_ID")))). \
            where(F.col("percent_presence") > 0.3). \
            select("PROGRAM_ID", "PERSON_ID"). \
            dropDuplicates()

        personality_available_list = prog_with_personality_df.select("PERSON_ID"). \
            toPandas()["PERSON_ID"].unique()

        # Keep similar similar personality with a score > 0.7
        person_person_df = person_person_df. \
            where(F.col("SCORE") > 0.7). \
            where(F.col("SIMILAR_PERSON_ID").isin(*personality_available_list))

        # Add similar person to programs with IS_SIMILAR_CAST flag
        prog_with_similar_personality_df = prog_with_personality_df. \
            join(person_person_df, "PERSON_ID", "left"). \
            withColumn("IS_SIMILAR_CAST", F.when((F.col("PERSON_ID") == F.col("SIMILAR_PERSON_ID")) |
                                                 (F.col("SIMILAR_PERSON_ID").isNull()), F.lit(0)).
                       otherwise(F.lit(1))). \
            withColumn("SIMILAR_PERSON_ID", F.coalesce("SIMILAR_PERSON_ID", "PERSON_ID")). \
            select("PROGRAM_ID", F.col("SIMILAR_PERSON_ID").alias("PERSON_ID"), "IS_SIMILAR_CAST"). \
            dropDuplicates()

        # Compute person_score based on total follows and weighted for Similar person
        top_person_per_prog_df = prog_with_similar_personality_df. \
            join(count_follows_person_df,
                 count_follows_person_df.PERSON_ID == prog_with_similar_personality_df.PERSON_ID, "left"). \
            drop(count_follows_person_df.PERSON_ID). \
            na.fill({'tot_fols': 0.5}). \
            withColumn("person_score", F.when(F.col("IS_SIMILAR_CAST") == 0, F.lit(1) * F.col("tot_fols")).
                       otherwise(
                F.lit(0.1) * F.col("tot_fols") / F.sum("IS_SIMILAR_CAST").over(Window.partitionBy("PROGRAM_ID")))). \
            select("PROGRAM_ID", "PERSON_ID", "person_score"). \
            dropDuplicates(). \
            groupBy("PROGRAM_ID"). \
            pivot("PERSON_ID", personality_available_list). \
            sum("person_score").na.fill(0)

        return top_person_per_prog_df

    def build_prog_sim(self, top_person_per_prog_df):

        input_cols = list(top_person_per_prog_df.columns)[1:]
        # We create Vector by combining columns
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="features")

        output = assembler.transform(top_person_per_prog_df)
        output_df = output.toPandas()
        # features and ids are converted in order to use faiss operations
        output_df["features"] = output_df["features"].apply(lambda x: np.float32(x.toArray()))
        output_array = np.stack(output_df["features"].values)

        d = output_array.shape[1]
        ids = output_df["PROGRAM_ID"].values.astype(np.int64)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))  # IP stands for Inner Product
        faiss.normalize_L2(output_array)  # normalize vector
        index.add_with_ids(output_array, ids)

        similarities = index.search(output_array, len(ids))
        data = []
        for prog_id, similar_prog_ids, scores in zip(ids, similarities[1], similarities[0]):
            for s_id, value in zip(similar_prog_ids, scores):
                data.append([prog_id, s_id, value])

        schema = StructType([
            StructField('PROGRAM_ID', IntegerType(), True),
            StructField('SIMILAR_PROG_ID', IntegerType(), True),
            StructField('SCORE', FloatType(), True)
        ])

        p_df = pd.DataFrame(data)
        res_df = self.spark.createDataFrame(data=p_df, schema=schema)

        return res_df

    def compute_n_nearest_personalities(self, n=50):
        person_info_all_df = load_snowflake_table(self.spark, self.options, "ML.PERSON_VECTORISATION_FEATURES")

        input_cols = list(person_info_all_df.columns)[1:]
        # We create Vector by combining columns
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="features")

        output = assembler.transform(person_info_all_df)
        output_df = output.toPandas()
        # features and ids are converted in order to use faiss operations
        output_df["features"] = output_df["features"].apply(lambda x: np.float32(x.toArray()))
        output_array = np.stack(output_df["features"].values)

        d = output_array.shape[1]
        ids = output_df["PERSON_ID"].values.astype(np.int64)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))  # IP stands for Inner Product
        faiss.normalize_L2(output_array)  # normalize vector
        index.add_with_ids(output_array, ids)

        similarities = index.search(output_array, n)
        data = []
        for person_id, similar_person_ids, scores in zip(ids, similarities[1], similarities[0]):
            for s_id, value in zip(similar_person_ids, scores):
                data.append([person_id, s_id, round(value, 6)])

        schema = StructType([
            StructField('PERSON_ID', IntegerType(), True),
            StructField('SIMILAR_PERSON_ID', IntegerType(), True),
            StructField('SCORE', FloatType(), True)
        ])

        p_df = pd.DataFrame(data)
        res_df = self.spark.createDataFrame(data=p_df, schema=schema)

        return res_df

    def write_recos_to_snowflake(self, sim_matrix_df):

        sim_progs_df = sim_matrix_df. \
            where(F.col("PROGRAM_ID") != F.col("SIMILAR_PROG_ID")). \
            where(F.col("SCORE") > 0). \
            withColumn("rank",
                       F.row_number().over(Window.partitionBy("PROGRAM_ID")
                                           .orderBy(F.desc("SCORE")))). \
            where("rank < 30"). \
            drop("rank"). \
            groupby("PROGRAM_ID"). \
            agg(F.collect_list(F.struct(F.col("SIMILAR_PROG_ID").alias("program_id"),
                                        F.col("SCORE").alias("rating"))).alias("recommendations")). \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("METHOD", F.lit("personality"))
        write_df_to_snowflake(sim_progs_df, self.write_options, "RECO_PROG_PROGS_PERSONALITY", 'append')

        # Debug
        # write_df_to_snowflake(sim_matrix_df. where(F.col("SCORE") > 0), self.write_options, "PROG_PERSON_SIMILARITY_MATRIX", "overwrite")
        # write_df_to_snowflake(person_person_df, self.write_options, "PERSON_PERSON_SIMILARITY", "overwrite")


if __name__ == "__main__":
    job = PersonalityBasedProgramSimilarityJob()
    job.launch()
