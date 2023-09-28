from databricks_jobs.jobs.utils.item_based_utils import JaccardHasher
from datetime import timedelta
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df, write_df_to_snowflake, get_snowflake_options
from pyspark.sql.window import Window
import pyspark.sql.functions as F


class ItemFilteringProgramSimilarity(Job):
    similarity_table_name = "ML.ITEM_BASED_FILTERING_PROGRAM_SIMILARITY"
    reco_table_name = "RECO_PROG_PROG_ITEM_BASED_FILTERING_VAR"

    def __init__(self, *args, **kwargs):
        super(ItemFilteringProgramSimilarity, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.delta = timedelta(days=7)
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")
        self.nb_monthes_lookback = 6

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 - load fact_watch
        user_item_sdf = self.get_fact_watch()

        # 2 - min hashing
        jaccard_hasher = JaccardHasher(user_item_sdf, node_col="PROGRAM_ID", edge_basis_col="USER_ID", n_hashes=1000)
        similarities_sdf = jaccard_hasher.compute_and_get_similarities()

        # aggregated recos
        window = Window.partitionBy('PROGRAM_ID_I').orderBy(F.col('SIMILARITY_SCORE').desc())
        recos = similarities_sdf \
            .where(F.col("PROGRAM_ID_I") != F.col("PROGRAM_ID_J")) \
            .withColumn("rank", F.row_number().over(window)) \
            .where(F.col('rank') <= 30) \
            .groupby("PROGRAM_ID_I") \
            .agg(F.collect_list(F.struct(F.col("PROGRAM_ID_J").alias("program_id"),
                                         F.col("SIMILARITY_SCORE").alias("rating"))).alias("recommendations")) \
            .withColumn("method", F.lit("item_filtering")) \
            .withColumn('UPDATE_DATE', F.lit(self.now))

        # 3a - write similarities

        write_df_to_snowflake(similarities_sdf, self.write_options, self.similarity_table_name, "overwrite")

        # 3b - write recos
        write_df_to_snowflake(df=recos, write_options=self.write_options,
                              table_name=self.reco_table_name, mode='overwrite')

    def get_fact_watch(self):
        query = f"""
           select
               distinct
               fw.USER_ID,
               fw.PROGRAM_ID
           from
               dw.fact_watch as fw
           where
               1=1
               and abs(datediff(month, current_date(), fw.DATE_DAY)) <= {self.nb_monthes_lookback}
               and DURATION > 900
       """
        return load_snowflake_query_df(self.spark, self.options, query)


if __name__ == "__main__":
    job = ItemFilteringProgramSimilarity()
    job.launch()
