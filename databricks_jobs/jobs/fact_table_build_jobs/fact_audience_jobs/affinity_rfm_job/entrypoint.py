import pyspark.sql.functions as F
from pyspark.sql import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_table, write_df_to_snowflake, \
    get_snowflake_options


class AffinityRFMJob(Job):

    def __init__(self, *args, **kwargs):
        super(AffinityRFMJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.rfm_window_slides = [7, 28]
        self.options = get_snowflake_options(self.conf, 'PROD', 'DW')
        self.write_options = get_snowflake_options(self.conf, 'PROD', 'DW', **{"keep_column_case": "on"})
        self.logger.info(f"Running on date : {self.now}")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :
            Updates FACT_AUDIENCE with RFM7 & RFM28 CLUSTER
            Computes RFM7 & RFM28 basic stats
        """
        self.logger.info("Launching Affinity RFM job")

        # 1. Update FACT_AUDIENCE with RFM7 & RMF28 CLUSTER
        fact_audience_df = self.add_rfm7_rfm28_to_fact_audience_job()
        fact_audience_df = self.add_psub_predictions(fact_audience_df)
        write_df_to_snowflake(fact_audience_df, self.write_options, 'FACT_AUDIENCE', 'overwrite')

        # 2. Compute RFM7 & RFM28 basic stats
        rfm_stats_dfs = self.compute_rfm_stats_job()

        for rfm_table_name, rfm_df in rfm_stats_dfs.items():
            write_df_to_snowflake(rfm_df, self.write_options, rfm_table_name, "append")

    def add_psub_predictions(self, fact_audience_df):
        """
        Adds to fact_audience 4 pSub segments + offers flags
        """
        psub_df = load_snowflake_table(self.spark, self.options, "dw.daily_sub_pred_flag"). \
            select("USER_ID", "SUB_CLUSTER", "OFFER")
        return fact_audience_df. \
            join(psub_df, psub_df.USER_ID == fact_audience_df.USER_ID, "left"). \
            drop(psub_df.USER_ID)

    def add_rfm7_rfm28_to_fact_audience_job(self):
        """
        Add RFM7 & RFM28 Cluster to FACT_AUDIENCE
        """

        rfm7_df = load_snowflake_table(self.spark, self.options, "dw.fact_rfm_window_slide_01d_size_07d"). \
            filter(F.col("update_date") == self.now)
        rfm28_df = load_snowflake_table(self.spark, self.options, "dw.fact_rfm_window_slide_01d_size_28d"). \
            filter(F.col("update_date") == self.now)
        fa_aff_only_flag_df = load_snowflake_table(self.spark, self.options, "dw.fact_audience_aff_only_flag"). \
            filter(F.col('"date"') == self.now)

        # 1 - Select desired users for fact_audience. We add here inactifs new reg users.
        fa_users_df = rfm7_df.select("USER_ID"). \
            filter(F.col("rfm_cluster") == "inactifs_new_reg"). \
            union(fa_aff_only_flag_df.select("USER_ID")). \
            dropDuplicates()

        # 2 - Left join with rfm and fact audience flag.
        rfm7_df = rfm7_df.withColumnRenamed("RFM_CLUSTER", "RFM7_CLUSTER")
        rfm28_df = rfm28_df.withColumnRenamed("RFM_CLUSTER", "RFM28_CLUSTER")

        cols_to_agg = [c for c in fa_aff_only_flag_df.columns
                       if c not in ['"date"', "USER_ID", "date"]]

        fact_audience_df = fa_users_df.join(rfm7_df, fa_users_df.USER_ID == rfm7_df.USER_ID, how='left'). \
            join(rfm28_df, fa_users_df.USER_ID == rfm28_df.USER_ID, how='left'). \
            join(fa_aff_only_flag_df, fa_users_df.USER_ID == fa_aff_only_flag_df.USER_ID, how='left'). \
            select(fa_users_df.USER_ID, *cols_to_agg, "RFM7_CLUSTER", "RFM28_CLUSTER")

        # 3 - Set affinities to 0 for unmatched users. 
        # Unmatched clusters stays at null indicating none clusterized users.
        fact_audience_df = fact_audience_df.fillna(value=0, subset=cols_to_agg)

        return fact_audience_df

    def compute_rfm_stats_job(self):
        """
        Compute basic stats on the RFM tables
        """

        rfm_dfs = [load_snowflake_table(
            self.spark, self.options, f"dw.fact_rfm_window_slide_01d_size_{str(slide).zfill(2)}d").
            filter(F.col("update_date") == self.now) for slide in self.rfm_window_slides]
        rfm_table_names = ['RFM{}_DAILY_STATS'.format(str(slide).zfill(2))
                           for slide in self.rfm_window_slides]

        def compute_stats(df):
            return df. \
                groupBy('RFM_CLUSTER').count(). \
                withColumn('percentage',
                           F.round(F.col('count') / F.sum('count').over(Window.partitionBy()), 3)). \
                withColumn("compute_date", F.lit(self.now))

        return {rfm_table_name: compute_stats(rfm_df) for rfm_table_name, rfm_df in zip(rfm_table_names, rfm_dfs)}


if __name__ == "__main__":
    job = AffinityRFMJob()
    job.launch()
