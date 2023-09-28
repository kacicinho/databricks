from databricks_jobs.common import Job
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.jobs.utils.utils import load_snowflake_table, \
    write_df_to_snowflake, get_snowflake_options
from datetime import timedelta


class PSubMetricsJob(Job):
    T_LOW = 0.2
    T_MEDIUM = 0.3
    T_HIGH = 0.75

    def __init__(self, *args, **kwargs):
        super(PSubMetricsJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=1)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        psub_metrics_df = self.prepare_data()
        self.write_metrics(psub_metrics_df)

    def prepare_data(self):
        # Loading psub predictions and user subscriptions info
        daily_sub_pred_df = load_snowflake_table(self.spark, self.options, "dw.daily_sub_pred"). \
            where(F.col("VERSION") == 1)
        user_sub_df = load_snowflake_table(self.spark, self.options, "temp.fact_user_subscription_day_bill_dev")
        # temp à remplacer après migration
        fact_watch_df = load_snowflake_table(self.spark, self.options, "dw.fact_watch"). \
            where(F.col("DATE_DAY") == self.now)

        # Select people with at least 1 subscription in their historic
        already_sub_df = daily_sub_pred_df. \
            join(user_sub_df, user_sub_df.USER_ID == daily_sub_pred_df.USER_ID). \
            drop(user_sub_df.USER_ID). \
            where((F.col("STATUS_DAY_DATE") < self.now - self.delta) & (F.col("PRED_DATE") == self.now - self.delta)). \
            where(F.col("IS_SUB_ACTIVE")). \
            withColumn("HAS_ALREADY_SUB", F.lit(1)). \
            select("USER_ID", "HAS_ALREADY_SUB").\
            dropDuplicates()

        # Build infos of subscriptions:
        # -sub cluster
        # -is_sub : a free user become subscriber
        # -is_first_sub : first subscription ever on molotov
        # -is_full_sub : subscription with "FULL" as type
        user_info_df = daily_sub_pred_df. \
            withColumn("SUB_CLUSTER", F.when(F.col("PSUB") < self.T_LOW, "psub_low").
                       when((F.col("PSUB") >= self.T_LOW) & (F.col("PSUB") < self.T_MEDIUM), "psub_low_medium").
                       when((F.col("PSUB") >= self.T_MEDIUM) & (F.col("PSUB") < self.T_HIGH), "psub_medium_high").
                       otherwise("psub_high")). \
            join(already_sub_df, "USER_ID", "left"). \
            join(user_sub_df, "USER_ID", "left"). \
            where((F.col("SUB_CLUSTER").isNotNull()) & (
                ((F.col("SUB_BEGIN_DATE") == self.now) & (F.col("STATUS_DAY_DATE") == self.now)) | (
                    F.col("STATUS_DAY_DATE").isNull())) & (
                        F.col("PRED_DATE") == self.now - self.delta)). \
            withColumn("IS_SUB", F.when(F.col("IS_SUB_ACTIVE") == 'TRUE', 1).otherwise(0)). \
            withColumn("IS_SUB_FULL", F.when(F.col("CURRENT_SUB_TYPE") == "FULL", 1).otherwise(0)). \
            withColumn("IS_FIRST_SUB",
                       F.when((F.col("HAS_ALREADY_SUB") == 1) & (F.col("IS_SUB_ACTIVE") == "TRUE"), 0).
                       when((F.col("HAS_ALREADY_SUB").isNull()) & (F.col("IS_SUB_ACTIVE") == "TRUE"), 1).otherwise(0)).\
            select("USER_ID", "SUB_CLUSTER", "IS_SUB", "IS_SUB_FULL", "IS_FIRST_SUB")

        # Select the most watched program corresponding to the evaluation date
        sub_best_prog_df = user_info_df. \
            join(fact_watch_df, "USER_ID", "left"). \
            groupBy("USER_ID", "SUB_CLUSTER", "IS_SUB", "IS_SUB_FULL", "IS_FIRST_SUB", "PROGRAM_ID"). \
            agg(F.sum("DURATION").alias("TOT_DURATION")). \
            withColumn("PROG_RANK", F.row_number().over(Window.partitionBy("USER_ID").orderBy(F.desc("TOT_DURATION")))). \
            where(F.col("PROG_RANK") == 1). \
            withColumnRenamed("PROGRAM_ID", "BEST_PROG"). \
            select("USER_ID", "SUB_CLUSTER", "IS_SUB", "IS_SUB_FULL", "IS_FIRST_SUB", "BEST_PROG")

        return sub_best_prog_df

    def write_metrics(self, psub_metrics_df):
        psub_metrics_df = psub_metrics_df. \
            withColumn("DATE_DAY", F.lit(self.now))

        write_df_to_snowflake(psub_metrics_df, self.write_options, "PSUB_METRICS", "append")


if __name__ == "__main__":
    job = PSubMetricsJob()
    job.launch()
