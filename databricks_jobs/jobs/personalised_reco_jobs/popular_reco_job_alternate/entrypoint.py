from datetime import timedelta

from pyspark.sql import functions as F
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, \
    write_df_to_snowflake


class PopularRecoJobAlternate(Job):
    DAILY_RECO_TABLE = "RECO_USER_PROGS_THIS_WEEK_VAR"  # Needs to write in this table with variation B
    AB_TEST_VARIATION = 'B'

    def __init__(self, *args, **kwargs):
        super(PopularRecoJobAlternate, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Dummy Job, logic to change
        """
        self.logger.info("This is the alternate popular reco job!")

        # Just for an example purpose, we take the yesterday reco of the lastest table
        user_reco_latest = load_snowflake_table(self.spark, self.options, "ML.RECO_USER_PROGS_THIS_WEEK_LATEST")

        user_reco_latest = user_reco_latest. \
            filter(F.col("UPDATE_DATE") == self.now - timedelta(days=1)). \
            drop(F.col("UPDATE_DATE"))

        # Write reco to snowflake
        self.write_recos_to_snowflake(user_reco_latest)

    def write_recos_to_snowflake(self, user_top_programs_df, table_name=DAILY_RECO_TABLE,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        final_user_recos_df = user_top_programs_df. \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)


if __name__ == "__main__":
    job = PopularRecoJobAlternate()
    job.launch()
