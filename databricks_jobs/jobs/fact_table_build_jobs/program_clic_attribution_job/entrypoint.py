from datetime import timedelta

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, load_snowflake_table, \
    write_df_to_snowflake, get_snowflake_connection, execute_snowflake_query


class ProgramClicAttributionJob(Job):
    TABLE_NAME = "FACT_PROGRAM_CLIC_ATTRIBUTION"

    def __init__(self, *args, **kwargs):
        super(ProgramClicAttributionJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=2)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "DW", keep_column_case="off")

        self.logger.info(f"Running on date : {self.now}")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        # 1 - Delete past 2 days => This allows to have an attribution of at least 1 day
        self.delete_past_2days()

        # 2 - Recompute clic attrbition on past 2 days.
        program_clic_attribution_df = self.build_program_clic_attribution()

        # 3 - Write updated data to snowflake
        write_df_to_snowflake(program_clic_attribution_df, self.write_options, self.TABLE_NAME, 'append')

    def build_program_clic_attribution(self):
        """
        For each click on the similar program slug, we attribute the last program page. If there isn't a click on the slug,
        then there is no attribution.
        The attribution window is at least 1 day.
        NB: attributions on the last day will not be completed, so the data is valid at d-2"""

        w = Window.partitionBy("user_id").orderBy(F.desc("timestamp"))
        program_clic_attribution_df = load_snowflake_table(self.spark, self.options, "dw.fact_page"). \
            filter(F.col('event_date') >= self.now - self.delta). \
            filter(F.col("page_name").isin(*["program"])). \
            withColumn("next_prog", F.lag(F.struct(*["timestamp", "program_id", "channel_id", "origin_section"]), 1).over(w)). \
            withColumn("next_prog", F.when(F.col("next_prog.origin_section") == 'similar-programs', F.col("next_prog")).otherwise(None)). \
            select("user_id", "timestamp", "origin_section", "program_id", "channel_id", "next_prog")

        return program_clic_attribution_df

    def delete_past_2days(self):

        conn = get_snowflake_connection(self.conf, 'PROD')

        query = f"""
        DELETE FROM DW.{self.TABLE_NAME}
        WHERE timestamp >= TIMESTAMPADD('day', -2, '{self.now}')
        """

        execute_snowflake_query(conn, query)


if __name__ == "__main__":
    job = ProgramClicAttributionJob()
    job.launch()
