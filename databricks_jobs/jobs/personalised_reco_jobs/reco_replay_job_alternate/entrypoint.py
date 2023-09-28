from datetime import timedelta

from pyspark.sql import functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake, \
    load_snowflake_query_df
from databricks_jobs.jobs.utils.popular_reco_utils import format_for_reco_output


class ReplayRecoAlternateJob(Job):
    DAILY_RECO_TABLE = "USER_REPLAY_RECOMMENDATIONS_VAR"
    AB_TEST_VARIATION = "B"

    def __init__(self, *args, **kwargs):
        super(ReplayRecoAlternateJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=2)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job
        """
        query = \
            """
            with cte_total_replay_watch as (
                select PROGRAM_ID, CHANNEL_ID, sum(DURATION) as total_replay_watch
                from dw.fact_watch
                where date_day > current_date() - 7 and asset_type = 'replay'
                group by PROGRAM_ID, CHANNEL_ID
            ),
            cte_available_replay as (
                SELECT
                DISTINCT p.id,v.episode_id
                , p.title
                , v.channel_id
                , rtc.tvbundle_id
                , t.name
                FROM backend.vod v
                    INNER JOIN backend.episode e ON v.episode_id = e.id
                    INNER JOIN backend.program p ON e.program_id = p.id
                    INNER JOIN backend.channel c ON v.channel_id = c.id
                    INNER JOIN backend.rel_tvbundle_channel rtc ON v.channel_id = rtc.channel_id
                    INNER JOIN backend.tvbundle t ON rtc.tvbundle_id = t.id    
                WHERE v.disabled = 0 
                AND t.is_commercialized = 1
                AND v.available_from <= current_date()
                AND v.available_until >= current_date()
                AND v.withdrawn_at IS NULL
                and v.deleted_at IS NULL
            )
            select 
                distinct PROGRAM_ID, cte_available_replay.CHANNEL_ID, EPISODE_ID, TITLE, 
                         total_replay_watch
            from cte_available_replay
            join cte_total_replay_watch
            ON cte_total_replay_watch.PROGRAM_ID = cte_available_replay.ID and cte_total_replay_watch.CHANNEL_ID = cte_available_replay.CHANNEL_ID
            qualify row_number() over(partition by program_id order by total_replay_watch desc) = 1
            order by total_replay_watch desc
            limit 50
            """
        best_replay = load_snowflake_query_df(self.spark, self.options, query)
        replay_ref_df = load_snowflake_table(self.spark, self.options, "ML.USER_REPLAY_RECOMMENDATIONS_LATEST"). \
            where(f"UPDATE_DATE = '{self.now - self.delta}'"). \
            select("USER_ID")

        df = replay_ref_df. \
            crossJoin(F.broadcast(best_replay)). \
            select("USER_ID", "PROGRAM_ID", "EPISODE_ID", "CHANNEL_ID",
                   F.lit("alternate").alias("reco_origin"), F.col("total_replay_watch").alias("rating"))

        formatted_recos = format_for_reco_output(
            df, "recommendations", field_names=("PROGRAM_ID", "reco_origin", "rating",
                                                "CHANNEL_ID", "EPISODE_ID"))
        self.write_recos_to_snowflake(formatted_recos)

    def write_recos_to_snowflake(self, user_top_programs_df, table_name=DAILY_RECO_TABLE,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        final_user_recos_df = user_top_programs_df. \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(final_user_recos_df, self.write_options, table_name, write_mode)


if __name__ == "__main__":
    job = ReplayRecoAlternateJob()
    job.launch()
