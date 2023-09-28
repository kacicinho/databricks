import math
import os
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import Row
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.databricks_to_s3 import DatabricksToS3
from databricks_jobs.jobs.utils.log_db import LogDB
from databricks_jobs.jobs.utils.onboarding_utils import get_watch_consent_df, prepare_lookup_channel_csv
from databricks_jobs.jobs.utils.utils import load_snowflake_table, write_df_to_snowflake, get_fresh_fa_unpivoted, \
    dump_dataframe_to_csv, load_snowflake_query_df


class PanelJob(Job):
    TIME_GRANULARITY = 60
    MIN_USER_PER_SEGMENT = 1000
    BANNED_AFFINITIES = ["Beauté & Bien-être", "Web & Gaming", "Horreur", "Santé & Médecine",
                         "Éducation", "Westerns", "Courts Métrages", "Cinéma"]

    def __init__(self, *args, **kwargs):
        super(PanelJob, self).__init__(*args, **kwargs)
        self.client = "realytics"

        # The .date ensure we round correctly to the current day
        self.now = self.parse_date_args()
        self.delta = timedelta(days=1)

        self.snowflake_user = self.conf.get("user_name", "")
        self.snowflake_password = self.conf.get("password", "")
        self.env = self.conf.get("env", "DEV")
        self.aws_access_key = self.conf.get("aws_access_key", "")
        self.aws_secret_key = self.conf.get("aws_secret_key", "")
        self.output_bucket = self.conf.get("s3_output_bucket", "")
        self.mount_name = "/mnt/s3-panel-exports"
        self.dump_folder = "/exports"
        self.dbutils = self.get_dbutils()

        self.options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflake_user,
            "sfPassword": self.snowflake_password,
            "sfDatabase": self.env,
            "sfSchema": "public",
            "sfWarehouse": "DATABRICKS_XS_WH"
        }

        self.write_options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflake_user,
            "sfPassword": self.snowflake_password,
            "sfDatabase": self.env,
            "sfSchema": "ML",
            "sfWarehouse": "DATABRICKS_XS_WH"
        }

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :
            Get user information on the user base : affinity, RFM28, gender, age
            Build user presence : +1/-1 when users arrive and depart from a channel
            Cumulative sum : get something like (t1, 30 viewers), (t2, 40 viewers)...
            export to snowflake and s3
        """
        self.logger.info("Launching panel job")
        enriched_user_df = self.build_user_info_base()
        user_presence_df = self.build_user_presence_df()
        cumsum_over_affinities_df = self.build_cumsum_over_segments(enriched_user_df, user_presence_df).persist()
        self.logger.info("Writing to table 'ML.PANEL'")
        write_df_to_snowflake(cumsum_over_affinities_df, self.write_options, "PANEL", "append")
        self.export_to_s3(cumsum_over_affinities_df)

    def build_affinity_df(self):
        """
        We want to export affinities from user, while getting rid of non trustworthy affinities

        Here is an example of affinity count found on the whole base.
        A threshold of 100K users might be a good direction in order to avoid very small samples
        We can also manually ban the affinites, it will make the query easier

        Beauté & Bien-être  :  51
        Web & Gaming  :  5700
        Horreur  :  30571
        Santé & Médecine  :  66399
        Éducation  :  70511
        Westerns  :  73026
        Courts Métrages  :  73727
        Cinéma  :  95330
        Famille  :  105451
        Comics & Animation  :  153176
        Économie  :  210396
        Musique  :  247033

        :return:
        """
        return get_fresh_fa_unpivoted(self.spark, self.options). \
            withColumnRenamed("category", "affinity"). \
            where(~F.col("affinity").isin(*self.BANNED_AFFINITIES))

    def filter_nightly_users(self):
        """
        We want to get rid of the users who watch too much at night

        around 43.6k on 21/06
        """
        query = \
            """
            with cte_nb_days_presence as (
              select USER_ID, count(distinct date_day) as nb_days
              from dw.fact_watch
              where date_day > current_date - 32 and date_day <= current_date - 2 and 
              HOUR(REAL_START_AT) >= 0 and HOUR(REAL_START_AT) < 4 and asset_type = 'live' and
              duration > 3 * 60
              group by USER_ID
            )
            select USER_ID
            from cte_nb_days_presence
            where nb_days > 5
            """
        return load_snowflake_query_df(self.spark, self.options, query)

    def build_user_info_base(self):
        """
        We mix user_raw db with fact_affinity to have the segment associated with each user along its socio-demo infos
        Only 'super_actifs' users are kept

        A weight column is added to combine
        """
        user_df = load_snowflake_table(self.spark, self.options, "backend.user_raw")
        panel_weights_df = load_snowflake_table(self.spark, self.options, "external_sources.users_panel_weight")
        affinity_df = self.build_affinity_df()
        nightly_user_df = self.filter_nightly_users()

        return user_df. \
            join(affinity_df, affinity_df.USER_ID == user_df.ID). \
            drop(affinity_df.USER_ID). \
            join(nightly_user_df, nightly_user_df.USER_ID == user_df.ID, "leftanti"). \
            drop(nightly_user_df.USER_ID). \
            join(panel_weights_df, panel_weights_df.USER_ID == user_df.ID). \
            withColumnRenamed("W", "weight"). \
            withColumnRenamed("PANEL_TYPE", "category"). \
            withColumn("max_pop_per_cat",
                       F.sum("weight").over(Window.partitionBy('category', 'affinity'))). \
            where(F.col("max_pop_per_cat") >= self.MIN_USER_PER_SEGMENT). \
            select("ID", "category", "affinity", "max_pop_per_cat", "weight")

    def build_user_presence_df(self):
        """
        A watch = Begin + end
        We do 1 watch => 1 event begin (= +1 user on channel at T) + 1 event  end (-1 user on channel at T)
        """
        # Events are received on watch_stopped, in order to receive events received between 23:00 and 1:00
        # we need to include watch received at 1:00 so at self.now and not self.now - 1
        fact_watch_df = get_watch_consent_df(self.spark, self.options,
                                             (self.now - timedelta(days=1)), self.now, with_consent=False)

        # Time filtering condition :
        #  - A watch started in [now - 2d, now] and ended on the timeframe [now - 1d, now]
        #  - A watch started in [now - 1d, now]
        time_condition = \
            (((F.col("WATCH_START") >= self.now - 2 * self.delta) & (F.col("WATCH_END") < self.now)) |
             ((F.col("WATCH_START") >= self.now - self.delta) & (F.col("WATCH_START") < self.now)))

        def build_date_ceiler(granularity):
            def date_ceiler(dt):
                d = math.ceil(dt.timestamp() / granularity) * granularity
                return datetime.fromtimestamp(d)

            return date_ceiler

        date_ceiler = build_date_ceiler(self.TIME_GRANULARITY)

        return fact_watch_df. \
            where(time_condition). \
            rdd. \
            flatMap(lambda x: [Row(int(x.USER_ID), int(x.CHANNEL_ID), date_ceiler(x.WATCH_START), 1),
                               Row(int(x.USER_ID), int(x.CHANNEL_ID), date_ceiler(x.WATCH_END), -1)]). \
            toDF(["USER_ID", "CHANNEL_ID", "timestamp", "watch"])

    def build_cumsum_over_segments(self, user_enriched_df, user_presence_df):
        """
        We have several steps :
        - Join data
        - Pre aggregation to reduce identical timestamps increments
        - Cumulative sum transformation
        """
        enriched_presence_df = user_enriched_df. \
            join(user_presence_df, user_presence_df.USER_ID == user_enriched_df.ID). \
            drop(user_presence_df.USER_ID). \
            drop(user_enriched_df.ID). \
            groupBy('category', 'affinity', "CHANNEL_ID", "timestamp"). \
            agg(F.sum(F.col("watch") * F.col("weight")).alias("weighted_watch"),
                F.max("max_pop_per_cat").alias("max_pop_per_cat"))

        def cumsum_over_cat():
            window_val = (Window.partitionBy('category', 'affinity', "CHANNEL_ID").orderBy('timestamp').
                          rangeBetween(Window.unboundedPreceding, 0))
            return F.sum('weighted_watch').over(window_val)

        return enriched_presence_df. \
            withColumn("total_watchers", F.round(cumsum_over_cat(), 0)). \
            withColumn("panel_percentage",
                       F.round(F.lit(100) * F.col("total_watchers") / F.col("max_pop_per_cat"), 3)). \
            select('category', 'affinity', "CHANNEL_ID", "timestamp", "total_watchers", "panel_percentage",
                   F.round(F.col("max_pop_per_cat"), 0).alias("total_users_in_group")). \
            withColumn("date_day", F.lit(self.now)). \
            where((F.col("timestamp") >= self.now - self.delta) & (F.col("timestamp") < self.now))

    def export_to_s3(self, cumsum_over_affinities_df):
        """
        Export the final dataframe to s3 with the usual log format
        :param cumsum_over_affinities_df:
        :return:
        """
        with DatabricksToS3(self.dbutils, self.mount_name,
                            self.aws_access_key, self.aws_secret_key, self.output_bucket) as uploader:
            # Export the panel data for the day
            csv_dir_path = os.path.join(self.dump_folder, "affinity_panel")
            temporary_csv_path = dump_dataframe_to_csv(self.dbutils, cumsum_over_affinities_df, csv_dir_path)
            uploader.dump_daily_table(temporary_csv_path, self.client, "panel", self.now)

            # We expect that we need to import the channel id lookup only once
            path = "lookup_feeds"
            lookup_feed_elems = LogDB.parse_dbfs_path(self.dbutils,
                                                      f"{self.mount_name}/{self.client}/{path}")
            if len(lookup_feed_elems.db) == 0:
                lookup_channel_csv_path = prepare_lookup_channel_csv(self.spark, self.options,
                                                                     self.dump_folder, self.dbutils)
                uploader.dump_daily_table(lookup_channel_csv_path, self.client, "lookup_feeds",
                                          self.now, identifier="channels")


if __name__ == "__main__":
    job = PanelJob()
    job.launch()
