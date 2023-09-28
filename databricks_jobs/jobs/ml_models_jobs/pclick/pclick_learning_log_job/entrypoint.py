import random
from datetime import timedelta

from functools import partial

from pyspark import Row
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from databricks_jobs.common import Job
from databricks_jobs.db_common import build_broadcast_df_with_episode_info, build_episode_df, build_channel_order_df
from databricks_jobs.jobs.utils.date_utils import build_datetime_ceiler, build_timestamp_ceiler
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake, load_snowflake_table


class InvalidRowException(Exception):
    pass


class PClickLearningLogJob(Job):
    """
    This job computes a table that will be used for the pClick model, it computes the following tables :

    1 - The full_epg of the day

    +-------------------+--------------------+
    |          TIMESTAMP|         CURRENT_EPG|
    +-------------------+--------------------+
    |2021-08-11 13:00:00|[{5, 3}, {1, 1}, ...|
    |2021-08-11 13:05:00|[{5, 3}, {1, 1}, ...|
    |2021-08-11 13:10:00|[{5, 3}, {1, 1}, ...|
    |2021-08-11 13:15:00|[{5, 3}, {1, 1}, ...|
    |2021-08-11 13:20:00|[{5, 3}, {1, 1}, ...|

    It can be read as : at 2021-08-11 13:00:00, at position 1, we have program_1, at position 3 we have program_5

    2 - The user clicks on the on-tv rail

    +-------------------+-------+----------+----------+
    |          TIMESTAMP|USER_ID|PROGRAM_ID|CLICK_RANK|
    +-------------------+-------+----------+----------+
    |2021-08-11 14:35:00|      1|         1|         1|
    |2021-08-11 14:35:00|      2|         1|         1|
    |2021-08-11 14:35:00|      3|         5|         3|
    +-------------------+-------+----------+----------+
    It is collected from fact_page

    3- The pCLick training log

    +-------------------+-------+----+----------+-----+----------+
    |          TIMESTAMP|USER_ID|RANK|PROGRAM_ID|LABEL|  DATE_DAY|
    +-------------------+-------+----+----------+-----+----------+
    |2021-08-11 14:35:00|      1|   0|         1|    1|2021-08-09|
    |2021-08-11 14:35:00|      1|   1|         4|    0|2021-08-09|
    |2021-08-11 14:35:00|      2|   0|         1|    1|2021-08-09|
    |2021-08-11 14:35:00|      2|   2|         5|    0|2021-08-09|
    |2021-08-11 14:35:00|      3|   2|         5|    1|2021-08-09|
    |2021-08-11 14:35:00|      3|   0|         2|    0|2021-08-09|
    +-------------------+-------+----+----------+-----+----------+
    We generate
    - the LABEL=1 data with the click dataset defined before
    - the LABEL=0 data by finding program that the user could click but didn't

    """

    TIME_SAMPLING_FACTOR = 5 * 60
    LOG_SAMPLING_RATIO = 0.2
    N_NEGATIVE_SAMPLING = 2
    RANDOM_NEGATIVE_SAMPLING_RATIO = 0.1

    def __init__(self, *args, **kwargs):
        super(PClickLearningLogJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args() - timedelta(days=2)
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", sf_schema="PUBLIC")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Steps
        1 - Get the full_epg_df
        2 - Get the click_df from fact_page
        3 - Combine them to get the learning log
        """
        episode_info_df = build_episode_df(self.spark, self.options)
        complete_broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                                     self.now - timedelta(days=1),
                                                                     self.now + timedelta(days=1),
                                                                     free_bundle=True, filter_kind=False,
                                                                     min_duration_in_mins=0)

        full_epg_df = self.build_full_epg_table(complete_broadcast_df, self.TIME_SAMPLING_FACTOR)
        click_df = self.get_click_dataset()
        premium_user_df = load_snowflake_table(self.spark, self.options, "TEMP.FACT_USER_SUBSCRIBERS"). \
            select("USER_ID", "STATUS_DAY_DATE"). \
            distinct(). \
            withColumn("is_premium", F.lit(True))

        out_df = self.combine_and_sample(click_df, full_epg_df, premium_user_df, self.N_NEGATIVE_SAMPLING)
        write_df_to_snowflake(out_df, self.write_options, "ML.PCLICK_TRAINING_LOG", "append")

    def get_click_dataset(self):
        """
        In this function, we need to process from fact_page the properties_json to find :
        - origin_component_rank
        - program_id

        Another odd pre-processing is the timestamp rounding in order to be able to match this df with the epg one.
        """
        click_df = load_snowflake_table(self.spark, self.options, "DW.fact_page")

        timestamp_round = F.udf(build_datetime_ceiler(self.TIME_SAMPLING_FACTOR), TimestampType())
        # Min of rank is 1 !
        parsed_click_df = click_df. \
            where(F.col("ORIGIN_SECTION").isin(["on-tv"])). \
            where(F.col("PAGE_NAME") == "program"). \
            where(F.col("EVENT_DATE") == self.now). \
            select("USER_ID", "TIMESTAMP", "CHANNEL_ID", "PROGRAM_ID", "EVENT_DATE",
                   F.col("ORIGIN_COMPONENT_RANK").alias("CLICK_RANK")). \
            withColumn("TIMESTAMP", timestamp_round("TIMESTAMP")). \
            where(F.col("PROGRAM_ID").isNotNull())

        return parsed_click_df

    def build_full_epg_table(self, complete_broadcast_df, time_sampling_factor):
        """
        For any round to the X minutes timestamp, we have
        - all available programs on the free bundle
        - with tnt rank rather than channel

        +-------------------+--------------------+
        |          TIMESTAMP|         CURRENT_EPG|
        +-------------------+--------------------+
        |2021-08-09 12:35:00|[{1, 1}, {5, 3}, ...|
        |2021-08-09 12:40:00|[{1, 1}, {5, 3}, ...|
        |2021-08-09 12:45:00|[{1, 1}, {5, 3}, ...|
        |2021-08-09 12:50:00|[{1, 1}, {5, 3}, ...|
        """
        channel_order = build_channel_order_df(self.spark, self.options)

        dc = build_timestamp_ceiler(time_sampling_factor)
        full_epg = complete_broadcast_df. \
            join(channel_order, complete_broadcast_df.CHANNEL_ID == channel_order.ID). \
            rdd.flatMap(
                lambda x: [Row(int(t), int(x.CHANNEL_ID), int(x.RAIL_ORDER), int(x.PROGRAM_ID))
                           for t in range(dc(x.START_AT), dc(x.END_AT), time_sampling_factor)]). \
            toDF(["TIMESTAMP", "CHANNEL_ID", "RAIL_ORDER", "PROGRAM_ID"]). \
            groupBy("TIMESTAMP"). \
            agg(F.collect_set(F.struct("PROGRAM_ID", "RAIL_ORDER", "CHANNEL_ID")).alias("CURRENT_EPG")). \
            withColumn("TIMESTAMP", F.to_timestamp("TIMESTAMP"))

        return full_epg

    def combine_and_sample(self, click_df, epg_df, premium_user_df, n_negatives=1, ratio_neg=0.1):
        """
        We do the following processes :
        - Keep the positive clicks from users : CLICK_RANK is 1-indexed, so we store CLICK_RANK - 1
        - Find a negative (as non clicked) given the epg at that time

        This is done through a flapMap, this will allow to increase the negative ratio if wanted.
        """
        premium_tnt_bundle = list(range(1, 26))
        free_tnt_bundle_positions = [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 24, 25]

        def find_negative(row, n_negative=2, ratio_negative=0.1):
            """
            How do we do negative sampling :

            - We sample two negatives (N_NEGATIVE_SAMPLING) for one positive
            - if click is in top 5, we select negative among the top five
            - Else we sample among the ranks < click_rank
            - premium and freemium have a different carousel, we also take that into account

            Sometimes we may encounter incorrect data, so we may abort the processing

            CLICK_RANK and RAIL_ORDER are 1-indexed
            """
            def find_negative_element(row, *args, **kwargs):
                """
                This function is a loop over the sample function
                as we may not always find a negative element in the epg
                """
                cli_rank = row.CLICK_RANK

                def sample(is_free):
                    rank = min(int(cli_rank), 25)
                    if not is_free and rank <= 6:
                        positions = set(premium_tnt_bundle[:6]).difference({rank})
                    elif is_free and rank <= 11:
                        positions = set(free_tnt_bundle_positions[:6]).difference({rank})
                    elif random.random() >= ratio_negative:  # Sample negative with rank < click_rank
                        positions = range(1, rank) if not is_free else \
                            filter(lambda x: x < rank, free_tnt_bundle_positions)
                    else:
                        positions = range(1, 25) if not is_free else free_tnt_bundle_positions
                    return random.sample(list(positions), 1)[0]

                neg_rank = sample(not row.is_premium)
                elems = [r for r in row.CURRENT_EPG if int(r.RAIL_ORDER) == int(neg_rank)]

                try_cnt = 0
                while len(elems) == 0 and try_cnt < 5:
                    neg_rank = sample(not row.is_premium)
                    elems = [r for r in row.CURRENT_EPG if int(r.RAIL_ORDER) == int(neg_rank)]

                if len(elems) == 0:
                    raise InvalidRowException

                return elems[0], neg_rank

            click_rank = min(int(row.CLICK_RANK), 25)
            pos = Row(row.TIMESTAMP, int(row.USER_ID), int(row.CHANNEL_ID),
                      click_rank - 1, int(row.PROGRAM_ID), 1)
            try:
                negs = [Row(row.TIMESTAMP, int(row.USER_ID), int(epg_element.CHANNEL_ID),
                            neg_rank - 1, int(epg_element.PROGRAM_ID), 0)
                        for epg_element, neg_rank in map(partial(find_negative_element, row),
                                                         range(n_negative))]
            except InvalidRowException:
                # Something is wrong, we abort everything
                return []

            return negs + [pos]

        return click_df. \
            join(premium_user_df, (premium_user_df.USER_ID == click_df.USER_ID) &
                 (premium_user_df.STATUS_DAY_DATE == click_df.EVENT_DATE), "left"). \
            drop(premium_user_df.USER_ID). \
            drop(premium_user_df.STATUS_DAY_DATE). \
            fillna(False, subset=["is_premium"]). \
            where("CLICK_RANK <= 25"). \
            join(epg_df, click_df.TIMESTAMP == epg_df.TIMESTAMP). \
            drop(epg_df.TIMESTAMP). \
            where(F.col("CURRENT_EPG").isNotNull() & F.col("CLICK_RANK").isNotNull()). \
            dropna(how='any'). \
            sample(self.LOG_SAMPLING_RATIO). \
            rdd. \
            flatMap(
                lambda x: find_negative(x, n_negatives, ratio_neg) if x.CURRENT_EPG is not None else []
            ).toDF(["TIMESTAMP", "USER_ID", "CHANNEL_ID", "RANK", "PROGRAM_ID", "LABEL"]). \
            withColumn("DATE_DAY", F.lit(self.now))


if __name__ == "__main__":
    job = PClickLearningLogJob()
    job.launch()
