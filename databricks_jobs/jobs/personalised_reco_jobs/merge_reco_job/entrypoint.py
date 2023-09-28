from pyspark.sql import functions as F
from py4j.protocol import Py4JJavaError

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake, \
    load_snowflake_query_df
from databricks_jobs.jobs.utils.merge_reco_utils import ab_test_conf_df_schema, MergeInfos, QueryInfos
from databricks_jobs.jobs.utils.utils import create_spark_df_from_data


class MergeRecoJob(Job):

    def __init__(self, *args, **kwargs):
        super(MergeRecoJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.table_infos = self.parse_table_infos_args()

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML")

    @staticmethod
    def add_more_args(p):
        """
        Additional args to define the merge operation
        """
        p.add_argument("--merge_reco_table", type=str)
        p.add_argument("--latest_reco_table", type=str)
        p.add_argument("--key_name", type=str)

    def parse_table_infos_args(self):
        return MergeInfos(merge_reco_table=self.parsed_args.merge_reco_table,
                          latest_reco_table=self.parsed_args.latest_reco_table,
                          key_name=self.parsed_args.key_name.upper())

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :

        """
        self.logger.info("Launching merge reco job")

        # Load tables
        qinfos = QueryInfos.from_key_name(self.table_infos.key_name)
        # Load id table and add fake users
        raw_df = load_snowflake_query_df(self.spark, self.options, qinfos.query).\
            union(create_spark_df_from_data(self.spark, {self.table_infos.key_name.upper(): range(-5, 1)}))

        try:
            split_ab_test_df = load_snowflake_table(self.spark, self.options, qinfos.split_table_name)
        except Py4JJavaError as e:
            if "doesn't exist" or "does not exist" in str(e.java_exception):
                split_ab_test_df = self.spark.createDataFrame([], qinfos.split_table_schema)
            else:
                raise Exception(e)

        reco_df = load_snowflake_table(self.spark, self.options, f"ml.{self.table_infos.merge_reco_table}"). \
            filter(F.col("UPDATE_DATE") == self.now)

        ab_test_conf = self.get_ab_test_conf()

        # If running AB test, we do the split 
        if ab_test_conf:
            # Update new split users for a specific ab_test_id
            new_split_users = self.build_new_split(raw_df, split_ab_test_df, ab_test_conf[0])

            write_df_to_snowflake(new_split_users, self.write_options, qinfos.split_table_name, "append")

            all_split_users = load_snowflake_table(self.spark, self.options, qinfos.split_table_name)

            # Merge reco
            reco_latest_table = self.merge_latest_reco(all_split_users, reco_df)

        # AB test is finished/ no ab_test then we fallback on DEFAULT RECO TABLE
        else:
            reco_latest_table = reco_df. \
                drop("VARIATIONS")

        # Write latest reco to snwoflake
        write_df_to_snowflake(reco_latest_table, self.write_options, self.table_infos.latest_reco_table, "append")

        # Useful for tests
        return reco_latest_table

    def get_ab_test_conf(self):

        try:
            ab_test_conf_df = load_snowflake_table(self.spark, self.options, "dw.ab_test_conf")
        except Py4JJavaError as e:
            if "doesn't exist" or "does not exist" in str(e.java_exception):
                ab_test_conf_df = self.spark.createDataFrame([], ab_test_conf_df_schema)
            else:
                raise Exception(e)

        ab_test_conf = ab_test_conf_df. \
            filter((self.now >= F.col("START_AT")) & 
                   ((self.now <= F.col("END_AT")) | (F.col("END_AT").isNull())) &
                   (F.col("MERGE_RECO_TABLE") == self.table_infos.merge_reco_table)). \
            collect()

        if len(ab_test_conf) > 1:
            raise Exception('Too many ab test found running at the same time.')
        else:
            return ab_test_conf

    def build_new_split(self, raw_df, split_ab_test_df, ab_test_conf):
        """
        Adds new id to split_ab_test_df. If split_ab_test_df is empty, it will add all new ids until now.

        raw_df: self.table_infos.key_name
        split_ab_test_df: AB_TEST_ID, {{ self.table_infos.key_name }}, VARIATIONS

        return:
            pyspark df: AB_TEST_ID, {{ self.table_infos.key_name }}, VARIATIONS

        """

        default_proportion = 1.0 - ab_test_conf.ALT_PROPORTION

        new_ids_df = raw_df. \
            join(split_ab_test_df.filter(F.col("AB_TEST_ID") == ab_test_conf.AB_TEST_ID),
                 on=[self.table_infos.key_name], how='leftanti').persist()

        # Passthrough ids correspond to fake users for thisWeek and topReplays
        # They cannot be in the B split of the A/B test
        pass_through_ids = new_ids_df. \
            where(f"{self.table_infos.key_name} <= 0")
        new_ids_df = new_ids_df. \
            where(f"{self.table_infos.key_name} > 0")

        a, b = new_ids_df. \
            select(F.lit(ab_test_conf.AB_TEST_ID).alias("AB_TEST_ID"), self.table_infos.key_name). \
            randomSplit([default_proportion, ab_test_conf.ALT_PROPORTION], seed=ab_test_conf.AB_TEST_ID)

        # We reunite the passthrough ids in the A split
        a = pass_through_ids. \
            select(F.lit(ab_test_conf.AB_TEST_ID).alias("AB_TEST_ID"), self.table_infos.key_name). \
            union(a)

        a = a.withColumn("VARIATIONS", F.lit('A'))
        b = b.withColumn("VARIATIONS", F.lit('B'))

        return a.union(b)

    def merge_latest_reco(self, all_split_users, reco_df):
        """
        Merge reco from the two variations based on the all_split_users_df
        all_split_users: AB_TEST_ID, {{ self.table_infos.key_name }}, VARIATIONS
        reco_df: {{ self.table_infos.key_name }}, RECOMMENDATIONS, UPDATE_DATE, VARIATIONS
        """

        reco_a = reco_df.join(all_split_users.filter(F.col("VARIATIONS") == 'A'), on=[self.table_infos.key_name, "VARIATIONS"])
        reco_b = reco_df.join(all_split_users.filter(F.col("VARIATIONS") == 'B'), on=[self.table_infos.key_name, "VARIATIONS"])

        return reco_a.union(reco_b). \
            drop("VARIATIONS", "AB_TEST_ID")


if __name__ == "__main__":
    job = MergeRecoJob()
    job.launch()
