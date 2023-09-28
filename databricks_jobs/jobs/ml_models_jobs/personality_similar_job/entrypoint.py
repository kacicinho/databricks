from datetime import timedelta
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks_jobs.common import Job
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake


class PersonalitySimilarityJob(Job):
    BAN_PERSON_ID = [169046]

    def __init__(self, *args, **kwargs):
        super(PersonalitySimilarityJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=1000)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        episode_df = build_episode_df(self.spark, self.options)
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False)

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True)

        prog_df = broadcast_df.union(vod_df)

        program_df = load_snowflake_table(self.spark, self.options, "BACKEND.PROGRAM")
        person_df = load_snowflake_table(self.spark, self.options, "BACKEND.PERSON")
        rel_episode_person_df = load_snowflake_table(self.spark, self.options, "BACKEND.REL_EPISODE_PERSON")
        ref_program_kind = load_snowflake_table(self.spark, self.options, "BACKEND.REF_PROGRAM_KIND")
        ref_program_category = load_snowflake_table(self.spark, self.options, "BACKEND.REF_PROGRAM_CATEGORY")
        ref_person_function = load_snowflake_table(self.spark, self.options, "BACKEND.REF_PERSON_FUNCTION")

        # Select available personalities filtering person with associated picture
        available_person_df = prog_df. \
            join(rel_episode_person_df, "EPISODE_ID"). \
            join(person_df, person_df.ID == rel_episode_person_df.PERSON_ID). \
            where(~F.col("PERSON_ID").isin(*self.BAN_PERSON_ID)). \
            where(F.col("PICTURE_HASH").isNotNull()). \
            select("PERSON_ID"). \
            dropDuplicates()

        categories = ref_program_category.select("NAME").toPandas()["NAME"].unique()
        kind = ref_program_kind.select("NAME").toPandas()["NAME"].unique()
        function = ref_person_function.select("NAME").toPandas()["NAME"].unique()

        # Collect information for all personalities programs (category, kind, function, production year)
        person_info_df = available_person_df. \
            join(rel_episode_person_df.alias("rel"), "PERSON_ID"). \
            join(episode_df, "EPISODE_ID"). \
            join(program_df, episode_df.PROGRAM_ID == program_df.ID). \
            join(ref_program_kind, ref_program_kind.ID == program_df.REF_PROGRAM_KIND_ID). \
            withColumnRenamed("NAME", "KIND"). \
            join(ref_program_category, ref_program_category.ID == program_df.REF_PROGRAM_CATEGORY_ID). \
            withColumnRenamed("NAME", "CATEGORY"). \
            join(ref_person_function.alias("ref"), F.col("ref.ID") == F.col("rel.REF_PERSON_FUNCTION_ID")). \
            withColumnRenamed("NAME", "FUNCTION"). \
            withColumn("nb_program", F.count("EPISODE_ID").over(Window().partitionBy("PERSON_ID"))). \
            where(F.col("nb_program") >= 10). \
            select("PROGRAM_ID", "EPISODE_ID", "PERSON_ID", "CATEGORY", "KIND", program_df.PRODUCTION_YEAR, "FUNCTION",
                   "ORDER"). \
            dropDuplicates(). \
            withColumn("PRODUCTION_YEAR", F.floor(F.col("PRODUCTION_YEAR") / 10) * 10). \
            withColumn("ORDER", F.floor(F.col("ORDER") / 10) * 10)

        # Compute normalised vector of categories for each personality
        person_info_program_df = person_info_df. \
            select("PERSON_ID", "CATEGORY", "EPISODE_ID"). \
            groupBy("PERSON_ID", "CATEGORY"). \
            agg(F.count("EPISODE_ID").alias("count")). \
            withColumn("normalise",
                       F.round(F.col("count") / F.sum("count").over(Window().partitionBy("PERSON_ID")), scale=4)). \
            groupBy("PERSON_ID"). \
            pivot("CATEGORY", categories). \
            sum("normalise"). \
            fillna(0). \
            select("PERSON_ID", *[F.col(c).alias(f'CATEGORY_{c}') for c in categories])

        # Compute normalised vector of kind for each personality
        person_info_kind_df = person_info_df. \
            select("PERSON_ID", "KIND", "EPISODE_ID"). \
            groupBy("PERSON_ID", "KIND"). \
            agg(F.count("EPISODE_ID").alias("count")). \
            withColumn("normalise",
                       F.round(F.col("count") / F.sum("count").over(Window().partitionBy("PERSON_ID")), scale=4)). \
            groupBy("PERSON_ID"). \
            pivot("KIND", kind). \
            sum("normalise"). \
            fillna(0). \
            select("PERSON_ID", *[F.col("{}".format(k)).alias('KIND_{}'.format(k)) for k in kind])

        # Compute normalised vector of production years for each personality
        person_info_year_df = person_info_df. \
            select("PERSON_ID", "PRODUCTION_YEAR", "EPISODE_ID"). \
            groupBy("PERSON_ID", "PRODUCTION_YEAR"). \
            agg(F.count("EPISODE_ID").alias("count")). \
            withColumn("normalise",
                       F.round(F.col("count") / F.sum("count").over(Window().partitionBy("PERSON_ID")), scale=4)). \
            groupBy("PERSON_ID"). \
            pivot("PRODUCTION_YEAR", [i for i in range(1900, 2030, 10)]). \
            sum("normalise"). \
            fillna(0)

        # Compute normalised vector of functions for each personality
        person_info_function_df = person_info_df. \
            select("PERSON_ID", "FUNCTION", "EPISODE_ID"). \
            groupBy("PERSON_ID", "FUNCTION"). \
            agg(F.count("EPISODE_ID").alias("count")). \
            withColumn("normalise",
                       F.round(F.col("count") / F.sum("count").over(Window().partitionBy("PERSON_ID")), scale=4)). \
            groupBy("PERSON_ID"). \
            pivot("FUNCTION", function). \
            sum("normalise"). \
            fillna(0)

        # Gather informations in a unique vector
        person_info_all_df = person_info_program_df. \
            join(person_info_kind_df, "PERSON_ID", "left"). \
            join(person_info_year_df, "PERSON_ID", "left"). \
            join(person_info_function_df, "PERSON_ID", "left"). \
            fillna(0)

        write_df_to_snowflake(person_info_all_df, self.write_options, "PERSON_VECTORISATION_FEATURES", "overwrite")


if __name__ == "__main__":
    job = PersonalitySimilarityJob()
    job.launch()
