from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
from pyspark.sql.types import StructType, StructField, IntegerType, DateType, StringType
from databricks_jobs.jobs.utils.utils import load_snowflake_table
import json
import pyspark.sql.functions as F


class SoreImdbKeywordsJob(Job):
    REF_KEYWORDS_TABLE = "ML.REF_IMDB_KEYWORDS"
    SCRAPED_KEYWORDS_TABLE = "ML.PROGRAMS_WITH_SCRAPED_KEYWORDS"

    def __init__(self, *args, **kwargs):
        super(SoreImdbKeywordsJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1 load already stored keywords
        existing_keywords_df = load_snowflake_table(self.spark, self.options, self.REF_KEYWORDS_TABLE)
        already_stored_keywords_list = existing_keywords_df.agg(F.collect_list("KEYWORD")).first()[0]

        # cast to int because spark give "Decimal" class
        max_existing_id = int(existing_keywords_df.agg(F.max("ID")).first()[0])

        # 2 load all scraped keywords
        scraped_imdb_keywords_df = load_snowflake_table(self.spark, self.options, self.SCRAPED_KEYWORDS_TABLE)
        scraped_stringed_keywords_list = scraped_imdb_keywords_df.agg(F.collect_list(F.col("IMDB_KW"))).first()[0]

        """ 
        * databricks can't cast SQL arrays to ArrayType(StringType())
        * SQL array is converted to StringType, for example:  
            '[\n  "sick-dog",\n  "baby-in-a-stroller",\n  "nacho"]'
        that I later call "stringed_list"
        * json.loads allows to decode those "stringed_list" in list of strings, for example :
            ["sick-dog","baby-in-a-stroller","nacho"]
        * When using collect_list, the result is a list of "stringed-list", for example :
            [
                '[\n  "sick-dog",\n  "baby-in-a-stroller",\n  "swordfish",\n  "nacho"]',
                '[\n  "1990s",\n  "french-teacher",\n  "country-vs-city-cultures"]',
                '[\n  "tied-feet",\n  "tape-over-mouth"\n]'
            ]
        """
        list_of_list_keywords = [json.loads(kw_l) for kw_l in scraped_stringed_keywords_list]
        """
        at this stage, we have a list of list of strings, for example :
            [
                ["sick-dog","baby-in-a-stroller","swordfish","nacho"],
                ["1990s","french-teacher","country-vs-city-cultures"],
                ["tied-feet","tape-over-mouth"]
            ]
        """
        flattened_list_of_keywords = [kw for lst in list_of_list_keywords for kw in lst]

        # 3 keep only non already stored keywords
        to_be_written_keywords = set(flattened_list_of_keywords) - set(already_stored_keywords_list)
        nb_keywords_to_be_written = len(to_be_written_keywords)

        # 4 create the dataframe that will be written to REF_KEYWORDS_TABLE
        index = list(range(1 + max_existing_id, 1 + nb_keywords_to_be_written + max_existing_id))
        date = nb_keywords_to_be_written * [self.now]

        schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("KEYWORD", StringType(), True),
            StructField("UPDATE_AT", DateType(), True)

        ])

        data = list(zip(index, to_be_written_keywords, date))
        keywords_spark_df = self.spark.createDataFrame(data=data, schema=schema)

        # 5 write dataframe to db
        write_df_to_snowflake(keywords_spark_df, self.write_options,
                              table_name=self.REF_KEYWORDS_TABLE,
                              mode="append")


if __name__ == "__main__":
    job = SoreImdbKeywordsJob()
    job.launch()
