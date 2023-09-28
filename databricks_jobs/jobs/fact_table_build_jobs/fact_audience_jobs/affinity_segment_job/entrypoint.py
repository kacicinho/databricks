import datetime

import pandas as pd
from pyspark.sql.functions import col, when, lit, struct, translate, pandas_udf, PandasUDFType, year, min, \
    datediff
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.affinity_lib import select_info_udf, unique_words_udf, \
    token_lemma_udf, count_subcategories_udf, get_spacy_model, load_affinities_df, aff_conditions
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df, write_df_to_snowflake, load_snowflake_table, \
    get_snowflake_options


class AffinitySegmentJob(Job):

    def __init__(self, *args, **kwargs):
        super(AffinitySegmentJob, self).__init__(*args, **kwargs)

        self.quantile_options = get_snowflake_options(self.conf, 'PROD', 'DW', sfWarehouse="PROD_WH")
        self.affinities_options = get_snowflake_options(self.conf, 'PROD', 'EXTERNAL_SOURCES',
                                                        keep_column_case="on", sfWarehouse="PROD_WH")
        self.user_affinity_options = \
            get_snowflake_options(self.conf, 'PROD', 'DW', keep_column_case="on", sfWarehouse="PROD_WH")

        self.users_options = get_snowflake_options(self.conf, 'PROD', 'BACKEND', sfWarehouse="PROD_WH")

        self.now = self.parse_date_args()
        self.lookback = datetime.timedelta(days=90)
        self.logger.info(f"Running on date : {self.now}")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        Main of the job :
            Creates a table with the best affinity score for each program_id.
            Creates 2 tables with the affinity score/ flag for each user for each
            affinity segment.
        """
        self.logger.info("Launching Affinity segment job")

        # 1. Daily watch time per type of show
        daily_watch_kind_df = self.daily_watch_kind_df_job()
        write_df_to_snowflake(daily_watch_kind_df, self.quantile_options, 'FACT_AUDIENCE_DAILY_USER_PROG_TYPE',
                              "append")

        # 2. Calculates quantile activity per type of show
        aff_df = self.audience_prog_type_affinity_df_job()
        write_df_to_snowflake(aff_df, self.quantile_options, 'FACT_AUDIENCE_PROG_TYPE_AFFINITY', 'overwrite')

        # 3. Prepare the program catalog
        program_df = self.program_df_job()

        # 4. Create the affinity segments
        affinities_df = self.affinities_df_job()

        # 5 Create affinity scores for each program
        grouped_df, grouped_df_subinfo = self.grouped_df_job(program_df, affinities_df)

        # 6. Keep only the best affinity scores per program
        result_df = self.result_info_job(grouped_df, grouped_df_subinfo)
        write_df_to_snowflake(result_df, self.affinities_options, "DAILY_PROG_AFFINITY", "append")

        # 7. Calculate affinity score for each users
        df_joined = self.fact_audience_aff_only_job(affinities_df)
        write_df_to_snowflake(df_joined, self.user_affinity_options, "FACT_AUDIENCE_AFF_ONLY", "append")

        # 8. Flag user affinities between 1 & 0
        global_agg_fact_affinity_df = self.fact_audience_aff_only_flag_job()
        write_df_to_snowflake(global_agg_fact_affinity_df, self.user_affinity_options,
                              "FACT_AUDIENCE_AFF_ONLY_FLAG", "append")

        fact_audience_aff_regie_df = self.fact_audience_aff_regie_job(global_agg_fact_affinity_df)
        write_df_to_snowflake(fact_audience_aff_regie_df, self.user_affinity_options,
                              "FACT_AUDIENCE_AFF_REGIE_FLAG", "overwrite")

    def daily_watch_kind_df_job(self):
        """
        We compute for each day the amount of watch time per type of shows among
         - Movie
         - Series
         - Broadcast (a mix of several kinds)
         - Documentaries
         Watch < 15 min are removed
        """

        daily_watch_kind_query = """
        select user_id, date_day,
                sum(CASE WHEN category_id=1 THEN duration else 0 END) as movie_duration,
                sum(CASE WHEN category_id=2 THEN duration else 0 END) as serie_duration,
                sum(CASE WHEN category_id=3 OR category_id=4 OR category_id=5 OR category_id=9 THEN duration else 0 END) as broadcast_duration,
                sum(CASE WHEN category_id=8 THEN duration else 0 END) as documentary_duration,
                sum(duration) as watch_duration
        from dw.fact_watch
        where date_day = '{date}'
        and DURATION > {min_duration_in_min} * 60
        group by user_id, date_day
        """.format(date=self.now - datetime.timedelta(days=1), min_duration_in_min=10)

        # Load query to spark df
        daily_watch_kind_df = load_snowflake_query_df(self.spark, self.affinities_options, daily_watch_kind_query)

        return daily_watch_kind_df

    def audience_prog_type_affinity_df_job(self):
        """
        Based on the table aggregating daily watch time on type of show per user,
        We compute quantiles of activity per type of show
        - A user with 0.1 means is among the top 90% top users
        - A user with 0.7 means is among the top 30% top users
        Watch time is computed on a 90 days window
        """
        start = self.now - self.lookback
        end = self.now
        quantile_query = \
            f"""
            with cte_duration as (select user_id,
                sum(movie_duration) as movie_duration,
                sum(serie_duration) as serie_duration,
                sum(broadcast_duration) as broadcast_duration,
                sum(documentary_duration) as documentary_duration
            from DW.FACT_AUDIENCE_DAILY_USER_PROG_TYPE
            where date_day >= '{start}' and date_day <= '{end}' 
            group by user_id),

            cte_movie_ntile as (select user_id,ntile(10) over(order by movie_duration desc) as movie_ntile
            from cte_duration
            where movie_duration <> 0),

            cte_serie_ntile as (select user_id,ntile(10) over(order by serie_duration desc) as serie_ntile
            from cte_duration
            where serie_duration <> 0),

            cte_broadcast_ntile as (select user_id,ntile(10) over(order by broadcast_duration desc) as broadcast_ntile
            from cte_duration
            where broadcast_duration <> 0),    
            cte_documentary_ntile as (select user_id,ntile(10) over(order by documentary_duration desc) as documentary_ntile
            from cte_duration
            where documentary_duration <> 0)

            select
            cte_duration.user_id,
            movie_duration, serie_duration, documentary_duration, broadcast_duration,
            ifnull(1.0 - 0.1 * movie_ntile, 0) as movie_ntile,
            ifnull(1.0 - 0.1 * serie_ntile, 0) as serie_ntile ,
            ifnull(1.0 - 0.1 * documentary_ntile, 0) as documentary_ntile,
            ifnull(1.0 - 0.1 * broadcast_ntile, 0) as broadcast_ntile
            from cte_duration
            left join cte_movie_ntile
            on cte_duration.user_id=cte_movie_ntile.user_id
            left join cte_serie_ntile
            on cte_duration.user_id=cte_serie_ntile.user_id
            left join cte_documentary_ntile
            on cte_duration.user_id=cte_documentary_ntile.user_id
            left join cte_broadcast_ntile
            on cte_duration.user_id=cte_broadcast_ntile.user_id
            """

        program_type_quantiles_df = load_snowflake_query_df(self.spark, self.quantile_options, quantile_query)

        def quantile_more_than_k(col_name, k):
            return when(col(col_name) >= k, 1).otherwise(0)

        def build_tag_dataframe(df, quantile, quantile_columns):
            """
            We create the tags for the final table which indicates if user are in the affinity movie/series/...
            """
            basic_col_names = [col_name.split("_")[0] for col_name in quantile_columns]
            for col_name, q_col_name in zip(basic_col_names, quantile_columns):
                df = df.withColumn(col_name, quantile_more_than_k(q_col_name, quantile))

            return df. \
                select("USER_ID", *basic_col_names). \
                withColumn("date_day", lit(datetime.datetime.now().date()))

        aff_df = build_tag_dataframe(program_type_quantiles_df, quantile=0.7,
                                     quantile_columns=["movie_ntile", "serie_ntile", "broadcast_ntile",
                                                       "documentary_ntile"])

        return aff_df

    def program_df_job(self):
        """
        Fetch the yesterday program catalog, clean it and apply nlp function to
        extract topics/information about the program.
        """
        yesterday = self.now - datetime.timedelta(days=1)
        query = f"""
            with already_tagged as (
              select
                  program_id
              from external_sources.DAILY_PROG_AFFINITY
            ),
            new_progs as (
                select distinct program_id
                from backend.vod
                join backend.episode
                on episode_id = backend.episode.id
                where available_from >= '{yesterday}'
                union (
                    select distinct program_id
                    from backend.broadcast
                    where start_at >= '{yesterday}'
                )
            )
            select
                distinct new_progs.program_id,
                         edito_program.title as program,
                         ref_program_category.name as category,
                         ref_program_kind.name as kind,
                         edito_program.summary
            from new_progs
            join backend.program
            on backend.program.id = new_progs.program_id
            join backend.ref_program_category
            on backend.program.ref_program_category_id=ref_program_category.id
            join backend.ref_program_kind
            on backend.program.ref_program_kind_id=ref_program_kind.id
            left join backend.edito_program
            on edito_program.program_id=new_progs.program_id
            where new_progs.program_id not in (
              select program_id from already_tagged
            )
        """

        program_df = load_snowflake_query_df(self.spark, self.affinities_options, query)

        # We drop duplicates program IDs
        program_df = program_df.dropDuplicates(["program_id"])
        # Optimisation on Decimal type to Integer
        program_df = program_df.withColumn("PROGRAM_ID", col("PROGRAM_ID").cast(IntegerType()))

        # We remove some of the programs which are not to be analysed
        deleted_kind = [
            "Bonus",
            "Météo",
            "Web TV",
            "Erotiques",
            "Fin de programme",
            "Indéterminé",
            "Pornographiques",
            "Portraits",
            "Religion",
        ]

        # Filter only on wanted kinds
        program_df = program_df. \
            filter(~program_df.KIND.isin(deleted_kind)). \
            filter(~program_df.CATEGORY.isin(deleted_kind))

        list_animated_movies = [
            "+ de 10 ans - Dessins animés",
            "+ de 3 ans - Dessins animés",
            "- de 2 ans - Dessins animés",
            "4 à 6 ans - Dessins animés",
            "6 à 10 ans - Dessins animés"
        ]

        list_films_age = [
            "+ de 10 ans - Films",
            "+ de 3 ans - Films",
            "- de 2 ans - Films",
            "4 à 6 ans - Films",
            "6 à 10 ans - Films"
        ]

        list_isin = [list_animated_movies, list_films_age, ["+ de 10 ans - Séries", "6 à 10 ans - Séries"],
                     ["3/4/5ème"], ["E-sport"]]
        list_cat = ["Enfants - Dessins animés", "Enfants - Films", "Enfants - Series", "Pour le Collège",
                    "Web & Gaming"]

        # Apply transformation on KIND & CATEGORY cols
        for list_in, cat in zip(list_isin, list_cat):
            for col_name in ["KIND", "CATEGORY"]:
                program_df = program_df. \
                    withColumn(col_name, when(program_df[col_name].isin(list_in), cat).otherwise(program_df[col_name]))

        # Apply nlp to extract infos about the program
        program_df = program_df \
            .withColumn("info",
                        select_info_udf(struct([program_df[x] for x in program_df.columns]), lit("info")))

        program_df = program_df \
            .withColumn("subinfo",
                        select_info_udf(struct([program_df[x] for x in program_df.columns]), lit("subinfo")))

        # Reformating string
        for item in ["info", "subinfo"]:
            for char in ["[", "]", "'", " ", ","]:
                if char == ',':
                    program_df = program_df.withColumn(item, translate(item, char, " "))
                else:
                    program_df = program_df.withColumn(item, translate(item, char, ""))

        # Add default value for score (used later)
        program_df = program_df.withColumn("score", lit(0))

        # Repartitioning df
        program_df = program_df.repartition(16)

        return program_df

    def affinities_df_job(self):
        """
        Clean the affinity file and create the affinity segment with nlp.
        """

        # Loading data
        affinities_df = load_affinities_df(self.spark)

        # Rename col names
        affinities_df = affinities_df.toDF(*(c.lower().replace(' ', '_') for c in affinities_df.columns))

        # Process data
        affinities_df = affinities_df \
            .withColumn("segment_name_tokens", unique_words_udf(affinities_df["segment_algorithm"]))
        affinities_df = affinities_df \
            .withColumn("segment_name_tokens", token_lemma_udf(affinities_df["segment_name_tokens"]))
        affinities_df = affinities_df \
            .withColumn("subcategories", count_subcategories_udf(affinities_df["segment_algorithm"]))

        # Filtering on subcategories
        affinities_df = affinities_df \
            .filter(affinities_df["subcategories"] < 3)
        affinities_df = affinities_df \
            .select("segment_name_tokens", "segment_algorithm", "segment_table", "profil", "subcategories")

        # Reformating string
        for char in ["[", "]", "'", " ", ","]:
            if char == ',':
                affinities_df = affinities_df.withColumn("segment_name_tokens",
                                                         translate("segment_name_tokens", char, " "))
            else:
                affinities_df = affinities_df.withColumn("segment_name_tokens",
                                                         translate("segment_name_tokens", char, ""))

        # Repartitioning df
        affinities_df = affinities_df.repartition(16)

        return affinities_df

    def grouped_df_job(self, program_df, affinities_df):
        """
        Cross Join the program catalog with the affinities and create affinity scores
        for each program
        """

        @pandas_udf("float", PandasUDFType.SCALAR)
        def score_tokens_func(a, b):
            nlp = get_spacy_model()
            return pd.Series([nlp(c1).similarity(nlp(c2)) if len(c1) > 0 and len(c2) > 0 else 0
                              for c1, c2 in zip(a, b)])

        # Cross join with affinities
        grouped_df = program_df. \
            crossJoin(affinities_df). \
            select("PROGRAM_ID",
                   "PROGRAM",
                   "CATEGORY",
                   "KIND",
                   "segment_table",
                   "profil",
                   "info",
                   "subinfo",
                   affinities_df.segment_name_tokens)

        # Repartition df
        grouped_df = grouped_df.repartition(256)

        def compute(input_df, col_name):
            # Calculate similarity scores info
            input_df = input_df. \
                withColumn("score", score_tokens_func(col(col_name), col("segment_name_tokens")))
            # Rename col
            return input_df. \
                withColumnRenamed("segment_table", "Affinity"). \
                filter(~(input_df.score < 0.5))

        # Rename col
        grouped_df_info = compute(grouped_df, "info")
        grouped_df_subinfo = compute(grouped_df, "subinfo")

        # Rename col
        grouped_df_subinfo = grouped_df_subinfo. \
            withColumnRenamed("segment_table", "Affinity")

        # We drop rows where the score is less than 50% affinity for performance reasons
        grouped_df_info = grouped_df_info.filter(grouped_df_info.score >= 0.5)
        grouped_df_subinfo = grouped_df_subinfo.filter(grouped_df_subinfo.score >= 0.5)

        return grouped_df_info, grouped_df_subinfo

    def result_info_job(self, grouped_df, grouped_df_subinfo):
        """
        For each program, keep only the best affinity score.
        """

        # This one is used to window over info scores
        grouped_df.registerTempTable('table_info')
        q = """
        SELECT
        PROGRAM_ID,
        PROGRAM,
        CATEGORY,
        KIND,
        AFFINITY,
        PROFIL,
        SCORE
        FROM
        (SELECT
            *,
            MAX(score) OVER (PARTITION BY PROGRAM_ID) AS maxscore
        FROM table_info) M
        WHERE score = maxscore
        """
        result_df_info = self.spark.sql(q)

        # This one is used to window over subinfo scores
        grouped_df_subinfo.registerTempTable('table_subinfo')
        q = """
        SELECT
        PROGRAM_ID,
        PROGRAM,
        CATEGORY,
        KIND,
        AFFINITY,
        PROFIL,
        SCORE
        FROM
        (SELECT
            *,
            MAX(score) OVER (PARTITION BY PROGRAM_ID) AS maxscore
        FROM table_subinfo) M
        WHERE score = maxscore
        """
        result_df_subinfo = self.spark.sql(q)

        result_df = result_df_info.union(result_df_subinfo)

        return result_df

    def fact_audience_aff_only_job(self, affinities_df):
        """
        Outputs
         - for each user
         - for each segment
         - the affinity score

        +-------+----------+----------------+-----------+-----------------+
        |USER_ID|      date|Banque & Finance|Téléréalité|Action & Aventure|
        +-------+----------+----------------+-----------+-----------------+
        |      0|2022-03-10|               0|          0|              0.1|
        |      1|2022-03-10|               0|          0|              0.9|
        |      2|2022-03-10|               0|          0|              0.8|
        |      3|2022-03-10|               0|          0|              0.5|
        +-------+----------+----------------+-----------+-----------------+
        """

        # SF query
        yesterday = self.now - datetime.timedelta(days=1)
        query = f"""
        with temp as (
            Select distinct epg.episode_id,epg.duration*60 as duration_sec
            from backend.epg
            where duration_sec>120
        ),
        cte as(
            Select user_id,
                daily_prog_affinity.program_id,
                affinity,
                CASE WHEN fact_watch.duration / duration_sec > 1 THEN 1
                    else fact_watch.duration / duration_sec
                end as tot_score, 
                duration, 
                duration_sec
            from dw.fact_watch
            inner join external_sources.daily_prog_affinity
            on fact_watch.program_id = external_sources.daily_prog_affinity.program_id
            inner join temp
            on fact_watch.episode_id = temp.episode_id
            where date_day = '{yesterday}'
            group by user_id, daily_prog_affinity.program_id, affinity, duration, duration_sec
            having tot_score > 0.2 or duration > 600
            order by user_id
        )
        select user_id, affinity, round(sum(tot_score) / count(tot_score), 2) as affinity_rate
        from cte
        group by user_id,affinity
        having affinity_rate > 0.3
        order by user_id desc, affinity_rate desc
        """

        # Load snowflake query to df
        daily_user_affinities = load_snowflake_query_df(self.spark, self.affinities_options, query)
        segment_table_list = [row.segment_table
                              for row in affinities_df.select("segment_table").distinct().collect()]

        df_joined = daily_user_affinities. \
            withColumn("date", lit(self.now)). \
            groupBy("USER_ID", "date"). \
            pivot("AFFINITY", values=segment_table_list). \
            agg(F.max("AFFINITY_RATE")). \
            fillna(0)

        return df_joined

    def fact_audience_aff_only_flag_job(self):
        """
        Outputs the same as the previous function but flags the scores to be binary.
        If score < 0.4 then 0 else 1. For some specific columns, scores are joined from
        the table DW.FACT_AUDIENCE_PROG_TYPE_AFFINITY which uses quantile info
        """
        # 1 - Work on traditional affinities
        # Load data
        fact_affinity_df = load_snowflake_table(self.spark, self.affinities_options, "DW.FACT_AUDIENCE_AFF_ONLY")

        # Define the names of the cols to aggregate
        cols_to_agg = [c for c in fact_affinity_df.columns
                       if c not in ['"date"', "USER_ID", "date"]]
        agg_dict = {k: "max" for k in cols_to_agg}

        # There is a misunderstanding on the date name, let's be sure and avoid an error
        date_name = "date" if "date" in fact_affinity_df.columns else '"date"'

        # Get the last 90 days of user affinity tagging
        agg_fact_affinity_df = fact_affinity_df. \
            where((col(date_name) >= self.now - self.lookback) & (col(date_name) <= self.now)). \
            groupby("USER_ID"). \
            agg(agg_dict)

        # Threshold the value for each affinity with 40%
        for column in cols_to_agg:
            agg_fact_affinity_df = agg_fact_affinity_df. \
                withColumn(column,
                           when(col("max({})".format(column)) >= 0.4, lit(1)).
                           otherwise(lit(0)))

        # 2 - Work on quantile affinities
        # Now join on the quantile table for the following columns
        # Keys will be dropped and replaced with columns in the values from
        # the table DW.FACT_AUDIENCE_PROG_TYPE_AFFINITY which uses quantile info
        corresponding_columns = {"Film": "MOVIE", "Séries": "SERIE", "Documentaires": "DOCUMENTARY"}

        # Load the table on the right day
        quantile_prog_type_df = load_snowflake_table(self.spark, self.affinities_options,
                                                     "DW.FACT_AUDIENCE_PROG_TYPE_AFFINITY"). \
            where(col("DATE_DAY") == self.now)

        # Join the two affinity tables on USER_ID, drop the attributes that will be used from the other tables
        global_agg_fact_affinity_df = agg_fact_affinity_df. \
            drop(*list(corresponding_columns.keys())). \
            join(quantile_prog_type_df, quantile_prog_type_df.USER_ID == agg_fact_affinity_df.USER_ID). \
            drop(agg_fact_affinity_df.USER_ID)

        # Rename attributes to match the one droped
        for k, v in corresponding_columns.items():
            global_agg_fact_affinity_df = global_agg_fact_affinity_df. \
                withColumnRenamed(v, k)

        # 3 - Work on ad affinities
        ad_affinities_df = self.compute_ad_affinities()

        # Keep only the column with the right names
        global_agg_fact_affinity_df = global_agg_fact_affinity_df. \
            join(ad_affinities_df, ad_affinities_df.USER_ID == global_agg_fact_affinity_df.USER_ID, "left"). \
            drop(ad_affinities_df.USER_ID). \
            withColumn('date', lit(self.now)). \
            select("USER_ID", 'date', *cols_to_agg, "small_watcher"). \
            fillna(0, subset=["small_watcher"])

        return global_agg_fact_affinity_df

    def fact_audience_aff_regie_job(self, global_agg_fact_affinity_df):
        users_df = load_snowflake_table(self.spark, self.users_options, "BACKEND.USER_RAW")
        fact_audience_aff_regie = global_agg_fact_affinity_df.select("USER_ID")
        aff_cols = [c for c in global_agg_fact_affinity_df.columns
                    if c not in ['"date"', "USER_ID", "date"]]

        global_agg_fact_affinity_df = global_agg_fact_affinity_df. \
            join(users_df, users_df.ID == global_agg_fact_affinity_df.USER_ID). \
            drop(users_df.ID). \
            withColumn("AGE", self.now.year - year("BIRTHDAY")). \
            select("USER_ID", "AGE", "GENDER", *aff_cols)

        aff_dict = {"Cuisine": {
            "gender": "F",
            "low_age": 18,
            "high_age": 49,
            "refresh": 45}

        }

        def create_segments(aff, params, global_agg_fact_affinity_df):
            user_aff_hist_df = load_snowflake_table(self.spark, self.affinities_options,
                                                    "DW.FACT_AUDIENCE_AFF_ONLY"). \
                where(col("date") >= self.now - self.lookback). \
                where(col(aff) > 0). \
                withColumn("REFRESH", datediff(lit(self.now), col("date"))). \
                groupBy("USER_ID"). \
                agg(min("REFRESH").alias("REFRESH")). \
                select("USER_ID", "REFRESH"). \
                distinct()

            aff_df = global_agg_fact_affinity_df. \
                where(col(aff) == 1). \
                join(user_aff_hist_df, user_aff_hist_df.USER_ID == global_agg_fact_affinity_df.USER_ID). \
                drop(user_aff_hist_df.USER_ID)

            name, condition = aff_conditions(aff, params)

            aff_df = aff_df. \
                withColumn(name, when(condition, 1).otherwise(0))

            return aff_df, name

        aff_regie = []
        for key, values in aff_dict.items():
            aff_df, name = create_segments(key, values, global_agg_fact_affinity_df)
            fact_audience_aff_regie = fact_audience_aff_regie. \
                join(aff_df.alias("aff_df"), col("aff_df.USER_ID") == fact_audience_aff_regie.USER_ID, "left"). \
                drop(col("aff_df.USER_ID"))
            aff_regie.append(name)

        fact_audience_aff_regie = fact_audience_aff_regie. \
            fillna(0). \
            select("USER_ID", *aff_regie)

        return fact_audience_aff_regie

    def compute_ad_affinities(self, max_duration_small_watch_in_mins=90, min_ratio=0.8):
        """
        for now ad affinities are :
        - small watch : watch < 1h30 per connexion is true 80% of the time
        """
        df = load_snowflake_table(self.spark, self.quantile_options, "DW.FACT_AUDIENCE_DAILY_USER_PROG_TYPE"). \
            where(f"DATE_DAY > '{str(self.now - self.lookback)}' and DATE_DAY <= '{str(self.now)}'")

        small_watches_expression = \
            (F.sum(F.when(F.col("watch_per_day") < F.lit(max_duration_small_watch_in_mins * 60), 1).
                   otherwise(0)) / F.count("watch_per_day")). \
            alias("small_watch_ratio")

        return df. \
            groupBy("USER_ID", "DATE_DAY"). \
            agg(F.sum(F.col("watch_duration")).alias("watch_per_day")). \
            groupBy("USER_ID"). \
            agg(small_watches_expression). \
            select("USER_ID", F.when(F.col("small_watch_ratio") > min_ratio, 1).otherwise(0).alias("small_watcher"))


if __name__ == "__main__":
    job = AffinitySegmentJob()
    job.launch()
