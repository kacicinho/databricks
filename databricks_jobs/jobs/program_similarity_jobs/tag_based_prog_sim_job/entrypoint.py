import re
from datetime import timedelta
from collections import Counter

import nltk
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, ArrayType
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.affinity_lib import get_spacy_model
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos, build_full_program_with_infos, MANGO_CHANNELS
from databricks_jobs.jobs.utils.spark_utils import typed_udf


class TagBasedProgramSimilarityJob(Job):
    DAILY_RECO_TABLE = "RECO_PROG_PROGS_META_VAR"
    AB_TEST_VARIATION = 'A'

    def __init__(self, *args, **kwargs):
        super(TagBasedProgramSimilarityJob, self).__init__(*args, **kwargs)

        self.now = self.parse_date_args()
        self.delta = timedelta(days=14)

        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

        self.stop_words = set(nltk.corpus.stopwords.words('french'))
        self.stop_words.update(['avoir', 'être', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'je', 'tu'])

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):

        tag_df = load_snowflake_table(self.spark, self.options, "backend.rel_episode_tag")
        episode_info_df = build_episode_df(self.spark, self.options)

        # 1 - Import Broadcast and VOD program available
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False). \
            where("REF_PROGRAM_KIND_ID != 93")

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True). \
            where("REF_PROGRAM_KIND_ID != 93")

        # 2 - Filter only on Mango channels for the moment
        prog_df = broadcast_df.union(vod_df). \
            where(F.col("CHANNEL_ID").isin(*MANGO_CHANNELS))

        # 3 - Buils sim matrix
        sim_matrix_df = self.build_program_sim_matrix(prog_df, tag_df, description_similarity=True). \
            where("method = 'description'")

        # 4.1 Write to daily reco
        self.write_recos_to_snowflake(sim_matrix_df, self.DAILY_RECO_TABLE)

    def process_description(self, program_description):
        # Process text informations and concatenate title, summary, kind, category
        nlp = get_spacy_model()

        def concat_info(x):
            special_charac = ['(', ')', '[', ']', '@', '{', '}', '&', '#', '*', '/', ',', '-', '_', '!', '?', ';', ':',
                              '.']
            reg = re.compile(r"\b([a-zA-ZÉéèêàâëïîçô\-]+)\b")
            summary = ' '.join(reg.findall(str(x['SUMMARY'])))
            title = ' '.join(reg.findall(str(x['TITLE'])))
            doc = nlp('.'.join([title, summary]))
            tokens = [t.lemma_.lower() for t in doc if t.lemma_.lower() not in list(self.stop_words) + special_charac]

            description = ' '.join(tokens)
            new_tokens = [description]

            if str(x['CATEGORY']) != 'Indéterminé':
                new_tokens.append(str(x['CATEGORY']))

            if str(x['KIND']) != 'Indéterminé':
                new_tokens.append(str(x['KIND']))

            info_concat = '.'.join(new_tokens)

            return info_concat

        # Handle the case where the input df is empty
        if len(program_description) > 0:
            program_description['info_concat'] = program_description.apply(concat_info, axis=1)
        else:
            program_description['info_concat'] = None

        return program_description

    def build_tags_program(self, program_to_tag):
        """
        Add tags to untag program:
        - import information of all programs with description and tags
        - Vectorize the programs using Tfidf Vectorizer
        - Fit KNN with tagged programs
        - Find the 10 nearest programs based on text information of untagged program
        - Assign top 15 tags to untagged program
        """
        program_with_full_infos = build_full_program_with_infos(self.spark, self.options). \
            where("REF_PROGRAM_KIND_ID != 93"). \
            select("PROGRAM_ID", "CATEGORY", "KIND", "TITLE",
                   "SUMMARY", "TAGS", "N_TAGS")

        program_with_description = program_with_full_infos. \
            where(program_with_full_infos.SUMMARY.isNotNull())

        program_no_tags = program_with_description. \
            join(program_to_tag, program_with_description.PROGRAM_ID == program_to_tag.PROGRAM_ID). \
            drop(program_to_tag.PROGRAM_ID). \
            withColumn("PROGRAM_ID", program_with_description.PROGRAM_ID.cast(IntegerType())). \
            distinct()

        def extract_tags(train, x, vectorizer):
            """
            :param train: dataframe with progs with tags
            :param x: a program with no tag
            :param vectorizer: tfidf or count vectorizer
            :return: List[str]
            """
            # 1 - Perform inference
            no_tag_prog = x['info_concat']
            no_tag_prog_vect = vectorizer.transform([no_tag_prog])
            res = knn.kneighbors(no_tag_prog_vect.reshape(1, -1), return_distance=False)

            # 2 - Map from programs to set of tags
            # 2.1 - each prog gives a list of tags
            nearest_prog = train.iloc[list(res[0])]
            all_tags = list(itertools.chain(*nearest_prog['TAGS']))
            # Keep the top 15 most common tags
            c = Counter(all_tags)
            return list(dict(c.most_common(15)).keys())

        program_with_description_tag = program_with_description.where("N_TAGS > 0").toPandas()
        train_data = self.process_description(program_with_description_tag)
        all_infos = list(train_data['info_concat'].values)
        tfidf = TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), max_features=32768,
                                min_df=1, stop_words=self.stop_words,
                                use_idf=True)
        train_data_vectorized = tfidf.fit_transform(all_infos)

        knn = NearestNeighbors(metric='euclidean', n_neighbors=10, p=2)
        knn.fit(train_data_vectorized)

        program_no_tags_df = program_no_tags.toPandas()
        program_no_tags_df = self.process_description(program_no_tags_df)
        program_no_tags_df['TAGS'] = program_no_tags_df.apply(
            lambda x: x['TAGS'] if x['N_TAGS'] > 0 else extract_tags(train_data, x, tfidf).tolist(), axis=1)
        program_no_tags_df['PROGRAM_ID'] = program_no_tags_df['PROGRAM_ID'].apply(int)

        schema = StructType([
            StructField("PROGRAM_ID", IntegerType(), nullable=True),
            StructField("CATEGORY", StringType(), nullable=True),
            StructField("KIND", StringType(), nullable=True),
            StructField("TITLE", StringType(), nullable=True),
            StructField("SUMMARY", StringType(), nullable=True),
            StructField("TAGS", ArrayType(StringType()), nullable=True),
            StructField("N_TAGS", IntegerType(), nullable=True),
            StructField("info_concat", StringType(), nullable=True)
        ])
        return self.spark.createDataFrame(program_no_tags_df, schema)

    def build_similarity_description(self, program_to_tag):
        """
        Add tags to untag program:
        - import information of all programs with description
        - Vectorize the programs using Tfidf Vectorizer
        - Fit KNN with available programs
        - Find the 30 nearest programs based on text information of untagged program
        """
        program_with_full_infos = build_full_program_with_infos(self.spark, self.options). \
            where("REF_PROGRAM_KIND_ID != 93"). \
            select("PROGRAM_ID", "CATEGORY", "KIND", "TITLE",
                   "SUMMARY", "TAGS", "N_TAGS")

        program_with_description = program_with_full_infos. \
            where(program_with_full_infos.SUMMARY.isNotNull())

        program_no_tags = program_with_description. \
            join(program_to_tag, program_to_tag.PROGRAM_ID == program_with_description.PROGRAM_ID). \
            drop(program_to_tag.PROGRAM_ID). \
            distinct()

        def similar_prog(train, x, vectorizer):
            no_tag_prog = x['info_concat']
            no_tag_prog_vect = vectorizer.transform([no_tag_prog])
            data = knn.kneighbors(no_tag_prog_vect.reshape(1, -1), return_distance=True)
            res = pd.DataFrame({"distance": data[0][0]}, index=data[1][0])
            nearest_prog = pd.merge(res, train, left_index=True, right_index=True)
            nearest_prog[['PROGRAM_ID']] = nearest_prog[['PROGRAM_ID']].astype('int64')

            rez = nearest_prog[["PROGRAM_ID", "distance"]].to_dict(orient='records')
            return list(
                map(lambda z: (z["PROGRAM_ID"], z["distance"]),
                    filter(lambda z: z["distance"] > 1e-4, rez))
            )

        broadcast_df_description = program_no_tags.toPandas()
        train_data = self.process_description(broadcast_df_description)
        all_infos = list(train_data['info_concat'].values)
        tfidf = TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), max_features=32768,
                                min_df=1, stop_words=self.stop_words,
                                use_idf=True)
        train_data_vectorized = tfidf.fit_transform(all_infos)

        knn = NearestNeighbors(metric='euclidean', n_neighbors=min(len(train_data), 29), p=2)
        knn.fit(train_data_vectorized)

        program_no_tags_df = program_no_tags.toPandas()
        program_no_tags_df = self.process_description(program_no_tags_df)
        program_no_tags_df['PROGRAM_ID'] = program_no_tags_df['PROGRAM_ID'].apply(int)

        if len(program_no_tags_df) > 0:
            program_no_tags_df['RECOMMENDATIONS'] = program_no_tags_df.apply(
                lambda x: similar_prog(train_data, x, tfidf), axis=1)
        else:
            program_no_tags_df['RECOMMENDATIONS'] = None

        schema = StructType([
            StructField("PROGRAM_ID", IntegerType(), nullable=True),
            StructField("CATEGORY", StringType(), nullable=True),
            StructField("KIND", StringType(), nullable=True),
            StructField("TITLE", StringType(), nullable=True),
            StructField("SUMMARY", StringType(), nullable=True),
            StructField("N_TAGS", IntegerType(), nullable=True),
            StructField("info_concat", StringType(), nullable=True),
            StructField("RECOMMENDATIONS", ArrayType(
                StructType([StructField("PROGRAM_ID", IntegerType(), False),
                            StructField("distance", FloatType(), False)])), nullable=True)
        ])
        program_no_tags_df = program_no_tags_df.drop(columns=["TAGS"])
        program_no_tags_df = self.spark.createDataFrame(program_no_tags_df, schema)

        return program_no_tags_df. \
            select("PROGRAM_ID", F.explode("RECOMMENDATIONS").alias("RECOMMENDATIONS")). \
            withColumn("RECOMMENDED_PROGRAM_ID", F.col("RECOMMENDATIONS.PROGRAM_ID")). \
            withColumn("score_t", F.lit(1) / F.col("RECOMMENDATIONS.distance")). \
            withColumn("score_k", F.lit(0)). \
            withColumn("score_N", F.lit(0)). \
            withColumn('rank', F.row_number().over(Window.partitionBy("PROGRAM_ID").orderBy("score_t"))).\
            drop("RECOMMENDATIONS")

    def build_program_sim_matrix(self, broadcast_df, tag_df, kind_based=False, description_similarity=False):
        """
        Steps :
        - Get a set of tags per prog_id
        for tagged program:
            - crossJoin and compute scores based on similar tag, kind and number of tag for each p1, p2 pair
            - Keep 30 most similar propositions
        for untagged program:
            -use text informations to find similar program
        """

        @typed_udf(FloatType())
        def score_tag(l1, l2):
            s1 = set(l1)
            s2 = set(l2)
            return 1.0 * len(s1.intersection(s2))

        program_df = load_snowflake_table(self.spark, self.options, "backend.program")
        category_df = load_snowflake_table(self.spark, self.options, "backend.ref_program_category")
        kind_df = load_snowflake_table(self.spark, self.options, "backend.ref_program_kind")

        # Select all the tags associated to a prog_id (some prog_id have several episode_id with different informations)
        # and keep tags with most occurences over episode_id (rank < 20)
        with_tag_df = broadcast_df. \
            join(tag_df, tag_df.EPISODE_ID == broadcast_df.EPISODE_ID). \
            join(program_df, program_df.ID == broadcast_df.PROGRAM_ID). \
            join(category_df, category_df.ID == program_df.REF_PROGRAM_CATEGORY_ID). \
            withColumnRenamed("NAME", "CATEGORY"). \
            join(kind_df, kind_df.ID == program_df.REF_PROGRAM_KIND_ID). \
            withColumnRenamed("NAME", "KIND"). \
            groupby("PROGRAM_ID", "REF_TAG_ID", "CATEGORY", "KIND"). \
            agg(F.countDistinct(broadcast_df.EPISODE_ID).alias("OCCUR")). \
            withColumn("rank", F.rank().over(Window.partitionBy("PROGRAM_ID").orderBy(F.desc("OCCUR")))). \
            where("rank <= 20"). \
            select("PROGRAM_ID", "REF_TAG_ID", "CATEGORY", "KIND", "OCCUR", "rank"). \
            groupBy("PROGRAM_ID", "KIND"). \
            agg(F.collect_list("REF_TAG_ID").alias("TAGS")). \
            withColumn("N_TAGS", F.size("TAGS"))

        # If we are tag based similarity, we need to do SUMMARY -> TAGS
        if not description_similarity:
            prog_no_tag_df = broadcast_df. \
                join(tag_df, tag_df.EPISODE_ID == broadcast_df.EPISODE_ID, "left"). \
                groupby("PROGRAM_ID"). \
                agg(F.countDistinct("REF_TAG_ID").alias("N_TAGS")). \
                where("N_TAGS = 0"). \
                select("PROGRAM_ID"). \
                distinct()
            # Here we use description as intermediate to build tags
            new_tag_df = self.build_tags_program(prog_no_tag_df). \
                withColumn("N_TAGS", F.size("TAGS")). \
                select("PROGRAM_ID", "KIND", "TAGS", "N_TAGS"). \
                distinct()
            tag_df = with_tag_df.union(new_tag_df)  # joining programs with new tags
        else:
            tag_df = with_tag_df

        # We compute the 3 scores f (t, k and N for each pairwise program)
        scoring_matrix_df = tag_df.alias("ref"). \
            crossJoin(F.broadcast(tag_df).alias("other")). \
            where("ref.PROGRAM_ID != other.PROGRAM_ID"). \
            withColumn("score_t", score_tag("ref.TAGS", "other.TAGS")). \
            withColumn("score_k", F.when(F.col("ref.KIND") == F.col("other.KIND"), 1).otherwise(F.lit(0))). \
            withColumn("score_N", F.col("other.N_TAGS")). \
            select(F.col("ref.PROGRAM_ID").alias("PROGRAM_ID"),
                   F.col("other.PROGRAM_ID").alias("RECOMMENDED_PROGRAM_ID"),
                   "score_t",
                   "score_k",
                   "score_N")

        # Preparing the output
        out_matrix = self.order_final_results(scoring_matrix_df, kind_based)
        if description_similarity:
            prog_no_tag_df = broadcast_df. \
                select("PROGRAM_ID"). \
                distinct()
            scoring_no_tag = self.build_similarity_description(prog_no_tag_df)
            return out_matrix.\
                withColumn("method", F.lit("tags")). \
                union(scoring_no_tag.
                      withColumn("method", F.lit("description")))  # joining no tag program results
        else:
            return out_matrix.\
                withColumn("method", F.lit("tags"))

    def order_final_results(self, scoring_matrix_df, kind_based):
        # Here we use score_k as main metric (1 if same kind 0 otherwise) or
        # score_t as main metric (number of tags in common)
        ordering_rule = (F.desc("score_k"), F.desc("score_t"), F.asc("score_N")) if kind_based \
            else (F.desc("score_t"), F.desc("score_k"), F.asc("score_N"))

        return scoring_matrix_df. \
            withColumn("rank",
                       F.row_number().over(Window.partitionBy("PROGRAM_ID")
                                           .orderBy(*ordering_rule))). \
            where("rank < 30")

    def write_recos_to_snowflake(self, sim_matrix_df, table_name,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        sim_matrix_df = sim_matrix_df. \
            groupby("PROGRAM_ID", "method"). \
            agg(F.collect_list(F.struct(F.col("RECOMMENDED_PROGRAM_ID").alias("program_id"),
                                        F.col("score_t").alias("rating"))).alias("recommendations")). \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(sim_matrix_df, self.write_options, table_name, write_mode)
        return sim_matrix_df


if __name__ == "__main__":
    job = TagBasedProgramSimilarityJob()
    job.launch()
