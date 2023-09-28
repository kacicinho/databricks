import re
from datetime import timedelta

import nltk
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.window import Window

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.tag_based_prog_sim_utils import build_sim_matrix_on_descritpion, \
    build_sim_matrix_on_tag
from databricks_jobs.jobs.utils.affinity_lib import get_spacy_model
from databricks_jobs.jobs.utils.utils import load_snowflake_table, get_snowflake_options, write_df_to_snowflake
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info, \
    build_vod_df_with_episode_infos, build_full_program_with_infos
from databricks_jobs.jobs.utils.spark_utils import typed_udf


class TagBasedChannelProgramSimilarityJob(Job):
    DAILY_RECO_TABLE = "RECO_CHANNEL_PROG_PROGS_META_VAR"
    AB_TEST_VARIATION = 'A'

    def __init__(self, *args, **kwargs):
        super(TagBasedChannelProgramSimilarityJob, self).__init__(*args, **kwargs)

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

        # Filter only on Mango: -1, TF1: 1, France 2: 2, M6: 3, AB: 4
        channel_group_df = load_snowflake_table(self.spark, self.options, "dw.dim_channel_group"). \
            filter((F.col("CHANNEL_GROUP_ID").isNotNull()) & F.col("CHANNEL_GROUP_ID").isin(*[-1, 1, 2, 3, 4])). \
            select("CHANNEL_ID", "CHANNEL_GROUP_ID")
        episode_info_df = build_episode_df(self.spark, self.options)

        # 1 - Import Broadcast and VOD program available
        broadcast_df = build_broadcast_df_with_episode_info(self.spark, self.options, episode_info_df,
                                                            self.now, self.now + self.delta,
                                                            free_bundle=False). \
            where("REF_PROGRAM_KIND_ID != 93")

        vod_df = build_vod_df_with_episode_infos(self.spark, self.options, episode_info_df, self.now, self.delta,
                                                 min_duration_in_mins=-1, allow_extra=True). \
            where("REF_PROGRAM_KIND_ID != 93")

        # 2 - Add channel group id info
        prog_df = broadcast_df.union(vod_df). \
            join(channel_group_df, broadcast_df.CHANNEL_ID == channel_group_df.CHANNEL_ID). \
            drop(channel_group_df.CHANNEL_ID)

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

        # Keep only one description by program_id
        program_with_description = program_with_full_infos. \
            where(program_with_full_infos.SUMMARY.isNotNull()). \
            withColumn("rank", F.row_number().over(Window.partitionBy(["PROGRAM_ID"]).orderBy(F.desc("N_TAGS")))).\
            where("rank = 1"). \
            drop("rank")

        program_no_tags = program_with_description. \
            join(program_to_tag, program_with_description.PROGRAM_ID == program_to_tag.PROGRAM_ID). \
            drop(program_to_tag.PROGRAM_ID). \
            withColumn("PROGRAM_ID", program_with_description.PROGRAM_ID.cast(IntegerType())). \
            distinct()

        program_with_description_tag = program_with_description.where("N_TAGS > 0").toPandas()
        train_data = self.process_description(program_with_description_tag)

        program_no_tags_df = program_no_tags.toPandas()
        program_no_tags_df = self.process_description(program_no_tags_df)

        res = build_sim_matrix_on_tag(self.spark, program_no_tags_df, train_data, self.stop_words, min_n_neighbours=10)

        return res

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
            where(program_with_full_infos.SUMMARY.isNotNull()). \
            withColumn("rank", F.row_number().over(Window.partitionBy(["PROGRAM_ID"]).orderBy(F.desc("N_TAGS")))).\
            where("rank = 1"). \
            drop("rank")

        program_no_tags = program_with_description. \
            join(program_to_tag, program_to_tag.PROGRAM_ID == program_with_description.PROGRAM_ID). \
            drop(program_to_tag.PROGRAM_ID). \
            distinct()

        broadcast_df_description = program_no_tags.toPandas()
        train_data = self.process_description(broadcast_df_description)

        program_no_tags_df = program_no_tags.toPandas()
        program_no_tags_df = self.process_description(program_no_tags_df)

        # Loop through each channel_group_id
        channel_group_ids = program_no_tags_df["CHANNEL_GROUP_ID"].unique()
        for i, channel_group_id in enumerate(channel_group_ids):
            channel_train_data = train_data[train_data["CHANNEL_GROUP_ID"] == channel_group_id]
            channel_train_data.reset_index(drop=True, inplace=True)
            program_no_tags_to_reco = program_no_tags_df[program_no_tags_df["CHANNEL_GROUP_ID"] == channel_group_id]

            res = build_sim_matrix_on_descritpion(self.spark, program_no_tags_to_reco, channel_train_data, self.stop_words, min_n_neighbours=30)
            if i == 0:
                final_program_no_tags_df = res
            else:
                final_program_no_tags_df = final_program_no_tags_df.union(res)

        return final_program_no_tags_df

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
            groupby("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID", "REF_TAG_ID", "CATEGORY", "KIND"). \
            agg(F.countDistinct(broadcast_df.EPISODE_ID).alias("OCCUR")). \
            withColumn("rank", F.rank().over(Window.partitionBy("PROGRAM_ID").orderBy(F.desc("OCCUR")))). \
            where("rank <= 20"). \
            select("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID", "REF_TAG_ID", "CATEGORY", "KIND", "OCCUR", "rank"). \
            groupBy("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID", "KIND"). \
            agg(F.collect_list("REF_TAG_ID").alias("TAGS")). \
            withColumn("N_TAGS", F.size("TAGS"))

        # If we are tag based similarity, we need to do SUMMARY -> TAGS
        if not description_similarity:
            prog_no_tag_df = broadcast_df. \
                join(tag_df, tag_df.EPISODE_ID == broadcast_df.EPISODE_ID, "left"). \
                groupby("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID"). \
                agg(F.countDistinct("REF_TAG_ID").alias("N_TAGS")). \
                where("N_TAGS = 0"). \
                select("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID"). \
                distinct()
            # Here we use description as intermediate to build tags
            new_tag_df = self.build_tags_program(prog_no_tag_df). \
                withColumn("N_TAGS", F.size("TAGS")). \
                select("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID", "KIND", "TAGS", "N_TAGS"). \
                distinct()
            tag_df = with_tag_df.union(new_tag_df)  # joining programs with new tags
        else:
            tag_df = with_tag_df

        # Cross join only with program with a unique channel ids (allows to not recommend same prog that are 
        # broadcast on different channels
        tag_df_unique_channel = tag_df. \
            withColumn("rank", F.row_number().over(Window.partitionBy("CHANNEL_GROUP_ID", "PROGRAM_ID").orderBy(F.desc("N_TAGS")))). \
            where("rank = 1"). \
            drop("rank")

        # We compute the 3 scores f (t, k and N for each pairwise program)
        scoring_matrix_df = tag_df.alias("ref"). \
            crossJoin(F.broadcast(tag_df_unique_channel).alias("other")). \
            where("ref.PROGRAM_ID != other.PROGRAM_ID AND ref.CHANNEL_GROUP_ID == other.CHANNEL_GROUP_ID"). \
            withColumn("score_t", score_tag("ref.TAGS", "other.TAGS")). \
            withColumn("score_k", F.when(F.col("ref.KIND") == F.col("other.KIND"), 1).otherwise(F.lit(0))). \
            withColumn("score_N", F.col("other.N_TAGS")). \
            select(F.col("ref.CHANNEL_ID").alias("CHANNEL_ID"),
                   F.col("ref.PROGRAM_ID").alias("PROGRAM_ID"),
                   F.col("ref.CHANNEL_ID").alias("RECOMMENDED_CHANNEL_ID"),
                   F.col("other.PROGRAM_ID").alias("RECOMMENDED_PROGRAM_ID"),
                   "score_t",
                   "score_k",
                   "score_N")

        # Preparing the output
        out_matrix = self.order_final_results(scoring_matrix_df, kind_based)
        if description_similarity:
            # We keep at most 1 channel_id per program_id x channel_group
            prog_no_tag_df = broadcast_df. \
                select("CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID"). \
                distinct(). \
                withColumn("rank", F.row_number().over(Window.partitionBy(["CHANNEL_GROUP_ID", "PROGRAM_ID"]).orderBy(F.desc("CHANNEL_ID")))).\
                where("rank = 1"). \
                drop("rank")
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
                       F.row_number().over(Window.partitionBy("CHANNEL_ID", "PROGRAM_ID")
                                           .orderBy(*ordering_rule))). \
            where("rank < 30")

    def write_recos_to_snowflake(self, sim_matrix_df, table_name,
                                 write_mode="append", variation=AB_TEST_VARIATION):
        sim_matrix_df = sim_matrix_df. \
            groupby("CHANNEL_ID", "PROGRAM_ID", "method"). \
            agg(F.collect_list(F.struct(F.col("RECOMMENDED_CHANNEL_ID").alias("channel_id"),
                                        F.col("RECOMMENDED_PROGRAM_ID").alias("program_id"),
                                        F.col("score_t").alias("rating"))).alias("recommendations")). \
            withColumn("UPDATE_DATE", F.lit(self.now)). \
            withColumn("VARIATIONS", F.lit(variation))
        write_df_to_snowflake(sim_matrix_df, self.write_options, table_name, write_mode)
        return sim_matrix_df


if __name__ == "__main__":
    job = TagBasedChannelProgramSimilarityJob()
    job.launch()
