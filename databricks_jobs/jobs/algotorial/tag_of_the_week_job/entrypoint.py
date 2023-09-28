from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import get_snowflake_options, write_df_to_snowflake
import databricks_jobs.jobs.algotorial.tag_of_the_week_job.helpers as h
from databricks_jobs.jobs.utils.utils import load_snowflake_table
from pyspark.sql import functions as F


class TagOfTheWeekJob(Job):
    nb_week_lookback = 52
    similarity_score_min = 0.3
    rail_size = 10
    max_program_per_channel_per_rail = 3
    nb_tags_for_next_week = 6

    def __init__(self, *args, **kwargs):
        super(TagOfTheWeekJob, self).__init__(*args, **kwargs)
        self.now = self.parse_date_args()
        self.options = get_snowflake_options(self.conf, "PROD")
        self.write_options = get_snowflake_options(self.conf, "PROD", "ML", keep_column_case="off")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        # 1. tables from snowflake loading
        rel_tvbundle_channel = load_snowflake_table(self.spark, self.options, "backend.rel_tvbundle_channel")
        broadcast = load_snowflake_table(self.spark, self.options, 'BACKEND.BROADCAST')
        ref_tag = load_snowflake_table(self.spark, self.options, 'ML.REF_KEYWORDS')
        similarities = load_snowflake_table(self.spark, self.options, 'ML.PROGRAM_KEYWORD_SIMILARITY')
        vod = load_snowflake_table(self.spark, self.options, 'ML.DIM_VOD')

        # 2. filtered broadcast with infos
        available_broadcast = h.available_broadcast(broadcast, rel_tvbundle_channel,
                                                    weeks_lookback=self.nb_week_lookback)
        broadcast_with_infos = h.program_with_tag_info(available_broadcast, similarities, ref_tag,
                                                       score_min=self.similarity_score_min)
        best_prog_only_broadcast = h.keep_only_best_programs(broadcast_with_infos,
                                                             self.max_program_per_channel_per_rail)

        # 3. keywords scores computations
        keyword_score_per_week_df = h.keyword_score_per_week(best_prog_only_broadcast, rail_size=self.rail_size)
        keyword_scores = keyword_score_per_week_df.pivot(index='WEEK_START_AT', columns='KEYWORD',
                                                         values='KEYWORD_SCORE').fillna(0)
        centered_scores = keyword_scores.apply(h.normalize_data, axis=0)
        ranking_per_week_df = h.ranking_per_week(centered_scores)

        # 4. tags selection
        next_week_number, next_monday = h.next_week_info(self.spark)
        # possible tags
        possible_tags_for_next_week = h.possible_tags(best_prog_only_broadcast, rail_size=self.rail_size,
                                                      week=next_week_number)
        # all tags ranked by score
        tags_of_next_week = ranking_per_week_df['week_' + str(next_week_number)].values.tolist()
        # truncated intersection of two previous lists
        top_tags = [tag for tag in tags_of_next_week if tag in possible_tags_for_next_week][:self.nb_tags_for_next_week]

        # 5. programs of the week selection
        # load filtered vod with infos and concat vod to broadcast to widen program pool
        available_vod = h.available_vod(vod).withColumn("WEEK_START_AT", F.lit(next_week_number))
        vod_with_infos = h.program_with_tag_info(available_vod, similarities, ref_tag,
                                                 score_min=self.similarity_score_min)
        best_prog_only_vod = h.keep_only_best_programs(vod_with_infos, self.max_program_per_channel_per_rail)

        final_columns = ['PROGRAM_ID', 'CHANNEL_ID', 'KIND', 'KEYWORD', 'SIMILARITY_SCORE', 'WEEK_START_AT']
        all_programs = best_prog_only_broadcast.select(*final_columns).union(best_prog_only_vod.select(*final_columns))

        # 6. programs of the week to snowflake
        # data table
        programs_of_the_week_df = h.programs_of_the_week(all_programs, top_tags, next_week_number)
        write_df_to_snowflake(self.spark.createDataFrame(programs_of_the_week_df), self.write_options,
                              "ML.TAG_OF_THE_WEEK", 'overwrite')
        # ref table
        ref_table = h.get_ref_table(self.spark, next_monday, top_tags)
        write_df_to_snowflake(ref_table, self.write_options, 'ML.REF_TAG_OF_THE_WEEK', 'overwrite')


if __name__ == "__main__":
    job = TagOfTheWeekJob()
    job.launch()
