import datetime
import importlib
import os
from collections import namedtuple
from unittest import TestCase, mock, skip
from unittest.mock import patch

from databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint import PopularRecoJob
from databricks_jobs.jobs.utils import popular_reco_utils
from databricks_jobs.jobs.utils.popular_reco_utils import complete_recos, keep_top_X_programs, build_empty_user_csa_pref
from tests.unit.utils.mocks import multiplex_mock, mock_free_channels_query, mock_full_reco_df, mock_user_channels, \
    mock_product_to_tvbundle, create_spark_df_from_data
from tests.unit.utils.test_utils import find_user_rows

from pyspark.sql import SparkSession


class TestPopularRecoJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = PopularRecoJob()
        cls.job.PRIME_START = 0
        cls.job.PRIME_END = 24
        cls.user_csa_perf_df = build_empty_user_csa_pref(cls.job.spark)

    def setUp(self) -> None:
        self.job.REQUIRED_NB_H_PER_AFF = 0
        self.job.MAX_RECO_PER_CAT = 5

    def test_compute_user_likes(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            df = self.job.compute_user_likes(self.job.spark, self.job.options, self.job.now - self.job.delta, 0)
            rows = df.collect()

            self.assertEqual(len(rows), 6)
            self.assertListEqual([r.nb_likes for r in rows], [1, 1, 1, 1, 1, 1])

    def test_total_watch_table(self):
        def create_fact_watch_table_mock(spark):
            now = datetime.datetime.now()
            data = {'REAL_START_AT': [now,
                                      now - datetime.timedelta(days=1),
                                      now - datetime.timedelta(days=2),
                                      datetime.datetime(1990, 1, 1)],
                    'PROGRAM_ID': [1, 0, 1, 1],
                    'DURATION': [501 * 60 * 60, 20 * 60, 500 * 60 * 60, 10000000]}
            return create_spark_df_from_data(spark, data)

        def mock_backend_program(spark):
            n_progs = 6
            pids = list(range(n_progs))
            data = {"ID": pids,
                    "REF_PROGRAM_CATEGORY_ID": [2 for _ in range(n_progs)],
                    "REF_PROGRAM_KIND_ID": [1, 1, 1, 1, 52, 2],
                    "DURATION": [1001 * 60 * 60 for _ in range(n_progs)]}
            # Program 4 will be filtered by program kind filter
            return create_spark_df_from_data(spark, data)

        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name:
                        mock_backend_program(spark)
                        if "program" in table_name else create_fact_watch_table_mock(spark)):
            pop_df = self.job.prepare_recent_pop_df()
            rows = pop_df.collect()
            self.assertEqual(len(rows), 1)
            self.assertEqual(sum([r.total_watch_duration for r in rows if r.PROGRAM_ID == 1]), 1001)

    def test_total_bookmarks_table(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            pop_df = self.job.prepare_total_distinct_bookmarks_df()
            rows = pop_df.collect()
            # One program does not have enough bookmarks to pass the threshold
            self.assertEqual(len(rows), 1)
            # We verify the final value
            self.assertEqual(sum([r.total_distinct_bookmarks for r in rows if r.PROGRAM_ID == 1]), 202)

    def test_prepare_rebroadcast(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            rebroadcast_df = self.job.prepare_rebroadcast()
            rows = rebroadcast_df.collect()
            rebroadcast_prog_0 = find_user_rows(rows, user_id=0, field="PROGRAM_ID")[0].total_rebroadcast
            rebroadcast_prog_1 = find_user_rows(rows, user_id=1, field="PROGRAM_ID")[0].total_rebroadcast
            # Program 0 was available 2 times in last 30 days and Program 1 one time
            self.assertEqual(rebroadcast_prog_0, 2)
            self.assertEqual(rebroadcast_prog_1, 1)

    def test_rating_table(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            pop_df = self.job.prepare_external_rating_df(self.job.spark, self.job.options, min_rating=1.15)
            rows = pop_df.collect()
            self.assertEqual(len(rows), 2)
            self.assertSetEqual({r.PROGRAM_ID for r in rows}, {0, 1})

    def test_affinity_preference(self):
        with mock.patch("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            self.job.REQUIRED_NB_H_PER_AFF = 2
            data = {'USER_ID': [1, 2, 1, 0, 3, 1], "PROGRAM_ID": [0, 0, 1, 1, 0, 1],
                    "total_duration": [10 * 60 * 60, 10 * 60 * 60, 4 * 60 * 60, 4 * 60 * 60, 1, 4 * 60 * 60]}
            user_watch_history_df = create_spark_df_from_data(self.job.spark, data)
            df = self.job.compute_user_affinity_preference(self.job.spark, self.job.options,
                                                           self.job.REQUIRED_NB_H_PER_AFF,
                                                           user_watch_history_df)
            rows = df.collect()

            # User 3 should be filtered out as he has very little time spend on any affinity
            rs = find_user_rows(rows, user_id=3)
            self.assertEqual(len(rs), 0)

            # Prog 0 has 4 affinities other have only 1, we have 2 users with prog_0 -> 4 aff + 1 with 1 only
            self.assertEqual(len(rows), 4 * 2 + 1)

            # User 1 has multiple affinities watched, Action & Aventure and is the most watched watched
            user_1_rows = find_user_rows(rows, user_id=1)
            action_for_user_1 = [r for r in user_1_rows if r.AFFINITY == "Action & Aventure"][0]
            self.assertEqual(action_for_user_1.USER_AFFINITY_RANK, 1)
            # All programs contribute to the time_on_affinity
            self.assertEqual(action_for_user_1.time_on_affinity, (10 + 4 * 2) * 60 * 60)

            # For the affinity Famille, we are behind "Action" so we should have rank 2
            family_for_user_1 = [r for r in user_1_rows if r.AFFINITY == "Famille"][0]
            self.assertEqual(family_for_user_1.USER_AFFINITY_RANK, 2)
            # Only prog_0 watch account for this affinity
            self.assertEqual(family_for_user_1.time_on_affinity, 10 * 60 * 60)

    def test_freely_available_programs(self):
        """
        Test the pipeline where programs are extracted and filtered based on information like duration and other
        """
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 load_snowflake_query_df=lambda spark, *args, **kwargs:
                                 mock_free_channels_query(spark)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                rows = self.job.prepare_freely_available_programs_df().collect()
                prog_ids = [r.PROGRAM_ID for r in rows]
                # Program 3 and 4 should be filtered by condition on duration and program kind
                self.assertSetEqual(set(prog_ids), {0, 1, 2, 5})

    def test_prog_popularity_addition(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         mock_free_channels_query(spark)):
                    prog_df = self.job.prepare_freely_available_programs_df()
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         create_spark_df_from_data(spark,
                                                                   {"PROGRAM_ID": [1], "total_celeb_points": 1})):
                    df = self.job.join_pop_info_to_programs(prog_df)

            rows = df.collect()
            self.assertSetEqual(set([r.PROGRAM_ID for r in rows]), {0, 1, 2, 5})
            self.assertSetEqual(set([r.AFFINITY for r in rows]),
                                {"Action & Aventure", "Famille", "Maison", "Autre", "Thrillers & Policiers"})

    def test_top_k_selection_df(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         mock_free_channels_query(spark)):
                    prog_df = self.job.prepare_freely_available_programs_df()
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         create_spark_df_from_data(spark,
                                                                   {"PROGRAM_ID": [1], "total_celeb_points": 1})):
                    prog_and_info_df = self.job.join_pop_info_to_programs(prog_df)

            df = keep_top_X_programs(prog_and_info_df)
            rows = df.collect()

            # Assert we have a total of 4 + 1 + 1 + 1 rows : prog 0 has 4 affintiies, the other only one
            self.assertEqual(len(rows), 7)

    def test_user_reco(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         mock_free_channels_query(spark)):
                    prog_df = self.job.prepare_freely_available_programs_df()
                with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                         load_snowflake_query_df=lambda spark, *args, **kwargs:
                                         create_spark_df_from_data(spark,
                                                                   {"PROGRAM_ID": [1], "total_celeb_points": 1})):
                    prog_and_info_df = self.job.join_pop_info_to_programs(prog_df)
            _, fa_df = self.job.build_fa_dfs()
            user_watch_history = self.job.build_user_watch_history(self.job.spark, self.job.options, self.job.now)
            user_reco_history = self.job.build_user_reco_history()
            top_k_df = keep_top_X_programs(prog_and_info_df)
            df = self.job.build_user_recos(top_k_df, fa_df, user_watch_history, user_reco_history)
            rows = df.collect()

            # Only user 1 should remain because of the absence of other users in fact watch
            self.assertEqual(len(rows), 3)

    def test_selection_step(self):
        # Create the equivalent
        reco_df_mocked = mock_full_reco_df(self.job.spark)
        user_cha_df = mock_user_channels(self.job.spark)

        # Test 1 : Basic checks on the results
        n_recos = 3
        selected_recos = self.job.select_among_categories(reco_df_mocked, user_cha_df, self.user_csa_perf_df,
                                                          n_recos, n_recos, True)
        rows = selected_recos.collect()
        user_0_rows = find_user_rows(rows, user_id=0)

        # Did we get the right number of returned results
        self.assertEqual(len(user_0_rows), n_recos)

        # Program 3 should be removed because its ranking is 4 > to the others
        reco_ids = [r.PROGRAM_ID for r in user_0_rows]
        self.assertEqual(set(reco_ids), {0, 1, 2})

        # Ranking is computed as affinity_ranking + program_ranking
        reco_ranking = [r.ranking for r in user_0_rows]
        self.assertListEqual(sorted(reco_ranking), [1, 3, 3])

    def test_reco_selection_too_many(self):
        # Test 2 : Sanity checks on full data
        n_recos = 5
        reco_df_mocked = mock_full_reco_df(self.job.spark)
        user_cha_df = mock_user_channels(self.job.spark)
        selected_recos = self.job.select_among_categories(reco_df_mocked, user_cha_df, self.user_csa_perf_df,
                                                          n_recos, n_recos, True)
        rows = selected_recos.collect()

        self.assertEqual(len(rows), 4)
        self.assertListEqual(sorted(r.PROGRAM_ID for r in rows), [0, 1, 2, 3])

    def test_reco_selection_duplicate_program(self):
        data = {"USER_ID": [0] * 5,
                "PROGRAM_ID": list(range(4)) + [0],
                "EPISODE_ID": list(range(4)) + [0],
                "TVBUNDLE_ID": [25, 25, 25, 25, 25],
                "REF_CSA_ID": [0, 0, 0, 0, 0],
                "CHANNEL_ID": [0, 0, 0, 0, 0],
                "AFFINITY": ["Action & Aventure", "Action & Aventure", "Action & Aventure",
                             "Famille", "Horreur"],
                "reco_origin": ["external_rating", "external_rating", "avg_watch_duration", "external_rating",
                                "external_rating"],
                "best_rank": [1, 2, 1, 1, 1],
                "start_rank": [0, 0, 0, 0, 0],
                "USER_AFFINITY_RANK": [0, 0, 0, 2, 0],
                "total_rebroadcast": [2, 1, 0, 1, 2]}
        reco_df_mocked = create_spark_df_from_data(self.job.spark, data)
        n_recos = 5
        user_cha_df = mock_user_channels(self.job.spark)
        selected_recos = self.job.select_among_categories(
            reco_df_mocked, user_cha_df, self.user_csa_perf_df, n_recos, n_recos, True
        )
        rows = selected_recos.collect()

        # We should not have 2 rows for program 0, only 0, 1, 2 and 3
        self.assertEqual(len(rows), 4)
        self.assertListEqual(sorted(r.PROGRAM_ID for r in rows), [0, 1, 2, 3])

    def test_reco_selection_max_per_cat(self):
        data = {"USER_ID": [0] * 5,
                "PROGRAM_ID": list(range(5)),
                "EPISODE_ID": list(range(5)),
                "TVBUNDLE_ID": [25, 25, 25, 25, 25],
                "CHANNEL_ID": [0, 0, 0, 0, 0],
                "REF_CSA_ID": [0, 0, 0, 0, 0],
                "AFFINITY": ["Action & Aventure", "Action & Aventure", "Action & Aventure",
                             "Famille", "Horreur"],
                "reco_origin": ["external_rating", "external_rating", "avg_watch_duration", "external_rating",
                                "external_rating"],
                "best_rank": [1, 2, 1, 1, 1],
                "start_rank": [0, 0, 0, 0, 0],
                "USER_AFFINITY_RANK": [0, 0, 1, 2, 0],
                "total_rebroadcast": [2, 1, 0, 1, 0]}
        self.job.MAX_RECO_PER_CAT = 1
        reco_df_mocked = create_spark_df_from_data(self.job.spark, data)
        n_recos = 5
        user_cha_df = mock_user_channels(self.job.spark)
        selected_recos = self.job.select_among_categories(reco_df_mocked, user_cha_df, self.user_csa_perf_df,
                                                          n_recos, self.job.MAX_RECO_PER_CAT, True)
        rows = selected_recos.collect()

        # We can have only 2 for Action, 3 for Famille and 4 for Horreur
        self.assertEqual(len(rows), 3)
        self.assertListEqual(sorted(r.PROGRAM_ID for r in rows), [2, 3, 4])

    @skip
    def test_csa_filtering(self):
        data = {"USER_ID": [0] * 5,
                "PROGRAM_ID": list(range(4)) + [0],
                "EPISODE_ID": list(range(4)) + [0],
                "TVBUNDLE_ID": [25, 25, 25, 25, 25],
                "REF_CSA_ID": [1, 2, 3, 4, 5],
                "CHANNEL_ID": [0, 0, 0, 0, 0],
                "AFFINITY": ["Action & Aventure", "Action & Aventure", "Action & Aventure",
                             "Famille", "Horreur"],
                "reco_origin": ["external_rating", "external_rating", "avg_watch_duration", "external_rating",
                                "external_rating"],
                "best_rank": [1, 2, 1, 1, 1],
                "start_rank": [0, 0, 0, 0, 0],
                "USER_AFFINITY_RANK": [0, 0, 0, 2, 0],
                "total_rebroadcast": [2, 1, 0, 1, 2]}
        reco_df_mocked = create_spark_df_from_data(self.job.spark, data)
        n_recos = 5
        user_cha_df = mock_user_channels(self.job.spark)
        user_csa_pref_df = create_spark_df_from_data(self.job.spark, {"USER_ID": [0], "max_csa_id": [1]})

        selected_recos = self.job.select_among_categories(
            reco_df_mocked, user_cha_df, user_csa_pref_df, n_recos, n_recos, True
        )

        # CSA rating is set to a minimum, we should only retrieve prog_id=0
        rows = selected_recos.collect()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].PROGRAM_ID, 0)

    def test_banned_kinds_are_banned(self):
        data = {"USER_ID": [0, 0, 0, 0, 0],
                "PROGRAM_ID": [0, 1, 2, 3, 4],
                "CHANNEL_ID": [0, 0, 0, 0, 0],
                "EPISODE_ID": [0, 1, 2, 3, 4],
                "REF_CSA_ID": [0, 1, 1, 1, 1],
                "AFFINITY": ["Information & Politique", "Divertissement", "Investigation & Reportage",
                             "Talk shows", "Enfants"],
                "reco_origin": ["external_rating", "external_rating", "avg_watch_duration", "external_rating",
                                "external_rating"],
                "best_rank": [1, 2, 1, 1, 2],
                "start_rank": [0, 0, 0, 0, 0],
                "USER_AFFINITY_RANK": [0, 0, 0, 2, 2],
                "total_rebroadcast": [2, 1, 0, 1, 0]}
        reco_df_mocked = create_spark_df_from_data(self.job.spark, data)
        user_channels_df = mock_user_channels(self.job.spark)
        selected_recos = self.job.select_among_categories(reco_df_mocked, user_channels_df, self.user_csa_perf_df,
                                                          5, 5, True)

        # All programs belong to banned categories and thus nothing should remain.
        rows = selected_recos.collect()
        self.assertEqual(len(rows), 0)

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                 build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(spark),
                                 load_snowflake_query_df=lambda spark, options, query:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1})
                                 if "person" in query else mock_free_channels_query(spark)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(spark),
                                     load_snowflake_table=multiplex_mock):
                self.job.launch()

    def test_general_reco_addition(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                 build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(spark),
                                 load_snowflake_query_df=lambda spark, options, query:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1})
                                 if "person" in query else mock_free_channels_query(spark)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                reco_df_mocked = mock_full_reco_df(self.job.spark, user_id=1)
                general_reco_df_mocked = mock_full_reco_df(self.job.spark, prog_ids=[4, 0, 6, 5, 8], user_id=1,
                                                           channel_ids=[45] * 5)

                user_cha_df = mock_user_channels(self.job.spark)
                user_watch_history = self.job.build_user_watch_history(self.job.spark, self.job.options, self.job.now)
                selected_recos = self.job.select_among_categories(reco_df_mocked, user_cha_df, self.user_csa_perf_df,
                                                                  2, 2, True)

                default_recos_df = self.job.select_among_categories(
                    general_reco_df_mocked, user_cha_df, self.user_csa_perf_df, 5, 5, True
                )

                rows = selected_recos.collect()
                self.assertSetEqual(set([r.PROGRAM_ID for r in rows]), {0, 2})
                rows = default_recos_df.collect()
                self.assertSetEqual(set([r.PROGRAM_ID for r in rows]), {4, 0, 6, 5, 8})

                user_to_hash_df = create_spark_df_from_data(self.job.spark, {"USER_ID": [1], "HASH_ID": [1]})
                user_bookmarks = self.job.build_user_bookmarks()
                rez = complete_recos(selected_recos, default_recos_df, user_watch_history, user_bookmarks, user_to_hash_df, 5)
                rows = rez.collect()
                self.assertSetEqual({r.PROGRAM_ID for r in rows[0].recommendations}, {2, 4, 6, 5, 8})

    def test_general_reco_addition_default_reco_only(self):
        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=multiplex_mock,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                 build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(spark),
                                 load_snowflake_query_df=lambda spark, options, query:
                                 create_spark_df_from_data(spark, {"PROGRAM_ID": [1], "total_celeb_points": 1})
                                 if "person" in query else mock_free_channels_query(spark)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                reco_df_mocked = mock_full_reco_df(self.job.spark, user_id=1)
                general_reco_df_mocked = mock_full_reco_df(self.job.spark, prog_ids=[12344444, 12344445, -1, -2, -3],
                                                           user_id=0, channel_ids=[25] * 5)

                user_cha_df = mock_user_channels(self.job.spark)
                user_watch_history = self.job.build_user_watch_history(self.job.spark, self.job.options, self.job.now)
                selected_recos = self.job.select_among_categories(
                    reco_df_mocked, user_cha_df, self.user_csa_perf_df, 2, 2, True
                )

                # The "default" reco
                generic_recos_df = self.job.select_among_categories(
                    general_reco_df_mocked, user_cha_df, self.user_csa_perf_df, 5, 5, True
                )

                # No match for user 123456 because no bundle
                user_to_hash_df = create_spark_df_from_data(self.job.spark, {"USER_ID": [12345], "HASH_ID": [123]})
                user_bookmarks = self.job.build_user_bookmarks()
                rez = complete_recos(selected_recos, generic_recos_df, user_watch_history, user_bookmarks,
                                     user_to_hash_df, 5)
                rows = rez.collect()

                # Success is met when the default reco prog_id is in the final list
                self.assertIn(12344444, {r.PROGRAM_ID for r in rows[0].recommendations})

    def test_generic_reco_creation(self):
        user_cha_df = mock_user_channels(self.job.spark)

        data = {
            "PROGRAM_ID": [0, 1, 2, 3],
            "EPISODE_ID": [0, 1, 2, 3],
            "CHANNEL_ID": [45, 45, 45, 0],
            "REF_CSA_ID": [0, 0, 0, 0],
            "nb_likes": [0, 0, 0, 0],
            "total_celeb_points": [30, 0, 0, 0],
            "external_rating": [0, 0, 0, 4.0],
            "total_rebroadcast": [1, 0, 0, 0],
            "AFFINITY": ["Action", "Action", "Action", "Action"]
        }
        prog_df = create_spark_df_from_data(self.job.spark, data)
        reco_df, id_to_hash_df, _ = self.job.build_best_of_bundle(self.job.spark, prog_df, user_cha_df)
        rows = reco_df.collect()
        self.assertNotIn(None, {r.USER_ID for r in rows})
        self.assertIn(3, {r.PROGRAM_ID for r in rows})

        # prog_id 3 should not be in results because we don't have the bundle
        no_bundle_df = create_spark_df_from_data(self.job.spark, {"USER_ID": [1234], "CHANNEL_ID": [12]})
        reco_df, id_to_hash_df, _ = self.job.build_best_of_bundle(self.job.spark, prog_df, no_bundle_df)
        rows = reco_df.collect()
        self.assertNotIn(3, set([r.PROGRAM_ID for r in rows]))

    def test_premium_reco_creation(self):
        data = {
            "PROGRAM_ID": [0, 1, 2, 3],
            "EPISODE_ID": [0, 1, 2, 3],
            "CHANNEL_ID": [45, 45, 45, 126],
            "REF_CSA_ID": [0, 0, 0, 0],
            "nb_likes": [0, 0, 0, 0],
            "total_celeb_points": [30, 0, 0, 0],
            "external_rating": [0, 0, 0, 4.0],
            "total_rebroadcast": [1, 0, 0, 0],
            "AFFINITY": ["Action", "Action", "Action", "Action"]
        }
        prog_df = create_spark_df_from_data(self.job.spark, data)
        bundle_df = create_spark_df_from_data(self.job.spark, {"USER_ID": [1234], "CHANNEL_ID": [126]})
        reco_df, id_to_hash_df, _ = self.job.build_best_of_bundle(self.job.spark, prog_df, bundle_df, False)
        rows = reco_df.collect()
        self.assertEqual(len(rows), 1)
        self.assertIn(3, set([r.PROGRAM_ID for r in rows]))

    def test_write_additional_bundle_recos(self):
        # The user id column needs to be the hash of the tvbundles available
        data = {"USER_ID": [883060535], "PROGRAM_ID": [1], "AFFINITY": ["truc"], "reco_origin": ["reco"],
                "ranking": [1], "rating": [1], "CHANNEL_ID": [134], "EPISODE_ID": [1]}
        bundle_recos = create_spark_df_from_data(self.job.spark, data)

        def run(tvbundle_ids, channel_ids):
            rel_tvbundle_cha_df = create_spark_df_from_data(self.job.spark, {"TVBUNDLE_ID": tvbundle_ids,
                                                                             "CHANNEL_ID": channel_ids})
            with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                     write_df_to_snowflake=lambda df, *args, **kwargs: None,
                                     load_snowflake_table=lambda spark, *args, **kwargs: rel_tvbundle_cha_df):
                with mock.patch.multiple("databricks_jobs.db_common",
                                         load_snowflake_table=lambda spark, *args, **kwargs: rel_tvbundle_cha_df,
                                         build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(
                                             spark)):
                    df = self.job.write_additional_bundle_recos(bundle_recos)
                    rows = df.collect()
                    return rows

        # 1 - Should get user corresponding to M+
        rows = run([90], [134])
        self.assertEqual(len(rows), 1)
        user_rows = find_user_rows(rows, -1)
        self.assertEqual(len(user_rows), 1)

        # 2 - Should get user corresponding to MExt
        rows = run([26], [134])
        self.assertEqual(len(rows), 1)
        user_rows = find_user_rows(rows, -2)
        self.assertEqual(len(user_rows), 1)

        # 3 - Hash id not matching = 0 recos
        rows = run([111111111], [33333])
        self.assertEqual(len(rows), 0)

        # 4 - Hash id matching on cha=134 only = 1 reco
        rows = run([90, 26], [2345, 134])
        self.assertEqual(len(rows), 1)
        user_rows = find_user_rows(rows, -2)
        self.assertEqual(len(user_rows), 1)

    def test_intermediate_write(self):
        ids = [1, 1, 2, 2]
        data = {
            "PROGRAM_ID": ids,
            "best_rank": ids,
            "reco_origin": ["reco" for _ in ids],
            "total_rebroadcast": [0 for _ in ids],
            "START_AT": [datetime.datetime.now() for _ in ids],
            "AFFINITY": ["Action" for _ in ids],
            "CHANNEL_ID": [45, 46, 45, 46]
        }
        top_k_simple_df = create_spark_df_from_data(self.job.spark, data)
        data = {
            "TVBUNDLE_ID": [26, 90, 26],
            "CHANNEL_ID": [45, 45, 46]
        }
        rel_tvbundle_cha_df = create_spark_df_from_data(self.job.spark, data)

        with mock.patch.multiple("databricks_jobs.jobs.personalised_reco_jobs.popular_reco_job.entrypoint",
                                 load_snowflake_table=lambda spark, *args, **kwargs: rel_tvbundle_cha_df,
                                 write_df_to_snowflake=lambda df, *args, **kwargs: None):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     build_product_to_tv_bundle=lambda spark, options: mock_product_to_tvbundle(spark),
                                     load_snowflake_table=lambda spark, *args, **kwargs: rel_tvbundle_cha_df):
                bundle_rez, aff_rez = self.job.save_intermediary_results(self.job.spark, self.job.write_options,
                                                                         top_k_simple_df)

        bundle_rez.show()
        aff_rez.show()

        rows = bundle_rez.collect()
        mext_rows = find_user_rows(rows, "EXTENDED", "EQUIVALENCE_CODE")
        mplus_rows = find_user_rows(rows, "OPTION_100H", "EQUIVALENCE_CODE")

        self.assertSetEqual({r.PROGRAM_ID for r in mext_rows}, {r.PROGRAM_ID for r in mplus_rows})

        rows = aff_rez.collect()
        action_rows = find_user_rows(rows, "Action", "AFFINITY")
        self.assertEqual(len(action_rows), 2)


class TestPopRecoUDF(TestCase):
    def setUp(self):
        # We need some python black magic in order to be able to mock the decorator of an already imported
        # page : add a patch for the decorator and reload the module, add a cleanup phase
        def kill_patches():
            patch.stopall()
            importlib.reload(popular_reco_utils)

        self.addCleanup(kill_patches)

        # We patch the decorator as a decorator doing nothing to the function provided
        mock.patch('databricks_jobs.jobs.utils.spark_utils.typed_udf', lambda *args, **kwargs: lambda x: x).start()
        # Reloads the module which applies our patched decorator
        importlib.reload(popular_reco_utils)

    def test_no_watch_history(self):
        MyRow = namedtuple("MyRow", ["USER_ID", "PROGRAM_ID", "reco_origin", "ranking"])
        rows = [MyRow(1, 1, "", 3), MyRow(1, 2, "", 6)]
        general_reco = [MyRow(0, 1, "", 3), MyRow(0, 3, "", 6), MyRow(0, 5, "", 1)]
        rez = popular_reco_utils.merge_udf(rows, general_reco, [], [], 3)

        self.assertEqual(len(rez), 3)
        self.assertSetEqual({r.PROGRAM_ID for r in rez}, {1, 2, 5})

    def test_with_watch_history(self):
        MyRow = namedtuple("MyRow", ["USER_ID", "PROGRAM_ID", "reco_origin", "ranking"])
        rows = [MyRow(1, 1, "", 3), MyRow(1, 2, "", 6)]
        general_reco = [MyRow(0, 1, "", 3), MyRow(0, 3, "", 6), MyRow(0, 5, "", 1)]
        already_seen = [1, 5]
        already_booked = [2]
        rez = popular_reco_utils.merge_udf(rows, general_reco, already_seen, already_booked, 3)

        self.assertEqual(len(rez), 1)
        self.assertSetEqual({r.PROGRAM_ID for r in rez}, {3})
