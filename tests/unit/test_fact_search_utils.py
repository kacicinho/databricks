from unittest import TestCase
from datetime import datetime

from databricks_jobs.jobs.fact_table_build_jobs.fact_search_job.query_substring_remover import TimedSearch, \
    find_largest_query_strings, N_SECOND_FUTURE_SEARCH


class TestQueryDedup(TestCase):

    def test_largest_substring_basic(self):
        searches = [TimedSearch(search="aa", timestamp=2), TimedSearch(search="aab", timestamp=60)]
        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 1)
        self.assertEqual(list(rez)[0].search, "aab")

    def test_largest_substring_2_time_clusters(self):

        searches = [TimedSearch(search="aa", timestamp=2), TimedSearch(search="aab", timestamp=3),
                    TimedSearch(search="aa", timestamp=70)]

        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 2)
        searches = {x.search for x in rez}
        self.assertSetEqual(searches, {"aa", "aab"})

    def test_largest_substring_multiple_substrings(self):
        searches = [TimedSearch(search="aa", timestamp=2), TimedSearch(search="aab", timestamp=3),
                    TimedSearch(search="aac", timestamp=5)]

        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 2)
        searches = {x.search for x in rez}
        self.assertSetEqual(searches, {"aac", "aab"})

    def test_largest_substring_older_smaller_query(self):
        searches = [TimedSearch(search="aa", timestamp=13), TimedSearch(search="aab", timestamp=3)]

        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 2)
        searches = {x.search for x in rez}
        # Both are kept because : the substring happened after the larger query, this is a new query
        self.assertSetEqual(searches, {"aa", "aab"})

    def test_same_timestamp_same_search(self):
        searches = [TimedSearch(search='ab', timestamp=1627976569), TimedSearch(search='ab', timestamp=1627976569),
                    TimedSearch(search='ab', timestamp=1627976569)]
        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 1)

    def test_empty_search(self):
        searches = []
        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 0)

    def test_none_search(self):
        searches = None
        rez = find_largest_query_strings(searches)

        self.assertEqual(len(rez), 0)

    def test_real_world_swap_example(self):
        t1 = datetime(2021, 8, 3, 15, 19, 3, 365).timestamp()
        t2 = datetime(2021, 8, 3, 15, 18, 58, 359).timestamp()

        searches = [TimedSearch(search='JO escalad', timestamp=t2), TimedSearch(search='JO escal', timestamp=t1)]

        rez = find_largest_query_strings(searches)
        self.assertEqual(len(rez), 1)

    def test_real_world_no_swap_example(self):
        t1 = datetime(2021, 8, 3, 15, 19, 0 + N_SECOND_FUTURE_SEARCH + 1, 359).timestamp()
        t2 = datetime(2021, 8, 3, 15, 19, 0, 359).timestamp()

        searches = [TimedSearch(search='JO escalad', timestamp=t2), TimedSearch(search='JO escal', timestamp=t1)]

        rez = find_largest_query_strings(searches)
        self.assertEqual(len(rez), 2)
