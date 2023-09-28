from datetime import datetime
from typing import NamedTuple, List

from databricks_jobs.jobs.utils.spark_utils import typed_udf
from pyspark.sql.types import StringType, ArrayType, StructField, StructType, TimestampType


N_SECOND_FUTURE_SEARCH = 6


class TimedSearch(NamedTuple):

    search: str
    timestamp: int

    @classmethod
    def from_row(cls, r):
        return cls(search=r.search_query, timestamp=r.timestamp.timestamp())

    def is_time_related(self, sub_search):
        """
        Usually prefix messages arrive before the full length message
        But not always, so we need to add some extra delay : 6s
        """
        low = self.timestamp - (60 + N_SECOND_FUTURE_SEARCH)
        high = self.timestamp + N_SECOND_FUTURE_SEARCH
        return low <= sub_search.timestamp <= high

    def is_substring(self, larger):
        return self.search == larger.search[0: len(self.search)]


def find_largest_query_strings(substring_list: List[TimedSearch]):
    """
    We have two constraints in this function :
    - Time sensitivity : matching can only be done on queries close one to another (< 1min)
    - The intermediate searches are prefixes of the final search

    Implementation :
    - Words are treated with older searches first
    - A first loop over a potential kept keyword
    - A second loop in search of the prefixes
    - Final returned searches substring_list minus subqueries found at least one

    Complexity :
    We prune a lot of comparisons by exiting early on already matched searches.
    Given that we start with oldest queries first, this should reduce a lot the number of string operation needed.

    :return:
    """
    if substring_list is None or len(substring_list) < 1:
        return []

    # Why this weird sorting func : to avoid cases where some message that should arrive after arrive before
    # However they should be close enough, so the trick is to help larger search to be handled first
    substring_list = sorted(substring_list, key=lambda x: x.timestamp + N_SECOND_FUTURE_SEARCH * len(x.search),
                            reverse=True)
    already_matched = set()
    # Index of the larger query string
    i = 0

    while i < len(substring_list):

        search = substring_list[i]
        # If already matched, all subqueries have already matched by the parent query
        if i in already_matched:
            i += 1
            continue

        # Index of the compared substring
        j = i + 1

        while j < len(substring_list):
            sub_search = substring_list[j]
            # Continue if already matched as a substring or too far from original query
            if j in already_matched or (not search.is_time_related(sub_search)):
                j += 1
                continue

            # Is j a prefix of i ?
            if sub_search.is_substring(search):
                already_matched.add(j)

            j += 1
        # End of while for sub_query

        i += 1

    # We keep only the keyword that were not subqueries of another search
    return [s for i, s in enumerate(substring_list) if i not in already_matched]


struct_schema = StructType([
    StructField('search_query', StringType(), nullable=False),
    StructField('timestamp', TimestampType(), nullable=False),
])


@typed_udf(ArrayType(struct_schema))
def search_filtering(rows):
    if rows is None or len(rows) < 1:
        return []
    searches = [TimedSearch.from_row(r) for r in rows]
    return map(lambda r: (r.search, datetime.fromtimestamp(r.timestamp)),
               find_largest_query_strings(searches))
