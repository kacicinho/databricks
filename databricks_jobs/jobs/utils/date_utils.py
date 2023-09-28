import math
from datetime import datetime


def build_timestamp_ceiler(granularity):
    def date_ceiler(dt):
        d = math.ceil(dt.timestamp() / granularity) * granularity
        return d

    return date_ceiler


def build_datetime_ceiler(granularity):
    def date_ceiler(dt):
        d = math.ceil(dt.timestamp() / granularity) * granularity
        return datetime.fromtimestamp(d)

    return date_ceiler
