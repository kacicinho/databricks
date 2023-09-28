import datetime
from typing import NamedTuple, List
from databricks_jobs.jobs.utils.spark_utils import file_exists

# start day for backup
USER_START_BACKUP_DAY = datetime.datetime.strptime('2014-01-01', '%Y-%m-%d').date()


def date_range(start, end):
    date_list = [start]
    a_date = start
    while a_date < end:
        a_date += datetime.timedelta(days=1)
        date_list.append(a_date)
    return date_list


class LogPath(NamedTuple):
    path: str

    @property
    def date(self):
        return datetime.datetime.strptime(self.path.replace('/', ''), '%Y-%m-%d').date()

    @classmethod
    def from_args(cls, *args, sep="/"):
        return cls(sep.join(args))


class LogDB(NamedTuple):
    db: List[LogPath]

    @classmethod
    def parse_dbfs_path(cls, dbutils, path):
        if file_exists(path, dbutils):
            return cls([LogPath(x.name) for x in dbutils.fs.ls(path)])
        else:
            return cls([])

    @property
    def last_dump_date(self):
        return max(map(lambda x: x.date, self.db)) if len(self.db) > 0 else USER_START_BACKUP_DAY

    @property
    def first_dump_date(self):
        return min(map(lambda x: x.date, self.db)) if len(self.db) > 0 else USER_START_BACKUP_DAY

    @property
    def get_missing_date(self):
        return set(date_range(self.first_dump_date, self.last_dump_date)).difference(set([x.date for x in self.db]))
