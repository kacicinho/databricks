from typing import NamedTuple, List


db_table_schema = {
    "channel_prog_schema":
        {"fields": [
            {"name": "CHANNEL_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "PROGRAM_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RECOMMENDATIONS", "type": "TEXT", "constraint": ""},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["CHANNEL_ID", "PROGRAM_ID"]},
    "prog_schema":
        {"fields": [
            {"name": "PROGRAM_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RECOMMENDATIONS", "type": "TEXT", "constraint": ""},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["PROGRAM_ID"]},
    "prog_best_rate_schema":
        {"fields": [
            {"name": "PROGRAM_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RATING", "type": "FLOAT", "constraint": "NOT NULL"},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["PROGRAM_ID"]},
    "channel_prog_episode_schema":
        {"fields": [
            {"name": "CHANNEL_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "PROGRAM_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "EPISODE_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RECOMMENDATIONS", "type": "TEXT", "constraint": ""},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["CHANNEL_ID", "PROGRAM_ID", "EPISODE_ID"]},
    "channel_schema":
        {"fields": [
            {"name": "CHANNEL_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RECOMMENDATIONS", "type": "TEXT", "constraint": ""},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["CHANNEL_ID"]},
    "user_schema":
        {"fields": [
            {"name": "USER_ID", "type": "BIGINT", "constraint": "NOT NULL"},
            {"name": "RECOMMENDATIONS", "type": "TEXT", "constraint": ""},
            {"name": "UPDATE_DATE", "type": "DATE", "constraint": "NOT NULL"}],
            "keys": ["USER_ID"]}
}


class SnowflakeToRdsLookup(NamedTuple):
    table_name: str
    schema: str

    @property
    def get_schema(self):
        return db_table_schema[self.schema]


class DbLookupList(NamedTuple):
    lookup_list: List[SnowflakeToRdsLookup]

    def get_lookup(self, table_name):
        tmp = [x for x in self.lookup_list if x.table_name in table_name]
        return tmp[0] if len(tmp) == 1 else None


db_lookup_list = DbLookupList([
    SnowflakeToRdsLookup("RECO_CHANNEL_PROG_PROGS_META_LATEST", "channel_prog_schema"),
    SnowflakeToRdsLookup("RECO_PROG_PROGS_META_LATEST", "prog_schema"),
    SnowflakeToRdsLookup("RECO_PROG_PROGS_PERSO_LATEST", "prog_schema"),
    SnowflakeToRdsLookup("RECO_PROG_PROGS_POPULAR_LATEST", "prog_schema"),
    SnowflakeToRdsLookup("RECO_CHANNEL_PROGRAM_EPISODE_USERS_LATEST", "channel_prog_episode_schema"),
    SnowflakeToRdsLookup("RECO_CHANNEL_TAGS_LATEST", "channel_schema"),
    SnowflakeToRdsLookup("RECO_USER_CATEGORYS_LATEST", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_THIS_WEEK_LATEST", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_THIS_DAY_LATEST", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_TVBUNDLES_LATEST", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_ADULTSWIM_LATEST_v2", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_CINEPLUS_LATEST_v2", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_OCS_LATEST_v2", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PERSON_LATEST", "user_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_THIS_WEEK_BEST_RATE_LATEST", "prog_best_rate_schema"),
    SnowflakeToRdsLookup("RECO_USER_PROGS_THIS_DAY_BEST_RATE_LATEST", "prog_best_rate_schema")
])
