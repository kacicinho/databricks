import os
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df, dump_dataframe_to_csv


def get_watch_consent_df(spark, options, start, end=None, with_consent=True):
    if end is None:
        end = start
    # Extract yesterday watch data
    query_channel_watch_duration = \
        """
        SELECT
            COALESCE(TRY_TO_NUMBER(
                    COALESCE(json_data:context:traits:userId::varchar,
                                json_data:context:traits:molotov_id,
                                json_data:userId::varchar)),
                                null::number) as USER_ID,
            TRY_TO_TIMESTAMP_LTZ(JSON_DATA:properties:min_client_offset::varchar) as WATCH_START,
            MAX(TRY_TO_TIMESTAMP_LTZ(JSON_DATA:properties:client_offset::varchar)) as WATCH_END,
            MAX(JSON_DATA:properties:channel_id::varchar) as CHANNEL_ID
        FROM
            RAW.PUBLIC.SEGMENT
        WHERE
            JSON_DATA:properties:asset_type::VARCHAR = 'live' AND
            event_name = 'watch_stopped' AND
            DATE(received_at) BETWEEN '{start}' AND '{end}'
        GROUP BY
            1,2
        HAVING
            COUNT(DISTINCT JSON_DATA:properties:channel_id::varchar) = 1 AND
            MAX(JSON_DATA:properties:seek_count) = 0
            AND DATEDIFF('seconds', watch_start, watch_end) > 0
        """.format(start=start, end=end)
    watch_df = load_snowflake_query_df(spark, options, query_channel_watch_duration)

    query_consent_users = \
        """
        SELECT
            USER_ID
         from dw.fact_cmp_user_consents cs
         WHERE
             CONSENT_ENABLED(purposes:enabled, array_construct(1,2,3,4,5,6,7,8,9,10)) AND
             CONSENT_ENABLED(specialfeatures:enabled, array_construct(1,2)) AND
             CONSENT_ENABLED(custompurposes:enabled, array_construct(1,2)) AND
             CONSENT_ENABLED(vendorsconsent:enabled, array_construct(97))
         """
    consent_users = load_snowflake_query_df(spark, options, query_consent_users)

    return watch_df.\
        join(consent_users, consent_users.USER_ID == watch_df.USER_ID).\
        drop(consent_users.USER_ID) if with_consent else watch_df


def prepare_lookup_channel_csv(spark, options, dump_folder, dbutils):
    """Update the channel_id lookup table"""

    # 1 - Query new optin users since last dump
    query_lookup_channel = \
        """
        SELECT
            id,
            display_name
        FROM PROD.BACKEND.CHANNEL
        ORDER BY 1
        """
    final_df = load_snowflake_query_df(spark, options, query_lookup_channel)

    # 2 - Dump the result to csv
    csv_dir_path = os.path.join(dump_folder, "lookup_channel_dump")
    temporary_csv_path = dump_dataframe_to_csv(dbutils, final_df, csv_dir_path)
    return temporary_csv_path
