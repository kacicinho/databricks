import os

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window

from databricks_jobs.jobs.utils.utils import load_snowflake_table, load_snowflake_query_df, get_pypsark_hash_function, \
    dump_dataframe_to_csv, format_tuple_for_sql

# Kinds always banned : meteo (52), adult films (83, 34)
BANNED_KIND_IDS = [83, 34, 52]
# Unfit for reco : information & politique (43, 45, 46) et Talk shows (97), Investigation (127)
UNFIT_KIND_IDS = [43, 45, 46, 97, 127]
# Unfit for reco : Information (4), Divertissement (5), Indéterminé (7)
UNFIT_CATEGORY_IDS = [4, 5, 7]
# Free channels from TNT bundle
FREE_BUNDLE_CHANNELS = [
    2, 5, 8, 9, 12, 18, 19, 26, 34, 35, 36, 38, 40, 42, 44, 45, 46, 48, 49, 51, 56, 58, 62, 67, 69, 131, 133, 136, 159,
    223, 227, 234, 248, 254, 314, 321, 329, 330, 331, 332, 333, 334, 335, 336, 337, 340, 371, 374, 375, 377, 378, 381,
    390, 420, 421, 422, 423, 424, 467, 516, 517, 554, 565, 566, 567, 568, 569, 596, 597, 598, 614, 628, 629, 636
]

# Mango channel
MANGO_CHANNELS = [
    329, 330, 331, 332, 333, 334, 335, 336, 337, 340, 343, 374, 375, 377, 378, 381, 390, 420, 421, 422,
    423, 424, 425, 467, 516, 517, 554, 565, 567, 596, 597, 598, 614, 628, 629, 636, 648
]


def reco_worthy_filtering(df):
    """
    Filtering is done on kinds and categories based on BANNED_KIND_IDS, UNFIT_KIND_IDS and UNFIT_CATEGORY_IDS
    """
    return (~df.REF_PROGRAM_CATEGORY_ID.isin(*UNFIT_CATEGORY_IDS)) & (~df.REF_PROGRAM_KIND_ID.isin(*UNFIT_KIND_IDS)) & \
           (~df.REF_PROGRAM_KIND_ID.isin(*BANNED_KIND_IDS))


def build_episode_df(spark, options):
    episode_df = load_snowflake_table(spark, options, "backend.episode")
    season_df = load_snowflake_table(spark, options, "backend.season")

    # Build the table to get episode number and season number
    episode_info_df = episode_df. \
        join(season_df, episode_df.SEASON_ID == season_df.ID). \
        withColumn("EPISODE_ID", episode_df.ID). \
        withColumn("SEASON_NUMBER", season_df.NUMBER). \
        withColumn("EPISODE_NUMBER", episode_df.NUMBER). \
        select("EPISODE_ID", "SEASON_NUMBER", "EPISODE_NUMBER", episode_df.PROGRAM_ID, "TITLE_ORIGINAL", "DURATION",
               "REF_CSA_ID")

    return episode_info_df


def build_broadcast_df_with_episode_info(spark, options, episode_info_df, start, end, min_duration_in_mins=15,
                                         free_bundle=False, filter_kind=True):
    # 0 : define data source and function
    broadcast_df = load_snowflake_table(spark, options, "backend.broadcast")
    program_df = load_snowflake_table(spark, options, "backend.program")

    # Define what is available next
    available_next = ((col("START_AT") >= start) & (col("START_AT") <= end))

    correct_broadcast_df = broadcast_df. \
        join(program_df, program_df.ID == broadcast_df.PROGRAM_ID). \
        drop(program_df.DURATION). \
        join(episode_info_df, episode_info_df.EPISODE_ID == broadcast_df.EPISODE_ID, "left"). \
        drop(episode_info_df.EPISODE_ID). \
        drop(episode_info_df.PROGRAM_ID). \
        drop(episode_info_df.DURATION). \
        where(available_next). \
        where((episode_info_df.REF_CSA_ID < 5) | (episode_info_df.REF_CSA_ID.isNull())). \
        drop(program_df.ID). \
        withColumnRenamed("ID", "BROADCAST_ID")

    rel_channel_df = load_snowflake_table(spark, options, "backend.rel_tvbundle_channel")
    correct_broadcast_df = correct_broadcast_df. \
        join(rel_channel_df, correct_broadcast_df.CHANNEL_ID == rel_channel_df.CHANNEL_ID). \
        drop(rel_channel_df.CHANNEL_ID)

    if filter_kind:
        correct_broadcast_df = correct_broadcast_df. \
            where((~program_df.REF_PROGRAM_KIND_ID.isin(*BANNED_KIND_IDS)))

    if min_duration_in_mins > 0:
        correct_broadcast_df = correct_broadcast_df. \
            where(broadcast_df.DURATION > min_duration_in_mins * 60)

    if free_bundle:
        correct_broadcast_df = correct_broadcast_df. \
            where(F.col("tvbundle_id") == 25)

    return correct_broadcast_df. \
        select("PROGRAM_ID", "START_AT", "END_AT", "SEASON_NUMBER", "EPISODE_NUMBER", "CHANNEL_ID", "DURATION",
               "REF_PROGRAM_KIND_ID", "REF_PROGRAM_CATEGORY_ID", "EPISODE_ID", "PRODUCTION_YEAR", "TVBUNDLE_ID",
               episode_info_df.REF_CSA_ID)


def build_vod_df_with_episode_infos(spark, options, episode_info_df, now, delta, min_duration_in_mins=15,
                                    video_type=("vod", "VOD", "REPLAY"), allow_extra=False):
    is_available = ((now >= F.col("available_from")) & (now + delta <= F.col("available_until")))

    program_df = load_snowflake_table(spark, options, "backend.program")
    rel_tvbundle_cha_df = load_snowflake_table(spark, options, "backend.rel_tvbundle_channel")
    tvbundle_df = load_snowflake_table(spark, options, "backend.tvbundle")

    if allow_extra:
        video_filter = (F.col("VOD_FORMAT_ID") == 1) & (
            (F.col("VIDEO_TYPE").isNull()) | (F.col("VIDEO_TYPE").isin(*video_type)))
    else:
        video_filter = (F.col("VOD_FORMAT_ID") == 1) & (F.col("VIDEO_TYPE").isin(*video_type))

    vod_df = load_snowflake_table(spark, options, "backend.vod"). \
        where(is_available). \
        where((F.col("disabled") == 0) & (F.col("withdrawn_at").isNull()) & (F.col("deleted_at").isNull())). \
        where(video_filter). \
        join(episode_info_df, "EPISODE_ID"). \
        drop(episode_info_df.EPISODE_ID)

    # filter out un-commercialized bundle programs
    vod_df = vod_df. \
        join(rel_tvbundle_cha_df, rel_tvbundle_cha_df.CHANNEL_ID == vod_df.CHANNEL_ID). \
        drop(rel_tvbundle_cha_df.CHANNEL_ID). \
        join(tvbundle_df, tvbundle_df.ID == rel_tvbundle_cha_df.TVBUNDLE_ID). \
        drop(tvbundle_df.ID). \
        where("IS_COMMERCIALIZED = 1")

    # Note, we keep programs with duration=0 as it is most likely a bug
    vod_enriched_df = vod_df. \
        join(program_df, program_df.ID == vod_df.PROGRAM_ID). \
        drop(program_df.ID). \
        where((~program_df.REF_PROGRAM_KIND_ID.isin(*BANNED_KIND_IDS))). \
        where((episode_info_df.DURATION == 0) | (episode_info_df.DURATION > min_duration_in_mins * 60)). \
        where((F.col("REF_CSA_ID") < 5) | (F.col("REF_CSA_ID").isNull())). \
        select("PROGRAM_ID", F.lit(None).alias("START_AT"), F.lit(None).alias("END_AT"),
               "SEASON_NUMBER", "EPISODE_NUMBER", "CHANNEL_ID", episode_info_df.DURATION,
               "REF_PROGRAM_KIND_ID", "REF_PROGRAM_CATEGORY_ID", "EPISODE_ID", "PRODUCTION_YEAR", "TVBUNDLE_ID",
               "REF_CSA_ID")

    return vod_enriched_df


def build_product_to_tv_bundle(spark, options):
    query = \
        """
        select distinct PRODUCT_ID, bp.CODE, bp.EQUIVALENCE_CODE,
        REGEXP_SUBSTR(bpf.CODE,'TVBUNDLE_(\\\\d+)', 1, 1, 'e')  as TVBUNDLE_ID
        from backend.product as bp
        JOIN backend.rel_product_feature as brpf
        ON brpf.PRODUCT_ID = bp.ID
        JOIN backend.product_feature as bpf
        ON brpf.PRODUCT_FEATURE_ID = bpf.ID
        where  bpf.CODE like 'TVBUNDLE_%'
        """
    return load_snowflake_query_df(spark, options, query)


def build_equivalence_code_to_channels(spark, options):
    product_to_tv_bundle_df = build_product_to_tv_bundle(spark, options)
    equ_code_to_bundle = product_to_tv_bundle_df. \
        select("EQUIVALENCE_CODE", "TVBUNDLE_ID"). \
        distinct()
    rel_tvbundle_cha_df = load_snowflake_table(spark, options, "backend.rel_tvbundle_channel")
    return equ_code_to_bundle. \
        join(rel_tvbundle_cha_df, equ_code_to_bundle.TVBUNDLE_ID == rel_tvbundle_cha_df.TVBUNDLE_ID). \
        drop(rel_tvbundle_cha_df.TVBUNDLE_ID). \
        select("EQUIVALENCE_CODE", "TVBUNDLE_ID", "CHANNEL_ID"). \
        distinct()


def build_user_to_paying_tv_bundle(spark, options, prod_to_bundle_df, now):
    subscribers_df = load_snowflake_table(spark, options, "backend.subscribers")

    valid_subs_df = subscribers_df. \
        select("USER_ID", "EQUIVALENCE_CODE", "EXPIRES_AT_DATE"). \
        groupBy("USER_ID", "EQUIVALENCE_CODE"). \
        agg(F.max("EXPIRES_AT_DATE").alias("max_date_per_offer")). \
        where((col("max_date_per_offer") > now) | (col("max_date_per_offer").isNull()))

    tv_bundle_per_user = valid_subs_df. \
        join(prod_to_bundle_df, valid_subs_df.EQUIVALENCE_CODE == prod_to_bundle_df.EQUIVALENCE_CODE). \
        select("USER_ID", "TVBUNDLE_ID"). \
        distinct()

    return tv_bundle_per_user


def build_channel_order_df(spark, options):
    """
    This function patches some missing data in the backend.channel table

    We need to add the channel order of NRJ and Cherie and remove france O
    """
    channel_info = load_snowflake_table(spark, options, "backend.channel"). \
        where(F.col("FRENCH_TNT_RANK").isNotNull()). \
        where(F.col("ID") != 43)

    missing_data = {
        "NAME": ["NRJ 12", "Cherie 25"],
        "ID": [26, 9],
        "FRENCH_TNT_RANK": [12, 25]
    }
    missing_channel_info = spark.createDataFrame(pd.DataFrame(missing_data))

    return channel_info. \
        select(missing_channel_info.columns). \
        union(missing_channel_info). \
        withColumn("RAIL_ORDER", F.row_number().over(Window.orderBy("FRENCH_TNT_RANK"))). \
        drop("FRENCH_TNT_RANK")


def build_full_program_with_infos(spark, options):
    """
    Select informations for all available programs :
    Program with description
    - description from edito_program when available (more consistent) otherwise takes description from backend.program
    Program with tags
    - select programs top tags : all_tags if number of episodes = 1 , rank < 20 if number of episodes > 1
    """
    program_df = load_snowflake_table(spark, options, "backend.program")
    episode_df = load_snowflake_table(spark, options, "backend.episode")
    program_edito_df = load_snowflake_table(spark, options, "backend.edito_program")
    category_df = load_snowflake_table(spark, options, "backend.ref_program_category")
    kind_df = load_snowflake_table(spark, options, "backend.ref_program_kind")
    tag_df = load_snowflake_table(spark, options, "backend.rel_episode_tag")

    program_with_description_df = program_df. \
        join(program_edito_df, program_df.ID == program_edito_df.PROGRAM_ID). \
        drop(program_edito_df.PROGRAM_ID). \
        withColumn("REAL_SUMMARY",
                   F.coalesce(program_edito_df.SUMMARY, program_df.SUMMARY)). \
        select(program_df.ID, program_df.REF_PROGRAM_CATEGORY_ID, program_df.REF_PROGRAM_KIND_ID,
               program_df.TITLE, "REAL_SUMMARY"). \
        withColumnRenamed("ID", "PROGRAM_ID"). \
        withColumnRenamed("REAL_SUMMARY", "SUMMARY")

    program_with_tags_df = program_df. \
        join(episode_df, program_df.ID == episode_df.PROGRAM_ID). \
        drop(episode_df.PROGRAM_ID). \
        join(tag_df, tag_df.EPISODE_ID == episode_df.ID, "left"). \
        groupby(program_df.ID, "REF_TAG_ID"). \
        agg(F.countDistinct(episode_df.ID).alias("OCCUR")). \
        withColumn("rank", F.rank().over(Window.partitionBy(program_df.ID).orderBy(F.desc("OCCUR")))). \
        where("rank <= 20"). \
        groupby(program_df.ID). \
        agg(F.collect_list("REF_TAG_ID").alias("TAGS")). \
        withColumn("N_TAGS", F.size("TAGS")). \
        select(program_df.ID, "TAGS", "N_TAGS"). \
        withColumnRenamed("ID", "PROGRAM_ID")

    program_with_full_infos_df = program_with_description_df. \
        join(program_with_tags_df, program_with_description_df.PROGRAM_ID == program_with_tags_df.PROGRAM_ID). \
        drop(program_with_tags_df.PROGRAM_ID). \
        join(category_df, category_df.ID == program_with_description_df.REF_PROGRAM_CATEGORY_ID). \
        withColumnRenamed("NAME", "CATEGORY"). \
        join(kind_df, kind_df.ID == program_with_description_df.REF_PROGRAM_KIND_ID). \
        withColumnRenamed("NAME", "KIND"). \
        where((~program_with_description_df.REF_PROGRAM_KIND_ID.isin(*BANNED_KIND_IDS))). \
        select("PROGRAM_ID", "REF_PROGRAM_CATEGORY_ID", "CATEGORY", "REF_PROGRAM_KIND_ID", "KIND", "TITLE", "SUMMARY",
               "TAGS", "N_TAGS")

    return program_with_full_infos_df


def prepare_new_user_optout_csv(spark, options, dbutils, dump_folder, start_date, end_date):
    """
    Query new opted-out users since last dump.
    """
    # 1 - Query new optout users since last dump
    query_new_user_optout = \
        """
        SELECT
            us.ID
        FROM BACKEND.USER_RAW us
        JOIN DW.FACT_CMP_USER_CONSENTS cs
        ON
            us.ID = cs.USER_ID
        WHERE
            us.UPDATED_AT >= '{0}' and us.UPDATED_AT < '{1}' AND
            TRANSFORM_SUBSCRIPTION_EMAIL_MOLOTOV(EMAIL) = FALSE AND
            (CONSENT_ENABLED(purposes:enabled, array_construct(1,2,3,4,5,6,7,8,9,10)) = FALSE OR
            CONSENT_ENABLED(specialfeatures:enabled, array_construct(1,2)) = FALSE OR
            CONSENT_ENABLED(custompurposes:enabled, array_construct(1,2)) = FALSE OR
            CONSENT_ENABLED(vendorsconsent:enabled, array_construct(97)) = FALSE)
        """.format(start_date, end_date)
    user_new_optout_df = load_snowflake_query_df(spark, options, query_new_user_optout)

    # 3 - Extract only the id of the users
    final_df = user_new_optout_df. \
        withColumn("external_id", get_pypsark_hash_function()). \
        select("external_id")

    # 4 - Dump the result to csv
    csv_dir_path = os.path.join(dump_folder, "user_optout_dump")
    temporary_csv_path = dump_dataframe_to_csv(dbutils, final_df, csv_dir_path)
    return temporary_csv_path


def build_keep_person_id(spark, options):
    """
    Query unwanted personalities.
    """

    # - Keep only person id with biography/ picture
    # - Remove dictators (person_function_id 122)
    # - Remove top person_ids that were present at least 10x in the past 30 days in user 0 reco
    query_keep_person_id = \
        """
        WITH blacklist_person AS (
            SELECT
                p.id
            FROM BACKEND.REL_EPISODE_PERSON rel
            JOIN BACKEND.REF_PERSON_FUNCTION f
            ON
                f.id = rel.ref_person_function_id
            JOIN BACKEND.PERSON p
            ON
                p.id = rel.person_id
            WHERE
                p.biography is null or p.picture_hash is null or f.id = 122
            GROUP BY 1)
            SELECT
                ID
            FROM BACKEND.PERSON
            WHERE
                ID NOT IN (SELECT ID FROM blacklist_person)
            GROUP BY 1
        """
    return load_snowflake_query_df(spark, options, query_keep_person_id)


def build_tvbundle_hashes(user_to_tvbundle_df):
    # 2 - Compute all bundle combinations
    user_to_tvbundle_df = user_to_tvbundle_df. \
        withColumn("HAS_OPTION", F.lit(1))

    user_id_to_hash = user_to_tvbundle_df. \
        groupBy("USER_ID"). \
        agg(F.sort_array(F.collect_set("CHANNEL_ID")).alias("available_channels")). \
        withColumn("HASH_ID", F.hash("available_channels"))

    # 3 - Create the hacked bundle_hash_to_tvbundle_id
    hash_to_bundle = user_id_to_hash.alias("hash"). \
        join(user_to_tvbundle_df.alias("tvbundle"), F.col("hash.USER_ID") == F.col("tvbundle.USER_ID")). \
        select("HASH_ID", "tvbundle.CHANNEL_ID"). \
        distinct(). \
        withColumnRenamed("HASH_ID", "USER_ID")

    return hash_to_bundle, user_id_to_hash


def build_channel_group_df(spark, options, target_channel_group):
    """
    This allow to understand if channels are in the same owner group : example M6
    """
    channel_df = load_snowflake_table(spark, options, "backend.channel")
    channel_group_df = load_snowflake_table(spark, options, "backend.channel_group")
    return channel_df. \
        join(channel_group_df, channel_group_df.ID == channel_df.GROUP_ID). \
        where(channel_group_df.ID == target_channel_group). \
        select(channel_df.ID)


def build_user_to_allowed_channels(spark, options, tvbundle_per_user,
                                   authorized_bundles=(25, 90, 26, 31, 60, 134, 142, 146)):
    rel_tvbundle_cha_df = load_snowflake_table(spark, options, "backend.rel_tvbundle_channel")
    return tvbundle_per_user. \
        join(rel_tvbundle_cha_df, rel_tvbundle_cha_df.TVBUNDLE_ID == tvbundle_per_user.TVBUNDLE_ID). \
        drop(rel_tvbundle_cha_df.TVBUNDLE_ID). \
        where(f"TVBUNDLE_ID in {format_tuple_for_sql(authorized_bundles)}"). \
        select("USER_ID", "CHANNEL_ID"). \
        distinct()


def build_user_csa_preference(spark, options):
    """
    ref_profile_setting_type_id = 6 correspond to csa limit per profile
    1: TP, 2: -10, 3: -12, 4: -16, 5: -18
    The default value is 4

    max_csa_id is defined as the maximum in order to avoid breaking one of the profile csa rule
    """
    profile_pref_df = load_snowflake_table(spark, options, "backend.profile_setting")
    profile_df = load_snowflake_table(spark, options, "backend.user_profile")
    return profile_df. \
        join(profile_pref_df, profile_pref_df.PROFILE_ID == profile_df.ID, "left"). \
        where("ref_profile_setting_type_id = 6"). \
        groupBy("USER_ID"). \
        agg(F.min("setting_value").alias("max_csa_id")). \
        fillna(4, subset=["max_csa_id"])
