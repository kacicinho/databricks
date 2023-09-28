import datetime
import random
from datetime import timedelta
from collections import OrderedDict

import pandas as pd
from pyspark.sql import functions as F

from databricks_jobs.jobs.utils.affinities import AFFINITIES


def create_spark_df_from_data(spark, data):
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)


def mock_reco_user_progs(spark):
    now = datetime.datetime.now()
    repeat = 5
    reco = [
        {
            "PROGRAM_ID": 16,
            "rating": 1
        },
        {
            "PROGRAM_ID": 43,
            "rating": 0.627
        }
    ]
    data = {'USER_ID': list(range(repeat)),
            'RECOMMENDATIONS': [reco] * repeat,
            'UPDATE_DATE': [now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def mock_reco_user_best_rate_ml_tables(spark):
    now = datetime.datetime.now()
    repeat = 5
    reco = 0.2
    data = {'PROGRAM_ID': list(range(repeat)),
            'RATING': [reco] * repeat,
            'UPDATE_DATE': [now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def mock_reco_channel_ml_tables(spark):
    now = datetime.datetime.now()
    repeat = 5
    reco = [
        {
            "TAG_ID": 507,
            "rating": 1
        },
        {
            "TAG_ID": 35,
            "rating": 0.696
        }
    ]
    data = {'CHANNEL_ID': list(range(repeat)),
            'RECOMMENDATIONS': [reco] * repeat,
            'UPDATE_DATE': [now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def mock_reco_channel_prog_ep_ml_tables(spark):
    now = datetime.datetime.now()
    repeat = 5
    reco = [1, 2, 3, 4]
    data = {'CHANNEL_ID': list(range(repeat)),
            'PROGRAM_ID': list(range(repeat)),
            'EPISODE_ID': list(range(repeat)),
            'RECOMMENDATIONS': [reco] * repeat,
            'UPDATE_DATE': [now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def mock_reco_prog_ml_tables(spark):
    now = datetime.datetime.now()
    repeat = 5
    reco = {
        "AVG_NB_WATCHERS_30S": 4,
        "AVG_USER_WATCH_MINUTES": 3.5,
        "MAX_AVG_USER_WATCH_MINUTES": 4.5,
        "MAX_NB_WATCHERS_30S": 4.5,
        "NB_BOOKMARKS_365D": 5,
        "NB_BOOKMARKS_7D": 5,
        "NB_LIKES_365D": 4.5,
        "NB_LIKES_7D": 4.5,
        "RATING": 0.047
    }
    data = {'PROGRAM_ID': list(range(repeat)),
            'RECOMMENDATIONS': [reco] * repeat,
            'UPDATE_DATE': [now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def create_fact_watch_table_mock(spark):
    """
    Only program 0 and 1 will have watch history
    """
    now = datetime.datetime.now()
    data = {'USER_ID': [1, 33, 1, 0],
            'RECEIVED_AT': [now - timedelta(days=1) for i in range(4)],
            'DEVICE_TYPE': ['Android', 'ios', 'ios', 'ios'],
            'ACTION_DEVICE_TYPE': ['tv', 'tablet', "phone", 'tv'],
            "ASSET_TYPE": ["replay", "vod", "replay", "replay"],
            'REAL_START_AT': [now,
                              now - datetime.timedelta(days=1),
                              now - datetime.timedelta(days=2),
                              datetime.datetime(1990, 1, 1)],
            'REAL_END_AT': [now, now, now, now],
            'DATE_DAY': [now, now - datetime.timedelta(days=1), now - datetime.timedelta(days=2), now],
            'CHANNEL_ID': [1, 1, 1, 1],
            'PROGRAM_ID': [1, 0, 1, 1],
            'DURATION': [11 * 60, 20 * 60, 12 * 60, 10000000]}
    data["EPISODE_ID"] = data["PROGRAM_ID"]
    return create_spark_df_from_data(spark, data)


def create_actual_recording_table_mock(spark):
    now = datetime.datetime.now()
    data = {'USER_ID': list(range(3)),
            'SCHEDULED_AT': [now, now - datetime.timedelta(days=1),
                             now - datetime.timedelta(days=2)],
            'PROGRAM_ID': [1, 0, 1],
            'DELETED_AT': [None, None, now]
            }
    return create_spark_df_from_data(spark, data)


def create_scheduled_recording_table_mock(spark):
    now = datetime.datetime.now()
    repeat = 101
    data = {'USER_ID': list(range(3 * repeat)),
            'SCHEDULED_AT': [now, now - datetime.timedelta(days=1),
                             now - datetime.timedelta(days=2)] * repeat,
            'PROGRAM_ID': [1, 0, 1] * repeat,
            'DELETED_AT': [None, None, now] * repeat
            }
    return create_spark_df_from_data(spark, data)


def create_rating_table_mock(spark):
    data = {'RATING': [1.2, 1.2, 1.9, 1.2, 1.1],
            'PROGRAM_RATING_TYPE_ID': [1, 2, 2, 33, 2],
            'PROGRAM_ID': [1, 0, 1, 1, 4]
            }
    return create_spark_df_from_data(spark, data)


def mock_free_channels_query(spark):
    data = {"CHANNEL_ID": list(range(100)), "TVBUNDLE_ID": [25] * 100}
    return create_spark_df_from_data(spark, data)


def mock_keep_person_id_query(spark):
    data = {"ID": list(range(10))}
    return create_spark_df_from_data(spark, data)


def mock_backend_program(spark):
    n_progs = 6
    pids = list(range(n_progs))
    data = {"ID": pids,
            "TITLE": ["prog_{}".format(i) for i in range(n_progs)],
            "REF_PROGRAM_CATEGORY_ID": [1 for _ in range(n_progs)],
            "REF_PROGRAM_KIND_ID": [1, 1, 1, 1, 52, 2],
            "PRODUCTION_YEAR": [1990 for _ in range(n_progs)],
            "DURATION": [30 * 60 for _ in range(n_progs)]}
    # Program 4 will be filtered by program kind filter
    return create_spark_df_from_data(spark, data)


def mock_backend_broadcast(spark, min_duration_in_minutes=10):
    now = datetime.datetime.now()
    min_duration = min_duration_in_minutes * 60 + 1
    program_ids = [0, 1, 2, 3, 4, 0, 0, 0, 5]
    rebroadcast_ids = [0, 0, 1, 3]
    channel_ids = [0 for _ in program_ids]
    rebroadcast_channel = [0 for _ in rebroadcast_ids]
    titles = [f"prog_{i}" for i in program_ids]
    rebroadcast_titles = [f"prog_{i}" for i in rebroadcast_ids]
    starts = [now + datetime.timedelta(hours=1) for _ in program_ids]
    ends = [t + datetime.timedelta(minutes=15) for t in starts]
    rebroadcast_starts = [now - datetime.timedelta(days=1) for _ in rebroadcast_ids]
    rebroadcast_ends = [t + datetime.timedelta(minutes=15) for t in rebroadcast_starts]
    durations = [min_duration for _ in program_ids]
    rebroadcast_durations = [min_duration for _ in rebroadcast_ids]
    # Program 3 will be filtered by duration filter
    durations[3] = 10
    data = {"PROGRAM_ID": program_ids + rebroadcast_ids,
            "EPISODE_ID": range(len(program_ids + rebroadcast_ids)),
            "REF_CSA_ID": [0 for _ in range(len(program_ids + rebroadcast_ids))],
            "TITLE": titles + rebroadcast_titles,
            "START_AT": starts + rebroadcast_starts,
            "END_AT": ends + rebroadcast_ends,
            "CHANNEL_ID": channel_ids + rebroadcast_channel,
            "REPLAY_START_AT": starts + rebroadcast_starts,
            "REPLAY_END_AT": starts + rebroadcast_starts,
            "DURATION": durations + rebroadcast_durations}
    return create_spark_df_from_data(spark, data)


def mock_affinity_programs(spark):
    program_ids = [0, 1, 2, 3, 4, 0, 0, 0, 5]
    affinities = ["Action & Aventure" for _ in range(5)] + ["Famille", "Maison", "Autre", "Thrillers & Policiers"]
    data = {"PROGRAM_ID": program_ids, "AFFINITY": affinities}
    return create_spark_df_from_data(spark, data)


def mock_fact_audience(spark):
    user_ids = list(range(5))
    data = {"USER_ID": user_ids}
    for col_name in AFFINITIES:
        data[f'"{col_name}"'] = [1 if uid != 3 else 0 for uid in user_ids]
    for col in ["RFM28_CLUSTER", "RFM7_CLUSTER"]:
        data[col] = ["super_actifs" if iid != 4 else "un peu actif" for iid in user_ids]
    return create_spark_df_from_data(spark, data)


def build_user_raw_mock(spark):
    bday = datetime.datetime.now() - datetime.timedelta(days=20 * 365)
    data = {'ID': [1, 4, 2, 0, 3],
            'GENDER': ['F', 'M', None, '', None],
            'BIRTHDAY': [bday, bday, bday, None, None],
            'FIRST_NAME': ["Jean", "Pierre", "Marie", "Gertrude", "Hans"],
            'LAST_NAME': ["Dupont", "Martin", "Beliard", "Pouet", "Zimmer"],
            'EMAIL': ["jd@gmail.com", "pm@gmail.com", "mb@gmail.com", "gp@gmail.com", "hz@gmail.com"]
            }
    return create_spark_df_from_data(spark, data)


def build_watch_raw_mock(spark):
    today = datetime.date.today()
    timestamp = datetime.datetime(today.year, today.month, today.day)
    data = {'USER_ID': [1, 4, 2, 0, 3, 99],
            'WATCH_START': [timestamp - datetime.timedelta(seconds=60 * 60),
                            timestamp - datetime.timedelta(seconds=60 * 60 * 2),
                            timestamp - datetime.timedelta(seconds=60 * 45),
                            timestamp - datetime.timedelta(seconds=60 * 60 * 3),
                            timestamp - datetime.timedelta(seconds=60 * 3),
                            timestamp - datetime.timedelta(seconds=60 * 3)],
            'WATCH_END': [timestamp - datetime.timedelta(seconds=60 * 10),
                          timestamp + datetime.timedelta(seconds=60 * 60 * 2),
                          timestamp + datetime.timedelta(seconds=60 * 45),
                          timestamp + datetime.timedelta(seconds=60 * 60 * 3),
                          timestamp + datetime.timedelta(seconds=60 * 3),
                          timestamp + datetime.timedelta(seconds=60 * 3)],
            'CHANNEL_ID': [1, 12, 8, 2, 47, 33]
            }
    return create_spark_df_from_data(spark, data)


def build_panel_mock(spark):
    data = {'USER_ID': [1, 4, 2, 0, 3],
            'PANEL_TYPE': ["H_4", "H_4", "H_4", "F_4", "H_4"],
            'W': [0.5, 0.1, 1.7, 1.3, 2.2]
            }
    return create_spark_df_from_data(spark, data)


def build_live_delay_mock(spark):
    data = {'DEVICE_TYPE': ["android", "macOs", "tvos", "windows", "tizen"],
            'DELAY_90': [None, 233, 198, 139, 345]
            }
    return create_spark_df_from_data(spark, data)


def mock_fact_audience_aff_only(spark):
    random.seed(10)
    now = datetime.datetime.now()
    user_ids = [1, 4, 2, 0, 3]
    dates = [now.date() for _ in user_ids]
    data = {"USER_ID": user_ids,
            '"date"': dates
            }
    for col_name in AFFINITIES:
        data[f'"{col_name}"'] = [random.uniform(0, 1) for uid in user_ids]

    return create_spark_df_from_data(spark, data)


def mock_fact_audience_aff_only_flag(spark):
    random.seed(10)
    now = datetime.datetime.now()
    user_ids = [1, 4, 2, 0, 3]
    dates = [now.date() for _ in user_ids]
    data = {"USER_ID": user_ids,
            '"date"': dates
            }
    for col_name in AFFINITIES:
        data[f'"{col_name}"'] = [random.randrange(2) for uid in user_ids]

    return create_spark_df_from_data(spark, data)


def mock_rfm_window_slide_01D_size_07d_df(spark):
    user_ids = [1, 4, 2, 0, 99]
    now = datetime.datetime.now()
    data = {"USER_ID": user_ids,
            "RFM_START_DAY": [now.date() - timedelta(days=7) for _ in user_ids],
            "RFM_END_DAY": [now.date() for _ in user_ids],
            "FIRST_WATCH_DAY": [now.date() - timedelta(days=6) for _ in user_ids],
            "LAST_WATCH_DAY": [now.date() - timedelta(days=2) for _ in user_ids],
            "R": [2 for _ in user_ids],
            "F": [3 for _ in user_ids],
            "M": [2000 for _ in user_ids],
            "RFM_CLUSTER": ['actifs', 'actifs', 'actifs', 'actifs', 'inactifs_new_reg'],
            "UPDATE_DATE": [now.date() for _ in user_ids],
            "IS_NEW_REG": [False for _ in user_ids]
            }
    return create_spark_df_from_data(spark, data)


def mock_rfm_window_slide_01D_size_28d_df(spark):
    user_ids = [1, 4, 2, 0, 3]
    now = datetime.datetime.now()
    data = {"USER_ID": user_ids,
            "RFM_START_DAY": [now.date() - timedelta(days=7) for _ in user_ids],
            "RFM_END_DAY": [now.date() for _ in user_ids],
            "FIRST_WATCH_DAY": [now.date() - timedelta(days=6) for _ in user_ids],
            "LAST_WATCH_DAY": [now.date() - timedelta(days=2) for _ in user_ids],
            "R": [2 for _ in user_ids],
            "F": [3 for _ in user_ids],
            "M": [2000 for _ in user_ids],
            "RFM_CLUSTER": ['actifs' for _ in user_ids],
            "UPDATE_DATE": [now.date() for _ in user_ids]
            }
    return create_spark_df_from_data(spark, data)


def create_detailed_episode_df(spark):
    """
    This a mock of the junction between the episode and the season tables
    """
    data = dict()
    n_elements = 10
    data["PROGRAM_ID"] = list(range(n_elements))
    data["REF_CSA_ID"] = [0 for _ in range(n_elements)]
    data["EPISODE_ID"] = list(range(n_elements))
    data["EPISODE_NUMBER"] = [1 for _ in range(n_elements)]
    data["DURATION"] = [16 * 60 for _ in range(n_elements)]
    data["SEASON_NUMBER"] = [1 for _ in range(n_elements)]

    return create_spark_df_from_data(spark, data)


def mock_broadcast_for_episode_match(spark, episode_df, broadcast_df, kind_id=0, cat_id=2):
    """
    We create a broadcast of episode following the ones given as input.
    So we are sure there is a follow up

    episode_df and broadcast_df should be matching
    1 - Find max episode id
    2 - Create a broadcast with new episode following the ones already defined
    3 - Join old and new df
    """
    episode_rows = episode_df.collect()
    max_episode_id = max(r.EPISODE_ID for r in episode_rows)
    max_episode_number = max(r.EPISODE_NUMBER for r in episode_rows)

    # New episode infos
    data = OrderedDict()
    current_program_ids = [r.PROGRAM_ID for r in episode_rows]
    data["PROGRAM_ID"] = current_program_ids
    data["REF_CSA_ID"] = [0 for _ in range(len(episode_rows))]
    data["EPISODE_ID"] = [max_episode_id + i for i in range(len(episode_rows))]
    # ALl next episode except, prog 0 which is next season
    data["EPISODE_NUMBER"] = [max_episode_number + 1 if r.PROGRAM_ID != 0 else 1 for r in episode_rows]
    data["DURATION"] = [16 * 60 for _ in current_program_ids]
    data["SEASON_NUMBER"] = [1 if r.PROGRAM_ID != 0 else 2 for r in episode_rows]
    new_episode_df = create_spark_df_from_data(spark, data)

    # New broadcast
    min_duration_in_minutes = 15
    now = datetime.datetime.now()
    min_duration = min_duration_in_minutes * 60 + 1
    channel_ids = [0 for _ in current_program_ids]
    starts = [now + datetime.timedelta(hours=1) for _ in current_program_ids]
    ends = [now + datetime.timedelta(hours=2) for _ in current_program_ids]
    durations = [min_duration for _ in current_program_ids]
    data = {"PROGRAM_ID": current_program_ids,
            "EPISODE_ID": data["EPISODE_ID"],
            "REF_CSA_ID": data["REF_CSA_ID"],
            "EPISODE_NUMBER": data["EPISODE_NUMBER"],
            "SEASON_NUMBER": data["SEASON_NUMBER"],
            "TITLE": ["" for _ in current_program_ids],
            "REF_PROGRAM_KIND_ID": [kind_id for _ in current_program_ids],
            "REF_PROGRAM_CATEGORY_ID": [cat_id for _ in current_program_ids],
            "START_AT": starts,
            "END_AT": ends,
            "CHANNEL_ID": channel_ids,
            "REPLAY_START_AT": starts,
            "REPLAY_END_AT": starts,
            "PRODUCTION_YEAR": [1999 for _ in current_program_ids],
            "TVBUNDLE_ID": [25 for _ in current_program_ids],
            "DURATION": durations}

    new_broadcast_df = create_spark_df_from_data(spark, data)

    full_broadcast = new_broadcast_df.select(broadcast_df.columns).unionAll(broadcast_df)
    full_episode = episode_df.unionAll(new_episode_df)
    return full_episode, full_broadcast


def mock_full_reco_df(spark, prog_ids=(0, 1, 2, 3, 0), user_id=0, channel_ids=None):
    data = {"USER_ID": [user_id] * 5,
            "PROGRAM_ID": list(prog_ids),
            "EPISODE_ID": list(prog_ids),
            "REF_CSA_ID": [1 for _ in prog_ids],
            "AFFINITY": ["Action & Aventure", "Action & Aventure", "Action & Aventure", "Famille", "Famille"],
            "reco_origin": ["external_rating", "external_rating", "avg_watch_duration", "external_rating",
                            "external_rating"],
            "CHANNEL_ID": channel_ids if channel_ids else [45, 45, 45, 45, 45],
            "best_rank": [1, 2, 1, 1, 2],
            "start_rank": [0, 0, 0, 0, 0],
            "USER_AFFINITY_RANK": [0, 0, 0, 2, 2],
            "total_rebroadcast": [2, 1, 0, 1, 2]}
    return create_spark_df_from_data(spark, data)


def mock_user_tvbundle_df(spark):
    data = {"USER_ID": [0],
            "TVBUNDLE_ID": [26]}
    return create_spark_df_from_data(spark, data)


def mock_user_channels(spark):
    data = {
        "USER_ID": [0 for _ in range(100)],
        "CHANNEL_ID": list(range(100))
    }
    return create_spark_df_from_data(spark, data)


def mock_product_to_tvbundle(spark):
    ids = list(range(100))
    data = {
        "PRODUCT_ID": ids,
        "EQUIVALENCE_CODE": ["FREE" if i < 33 else "OPTION_100H" if i < 66 else "EXTENDED" for i in ids],
        "TVBUNDLE_ID": [25 if i < 33 else 90 if i < 66 else 26 for i in ids]
    }
    return create_spark_df_from_data(spark, data)


def mock_episode_table(spark):
    id_range = list(range(10))
    data = {
        "ID": id_range,
        "NUMBER": id_range,
        "SEASON_ID": id_range,
        "DURATION": [16 * 60 for _ in id_range],
        "TITLE_ORIGINAL": [f"PROGR_0_ep_{i}" for i in id_range],
        "PROGRAM_ID": [0 for _ in id_range],
        "REF_CSA_ID": [1 for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_season_table(spark):
    id_range = list(range(10))
    data = {
        "ID": id_range,
        "NUMBER": [0 for _ in id_range],
        "PROGRAM_ID": [0 for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_replay_table(spark):
    id_range = list(range(10))
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=2)
    data = {
        "ID": id_range,
        "EPISODE_ID": id_range,
        "CHANNEL_ID": [0 for _ in id_range],
        "availability_start_at": [now - delta for _ in id_range],
        "availability_end_at": [now + delta for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_vod_table(spark):
    id_range = list(range(11))
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=2)
    data = {
        "ID": id_range,
        "EPISODE_ID": id_range,
        "CHANNEL_ID": [0 for _ in id_range],
        "AVAILABLE_FROM": [now - delta for _ in id_range],
        "AVAILABLE_UNTIL": [now + delta for _ in id_range],
        "VIDEO_TYPE": ["vod" for _ in id_range],
        "DISABLED": [0 for _ in id_range],
        "WITHDRAWN_AT": [None if i != 10 else 1 for i in id_range],
        "DELETED_AT": [None if i != 10 else 1 for i in id_range],
        "VOD_FORMAT_ID": [1 for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_subscribers_table(spark):
    id_range = list(range(10))
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=2)
    data = {
        "USER_ID": id_range,
        "EQUIVALENCE_CODE": ["OCS" for _ in id_range],
        "EXPIRES_AT_DATE": [now + delta for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_psub_flag_table(spark):
    id_range = list(range(10))
    data = {
        "USER_ID": id_range,
        "SUB_CLUSTER": ["psub_low_medium" if i % 2 == 0 else "psub_medium_high" for i in id_range],
        "OFFER": ["MTV" for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_registered(spark):
    id_range = list(range(10))
    now = datetime.datetime.now()
    data = {
        "USER_ID": list(id_range),
        "FILE_NAME": ["segment-logs" for _ in id_range],
        "REG_TYPE": ["B2C" for _ in id_range],
        "RECEIVED_AT": [now if i % 3 == 0 else now - timedelta(days=1) for i in id_range],
        "ACTIVE_REG": [True for _ in id_range],
        "DEVICE_ID": ["12E123" for _ in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_cmp_user_consents_table(spark):
    id_range = list(range(5))
    now = datetime.datetime.now()
    purposes = '{"enabled": [1,2,3,4,5,6,7,8,9,10], "enabled_li": [1,2,3,4,5,6,7,8,9,10]}'
    special_features = '{"enabled": [1,2], "enabled_li": [1,2]}'
    data = {
        "USER_ID": id_range,
        "CREATED_AT": [now for i in id_range],
        "PURPOSES": [purposes for i in id_range],
        "SPECIALFEATURES": [special_features for i in id_range],
        "CUSTOMPURPOSES": [special_features for i in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_user_subscription_table(spark):
    id_range = list(range(5))
    data = {
        "USER_ID": id_range,
        "PLATFORM": ["apple" if i % 2 == 0 else "molotovpay" for i in id_range]
    }
    return create_spark_df_from_data(spark, data)


def mock_user_consents_table(spark):
    id_range = list(range(10))
    data = {
        "USER_ID": list(id_range),
        "TCFCS": ["" for _ in id_range],
    }
    return create_spark_df_from_data(spark, data)


def mock_person_table(spark):
    data = {"ID": [1, 2, 3, 4],
            "FIRST_NAME": ["George", "Scarlett", "Martin", "Claire"],
            "LAST_NAME": ["Clooney", "Johansson", "Scorsese", "Chazal"]
            }
    return create_spark_df_from_data(spark, data)


def mock_person_program(spark):
    data = {"PERSON_ID": [3, 1, 2, 4, 1, 2, 1, 3, 4],
            "PROGRAM_ID": [0, 0, 0, 1, 1, 2, 3, 4, 2],
            "ORDER": [1, 2, 5, 7, 1, 3, 5, 1, 999],
            "REF_PERSON_FUNCTION_ID": [2, 4, 4, 64, 4, 4, 4, 2, 64]}
    return create_spark_df_from_data(spark, data)


def mock_rel_episode_person(spark):
    data = {"PERSON_ID": [3, 1, 2, 4, 1, 2, 1, 3, 4],
            "EPISODE_ID": [0, 0, 0, 1, 1, 2, 3, 4, 2],
            "ORDER": [1, 2, 5, 7, 1, 3, 5, 1, 999],
            "REF_PERSON_FUNCTION_ID": [2, 4, 4, 64, 4, 4, 4, 2, 64]}
    return create_spark_df_from_data(spark, data)


def mock_user_person_follows(spark):
    now = datetime.datetime.now()
    created_at = now - datetime.timedelta(days=1)
    data = {"USER_ID": [1, 1, 33, 0, 0, 0],
            "PERSON_ID": [4, 1, 2, 1, 3, 4],
            "SOURCE": ["molotov", "molotov", "molotov", "molotov", "molotov", "facebook"],
            "CREATED_AT": [created_at, created_at, created_at - datetime.timedelta(days=4), created_at,
                           created_at - timedelta(days=2), created_at]}
    return create_spark_df_from_data(spark, data)


def mock_function_person(spark):
    data = {"ID": [99, 4, 78, 64, 2],
            "DISPLAY_NAME": ["Sujet", "Acteur", "Présentateur", "Journaliste", "Réalisateur"]}
    return create_spark_df_from_data(spark, data)


def mock_nightly_watcher_query(spark):
    id_range = [6]
    data = {
        "USER_ID": list(id_range),
    }
    return create_spark_df_from_data(spark, data)


def mock_fact_page(spark):
    uids = list(range(3))
    data_uids = uids + uids
    now = datetime.datetime.now() - datetime.timedelta(days=1)
    data = {
        "USER_ID": data_uids,
        "EVENT_DATE": [now.date() for _ in data_uids],
        "TIMESTAMP": [now for _ in data_uids],
        "PAGE_NAME": ["search" for _ in data_uids],
        "QUERY": ["ab" if i <= 1 else "a" for i in data_uids],
        "ACTION_DEVICE_TYPE": ["tv" for i in data_uids]
    }
    return create_spark_df_from_data(spark, data)


def mock_backend_channel(spark):
    data = {
        "ID": [1, 2, 3],
        "NAME": ["TF1", "FR2", "FR3"],
        "FRENCH_TNT_RANK": [1, 2, 3],
    }
    return create_spark_df_from_data(spark, data)


def mock_rel_channel_table(spark):
    pids = list(range(100))
    data = {
        "CHANNEL_ID": pids,
        "TVBUNDLE_ID": [25 for _ in pids],
    }
    return create_spark_df_from_data(spark, data)


def mock_user_channel_item_table(spark):
    pids = list(range(100))
    now = datetime.datetime.now()
    data = {
        "PROGRAM_ID": pids,
        "USER_ID": pids,
        "CREATED_AT": [now for _ in pids]
    }
    return create_spark_df_from_data(spark, data)


def mock_rel_episode_tag_table(spark):
    pids = list(range(100))
    data = {
        "EPISODE_ID": pids,
        "TAG_ID": [i % 2 for i in pids],
    }
    return create_spark_df_from_data(spark, data)


def mock_tvbundle_table(spark):
    pids = list(range(100))
    data = {
        "ID": pids,
        "IS_COMMERCIALIZED": [1 for i in pids],
    }
    return create_spark_df_from_data(spark, data)


def mock_reco_user_progs_this_week_latest(spark):
    pids = 1
    now = datetime.datetime.now()
    data = {
        "USER_ID": pids,
        "RECOMMENDATIONS": [
            """[{
                "AFFINITY": "Drames & Sentiments",
                "PROGRAM_ID": 46802,
                "ranking": 8.734767,
                "rating": 0.1,
                "reco_origin": "total_celeb_points"
            }]"""],
        "UPDATE_DATE": now,
    }
    return create_spark_df_from_data(spark, data)


def mock_reco_user_person_latest(spark):
    pids = list(range(10))
    person_ids = list(range(4))
    now = datetime.datetime.now()
    recos = ','.join(["""{{"PERSON_ID": {0}, "rating": {1}}}""".format(i, (i + 1) / 2) for i in person_ids])
    data = {
        "USER_ID": [0 for _ in pids],
        "RECOMMENDATIONS": [f"[{recos}]" for _ in pids],
        "UPDATE_DATE": [now - datetime.timedelta(days=i) for i in pids]
    }
    return create_spark_df_from_data(spark, data)


def mock_reco_user_progs_this_week_var(spark):
    pids = list(range(5))
    now = datetime.datetime.now().date()

    data = {
        "USER_ID": 2 * [i for i in pids],
        "RECOMMENDATIONS": 2 * [
            """[{
                "AFFINITY": "Drames & Sentiments",
                "PROGRAM_ID": 46802,
                "ranking": 8.734767,
                "rating": 0.1,
                "reco_origin": "total_celeb_points"
            }]""" for _ in pids],
        "UPDATE_DATE": 2 * [now for _ in pids],
        "VARIATIONS": ["A" for _ in pids] + ["B" for _ in pids]
    }
    return create_spark_df_from_data(spark, data)


def mock_reco_prog_progs_meta_latest(spark):
    pids = 1
    now = datetime.datetime.now()
    data = {
        "PROGRAM_ID": pids,
        "RECOMMENDATIONS": [
            """[{
                "program_id": 193,
                "rating": 0.1
            }]"""],
        "UPDATE_DATE": now,
    }
    return create_spark_df_from_data(spark, data)


def mock_reco_prog_progs_meta_var(spark):
    pids = list(range(10))
    now = datetime.datetime.now()
    data = {
        "PROGRAM_ID": pids,
        "RECOMMENDATIONS": [
            """[{
                "program_id": 193,
                "rating": 0.1
            }]""" for _ in pids],
        "UPDATE_DATE": [now for _ in pids],
        "VARIATIONS": ["A" if i % 2 == 0 else "B" for i in pids]
    }
    return create_spark_df_from_data(spark, data)


def mock_user_split_ab_test(spark):
    pids = list(range(2))
    data = {
        "AB_TEST_ID": [1, 1],
        "USER_ID": pids,
        "VARIATIONS": ["A", "B"]
    }
    return create_spark_df_from_data(spark, data)   


def mock_program_split_ab_test(spark):
    pids = list(range(2))
    data = {
        "AB_TEST_ID": [1, 1],
        "PROGRAM_ID": pids,
        "VARIATIONS": ["A", "B"]
    }
    return create_spark_df_from_data(spark, data)  


def mock_ab_test_conf(spark):
    pids = 1
    now = datetime.datetime.now().date()
    data = {
        "AB_TEST_ID": [pids],
        "DESCRIPTION": ["test v1 vs v2"],
        "ALT_PROPORTION": [0.5],
        "MERGE_RECO_TABLE": ["RECO_USER_PROGS_THIS_WEEK_VAR"],
        "LATEST_RECO_TABLE": ["RECO_USER_PROGS_THIS_WEEK_LATEST"],
        "START_AT": [now],
        "END_AT": [now]
    }
    return create_spark_df_from_data(spark, data)


def mock_raw_user(spark):
    pids = list(range(5))
    data = {"USER_ID": pids}
    return create_spark_df_from_data(spark, data)


def mock_raw_program(spark):
    pids = list(range(5))
    data = {"PROGRAM_ID": pids}
    return create_spark_df_from_data(spark, data)


def mock_user_feature(spark):
    now = datetime.date.today() - datetime.timedelta(days=2)
    return create_spark_df_from_data(
        spark,
        {"USER_ID": [0, 1],
         "GENDER": ["U", "F"],
         "AGE": [20, 1],
         "DURATION_AFFINITY=Action": [0, 1],
         "DURATION_ACTION_DEVICE_TYPE=tv": [0, 1],
         "DURATION_CATEGORY=1": [0, 1],
         "DURATION_PERSON_ID": ['{"23": 32452, "2": 1}', ""],
         "DATE_DAY": [now, now]}
    )


def mock_program_feature(spark):
    now = datetime.date.today() - datetime.timedelta(days=2)
    return create_spark_df_from_data(
        spark,
        {"PROGRAM_ID": [0, 1],
         "PRODUCTION_YEAR": [0, 1999],
         "REF_PROGRAM_CATEGORY_ID": [1, 0],
         "REF_PROGRAM_KIND_ID": [1, 0],
         "TOTAL_CELEB_POINTS": [0, 1],
         "EXTERNAL_RATING": [0, 1],
         "PROGRAM_DURATION": [1200, 0],
         "AFFINITY": ["Action", "Action"],
         "DATE_DAY": [now, now],
         "FAMOUS_CAST": ['["33"]', '["3"]']
         }
    )


def mock_channel_feature(spark):
    now = datetime.date.today() - datetime.timedelta(days=2)
    return create_spark_df_from_data(
        spark,
        {"CHANNEL_ID": [0, 1],
         "channel_DURATION_socio=F_4": [0, 1],
         "channel_DURATION_socio=H_4": [0, 1],
         "channel_DURATION_AFFINITY=Action": [0, 1],
         "DATE_DAY": [now, now]}
    )


def mock_pclick_training_log(spark):
    now = datetime.date.today() - datetime.timedelta(days=2)
    return create_spark_df_from_data(
        spark,
        {"USER_ID": [0, 1, 0, 1], "PROGRAM_ID": [0, 1, 1, 0], "CHANNEL_ID": [0, 0, 1, 1],
         "label": [0, 0, 1, 1], "RANK": [1, 4, 5, 2],
         "DATE_DAY": [now, now, now, now]}
    )


def mock_user_profile(spark):
    ids = list(range(100))
    return create_spark_df_from_data(
        spark,
        {"USER_ID": ids, "ID": ids}
    )


def mock_profile_setting(spark):
    ids = list(range(100))
    return create_spark_df_from_data(
        spark,
        {"PROFILE_ID": ids,
         "REF_PROFILE_SETTING_TYPE_ID": [6 for _ in ids],
         "setting_value": [3 for _ in ids]}
    )


def multiplex_mock(spark, options, table_name, *args):
    table_name = table_name.lower()
    if table_name == "backend.program_rating":
        return create_rating_table_mock(spark)
    elif table_name == "dw.fact_watch":
        return create_fact_watch_table_mock(spark)
    elif table_name == "dw.fact_cmp_user_consents":
        return mock_fact_cmp_user_consents_table(spark)
    elif table_name == "dw.fact_user_subscription":
        return mock_fact_user_subscription_table(spark)
    elif table_name == "backend.actual_recording":
        return create_actual_recording_table_mock(spark)
    elif table_name == "backend.scheduled_recording":
        return create_scheduled_recording_table_mock(spark)
    elif table_name == "backend.program":
        return mock_backend_program(spark)
    elif table_name == "backend.broadcast":
        return mock_backend_broadcast(spark)
    elif table_name == "external_sources.daily_prog_affinity":
        return mock_affinity_programs(spark)
    elif table_name == "dw.fact_audience":
        return mock_fact_audience(spark)
    elif table_name == "backend.user_raw":
        return build_user_raw_mock(spark)
    elif table_name == "external_sources.users_panel_weight":
        return build_panel_mock(spark)
    elif table_name == "dw.live_delay_lookup":
        return build_live_delay_mock(spark)
    elif table_name == "dw.fact_audience_aff_only":
        return mock_fact_audience_aff_only(spark)
    elif table_name == "dw.fact_audience_aff_only_flag":
        return mock_fact_audience_aff_only_flag(spark)
    elif table_name == "dw.fact_rfm_window_slide_01d_size_07d":
        return mock_rfm_window_slide_01D_size_07d_df(spark)
    elif table_name == "dw.fact_rfm_window_slide_01d_size_28d":
        return mock_rfm_window_slide_01D_size_28d_df(spark)
    elif table_name == "backend.episode":
        return mock_episode_table(spark)
    elif table_name == "backend.season":
        return mock_season_table(spark)
    elif table_name == "backend.replay":
        return mock_replay_table(spark)
    elif table_name == "backend.vod":
        return mock_vod_table(spark)
    elif table_name == "backend.subscribers":
        return mock_subscribers_table(spark)
    elif table_name == "dw.daily_sub_pred_flag":
        return mock_psub_flag_table(spark)
    elif table_name == "dw.fact_registered":
        return mock_fact_registered(spark)
    elif table_name == "backend.cmp_user_consents":
        return mock_user_consents_table(spark)
    elif table_name == "backend.person":
        return mock_person_table(spark)
    elif table_name == "backend.rel_program_person":
        return mock_person_program(spark)
    elif table_name == "backend.rel_episode_person":
        return mock_rel_episode_person(spark)
    elif table_name == "backend.user_follow_person":
        return mock_user_person_follows(spark)
    elif table_name == "backend.ref_person_function":
        return mock_function_person(spark)
    elif table_name == "dw.fact_page":
        return mock_fact_page(spark)
    elif table_name == "backend.channel":
        return mock_backend_channel(spark)
    elif table_name == "backend.rel_tvbundle_channel":
        return mock_rel_channel_table(spark)
    elif table_name == "backend.user_channel_item":
        return mock_user_channel_item_table(spark)
    elif table_name == "backend.rel_episode_tag":
        return mock_rel_episode_tag_table(spark)
    elif table_name == "backend.tvbundle":
        return mock_tvbundle_table(spark)
    elif table_name == "ml.reco_user_person_latest":
        return mock_reco_user_person_latest(spark)
    elif table_name == "ml.reco_user_progs_this_week_latest":
        return mock_reco_user_progs_this_week_latest(spark)
    elif table_name == "ml.reco_user_progs_this_week_var":
        return mock_reco_user_progs_this_week_var(spark)
    elif table_name == "ml.reco_prog_progs_meta_latest":
        return mock_reco_prog_progs_meta_latest(spark)
    elif table_name == "ml.reco_prog_progs_meta_var":
        return mock_reco_prog_progs_meta_var(spark)
    elif table_name == "dw.ab_test_conf":
        return mock_ab_test_conf(spark)
    elif table_name == "dw.user_split_ab_test":
        return mock_user_split_ab_test(spark)
    elif table_name == "dw.program_split_ab_test":
        return mock_program_split_ab_test(spark)
    elif table_name == "ml.user_feature_log":
        return mock_user_feature(spark)
    elif table_name == "ml.program_feature_log":
        return mock_program_feature(spark)
    elif table_name == "ml.channel_feature_log":
        return mock_channel_feature(spark)
    elif table_name == "ml.pclick_training_log":
        return mock_pclick_training_log(spark)
    elif table_name == "backend.user_profile":
        return mock_user_profile(spark)
    elif table_name == "backend.profile_setting":
        return mock_profile_setting(spark)
    else:
        raise Exception(f"table {table_name} is not mocked")


def multiplex_query_mock(spark, options, query):
    if "dw.fact_cmp_user_consents" in query:
        return build_user_raw_mock(spark).select(F.col("ID").alias("USER_ID"))
    elif "RAW.PUBLIC.SEGMENT" in query and "min_client_offset" in query:
        return build_watch_raw_mock(spark)
    elif "cte_nb_days_presence" in query:
        return mock_nightly_watcher_query(spark)
    else:
        raise Exception(f"Query '{query[:20]}...' is not mocked")
