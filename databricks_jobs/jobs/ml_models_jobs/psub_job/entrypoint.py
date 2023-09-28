from datetime import timedelta

import mlflow
import pandas as pd
import pyspark.sql.types as T
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from databricks_jobs.common import Job
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df, \
    write_df_to_snowflake


class PsubJob(Job):

    def __init__(self, *args, **kwargs):
        super(PsubJob, self).__init__(*args, **kwargs)

        self.snowflake_user = self.conf.get("user_name", "")
        self.snowflake_password = self.conf.get("password", "")

        self.sf_database = self.conf.get("env", "DEV")
        self.now = self.parse_date_args()
        self.logger.info(f"Running on date : {self.now}")

        if self.sf_database == 'DEV':
            self.sf_wharehouse = 'PROD_WH'
            self.sf_role = 'DEV'
            self.daily_sub_pred_table = "DAILY_SUB_PRED" + '_TEST1'
            self.daily_sub_pred_flag_table = "DAILY_SUB_PRED_FLAG" + '_TEST1'

        elif self.sf_database == 'PROD':
            self.sf_wharehouse = 'PROD_WH'
            self.sf_role = 'PROD'
            self.daily_sub_pred_table = "DAILY_SUB_PRED"
            self.daily_sub_pred_flag_table = "DAILY_SUB_PRED_FLAG"

        self.interval = 15

        # Read options
        self.read_options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflake_user,
            "sfPassword": self.snowflake_password,
            "sfDatabase": self.sf_database,
            "sfSchema": "public",
            "sfWarehouse": self.sf_wharehouse,
            "sfRole": "ANALYST_{}".format(self.sf_role)
        }

        # Write options
        self.user_sub_pred_options = {
            "sfUrl": "molotovtv.eu-west-1.snowflakecomputing.com",
            "sfUser": self.snowflake_user,
            "sfPassword": self.snowflake_password,
            "sfDatabase": self.sf_database,
            "sfSchema": "DW",
            "sfWarehouse": self.sf_wharehouse,
            "keep_column_case": "off",
            "sfRole": "ANALYST_{}".format(self.sf_role)
        }

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        users_prediction_data_df = self.prepare_data_prediction()

        self.update_model_version("Psub")
        # self.update_model_version("Psub-v2")

        prediction_data_df = self.prediction(users_prediction_data_df, "Psub")
        prediction_data_df = prediction_data_df.withColumn("VERSION", F.lit(1))

        # prediction_data_df_v2 = self.prediction(users_prediction_data_df, "Psub-v2")
        # prediction_data_df_v2 = prediction_data_df_v2.withColumn("VERSION", F.lit(2))

        prediction_data_df_all = prediction_data_df
        users_offer_data_df = self.prepare_data_offer()
        offer_with_score_df = self.better_offer(users_offer_data_df)

        psub_offer_df = self.load_data_psub(prediction_data_df_all, offer_with_score_df)
        psub_with_cluster = self.psub_cluster(psub_offer_df)
        psub_df_with_cluster_vect = self.psub_offer_vect(psub_with_cluster)
        self.write_pred_to_snowflake(prediction_data_df_all, offer_with_score_df, psub_df_with_cluster_vect)

    def prepare_data_prediction(self):
        # Select free users information on 15 last days for subscription probability
        #      -page_viewed
        #      -category watch
        #      -devices
        #      -socio_demo
        date_1 = self.now - timedelta(days=1)
        date_15 = self.now - timedelta(days=self.interval)
        now_format = '{}-{}-{}'.format(self.now.year, self.now.month, self.now.day)
        date_1_format = '{}-{}-{}'.format(date_1.year, date_1.month, date_1.day)
        date_15_format = '{}-{}-{}'.format(date_15.year, date_15.month, date_15.day)
        query_data = """
          with testing_sample as (with sub_set as (select user_id, is_sub_active, status_day_date, sub_real_end_at
            from dw.fact_user_subscription_day_bill 
            where status_day_date = '{2}'
            and platform not in ('digital_virgo')
            and ((IS_SUB_ACTIVE = TRUE AND ((status_day_date < sub_real_end_at) OR sub_real_end_at is NULL))
            OR 
            (status_day_date = sub_real_end_at))
            )
          select cte_free.user_id,socio_demo
          from
          (select dw.fact_audience_aff_only_flag.user_id,dw.fact_audience_aff_only_flag."date"
          from dw.fact_audience_aff_only_flag
          left join sub_set
          on sub_set.user_id=dw.fact_audience_aff_only_flag.user_id
          where "date" = '{2}'
          and sub_set.is_sub_active is null
          ) as cte_free
          join backend.user
          on backend.user.id=cte_free.user_id
          where not email like any ('%@molotov%', '%@temp-molotov%')
          ),

          cte_devices as (select user_id,count(*) as connected_devices
          from (select distinct testing_sample.user_id,action_device_type,device_type,device_id 
                from dw.fact_watch 
                join testing_sample 
                on testing_sample.user_id=dw.fact_watch.user_id
                where date_day between '{1}' and '{0}')
              group by user_id),

          cte_sub_watch as (select testing_sample.user_id,
                  sum(duration) as tot_duration,
                  sum(CASE WHEN category_id=1 THEN duration else 0 END) as movie_duration,
                  sum(CASE WHEN category_id=2 THEN duration else 0 END) as serie_duration,
                  sum(CASE WHEN category_id=3 THEN duration else 0 END) as sport_duration,
                  sum(CASE WHEN category_id=4 THEN duration else 0 END) as information_duration,
                  sum(CASE WHEN category_id=5 THEN duration else 0 END) as entertainement_duration,
                  sum(CASE WHEN category_id=6 THEN duration else 0 END) as child_duration,
                  sum(CASE WHEN category_id=8 THEN duration else 0 END) as documentary_duration,
                  sum(CASE WHEN category_id=9 THEN duration else 0 END) as culture_duration,
                  sum(CASE WHEN action_device_type='tv' THEN duration else 0 END ) as tv_duration,
                  sum(CASE WHEN action_device_type='desktop' THEN duration else 0 END ) as desktop_duration,
                  sum(CASE WHEN action_device_type='phone' THEN duration else 0 END ) as phone_duration,
                  sum(CASE WHEN action_device_type='tablet' THEN duration else 0 END) as tablet_duration
          from dw.fact_watch
          left join testing_sample
          on dw.fact_watch.user_id=testing_sample.user_id
          where date_day between '{1}' and '{0}'
          group by testing_sample.user_id
                          ),

          cte_cross_features as ( with cte_device_cat as (select testing_sample.user_id,duration,concat(action_device_type,'_',name) as device_cat
          from dw.fact_watch
          join backend.ref_program_category
          on backend.ref_program_category.id=dw.fact_watch.category_id
          join testing_sample
          on dw.fact_watch.user_id=testing_sample.user_id
          where date_day between '{1}' and '{0}'
          and duration_over_30_sec
          and action_device_type in('tv','desktop','tablet','phone')
          and category_id in (1,2,3,4,5,6,8,9)
          )

          select *
          from cte_device_cat
          pivot(sum(duration) for device_cat in ('tv_Sport','tv_Documentaires', 'tablet_Films', 'desktop_Documentaires',
           'desktop_Informations', 'tv_Enfants', 'phone_Sport', 'tv_Informations', 'desktop_Séries', 'desktop_Films', 
           'tablet_Informations', 'tablet_Séries', 'phone_Culture', 'phone_Séries', 'phone_Films', 'tablet_Enfants', 
           'tablet_Sport', 'phone_Divertissement', 'tv_Films', 'tablet_Divertissement', 'tv_Culture', 'phone_Documentaires',
           'phone_Informations', 'desktop_Culture', 'desktop_Divertissement', 'phone_Enfants', 'tablet_Documentaires', 
           'tablet_Culture', 'desktop_Sport', 'tv_Divertissement', 'tv_Séries', 'desktop_Enfants')) 
            as p(user_id,tv_Sport_duration,tv_Documentaires_duration, tablet_Films_duration,
                desktop_Documentaires_duration, desktop_Informations_duration, 
                tv_Enfants_duration, phone_Sport_duration, tv_Informations_duration, 
                desktop_Series_duration, desktop_Films_duration, tablet_Informations_duration, 
                tablet_Series_duration, phone_Culture_duration, phone_Series_duration, phone_Films_duration, 
                tablet_Enfants_duration, tablet_Sport_duration, phone_Divertissement_duration, tv_Films_duration, 
                tablet_Divertissement_duration, tv_Culture_duration,phone_Documentaires_duration, phone_Informations_duration, 
                desktop_Culture_duration, desktop_Divertissement_duration, phone_Enfants_duration, tablet_Documentaires_duration, 
                tablet_Culture_duration, desktop_Sport_duration, tv_Divertissement_duration, tv_Series_duration, desktop_Enfants_duration)
          order by user_id),

          cte_event as (select testing_sample.user_id,
          count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' THEN 1
                    WHEN event_name='store_offer' THEN 1
                    WHEN event_name='offer' THEN 1
                    WHEN event_name='store' THEN 1
                ELSE null END )as page_offer,
          count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT not in (select channel_id
          from backend.channel 
          join backend.rel_tvbundle_channel
          on backend.channel.id=backend.rel_tvbundle_channel.channel_id
          where tvbundle_id=25) THEN properties_json:channel_name::STRING
                  WHEN event_name='program' and properties_json:channel_id::INT not in (select channel_id
          from backend.channel 
          join backend.rel_tvbundle_channel
          on backend.channel.id=backend.rel_tvbundle_channel.channel_id
          where tvbundle_id=25) THEN properties_json:channel_name::STRING
                ELSE null END) as program,
          count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','ocs%','%molotov%','cine%','%extended%','rick %','%morty','adult%','%swim') THEN properties_json:query::STRING
                    WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','ocs%','%molotov%','cine%','%extended%','rick %','%morty','adult%','%swim') THEN properties_json:search_query::STRING
                    WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','ocs%','%molotov%','cine%','%extended%','rick %','%morty','adult%','%swim') THEN properties_json:keyword::STRING
                    WHEN event_name='search_result_clicked' 
                          and (properties_json:search_result_type::STRING ='channel' or properties_json:search_result_type::STRING ='program')
                          and properties_json:channel_id::INT not in (select channel_id
                              from backend.channel 
                              join backend.rel_tvbundle_channel
                              on backend.channel.id=backend.rel_tvbundle_channel.channel_id
                              where tvbundle_id=25) 
                    THEN properties_json:channel_name::STRING
              ELSE null END) as search,
          count(CASE WHEN event_name='bookmarks' THEN 1
              ELSE null END) as bookmark_page,
          count(CASE WHEN event_name='logged_in' THEN 1 
                ELSE null END) as app_login
          from testing_sample
          left join segment.snowpipe_all
          on segment.snowpipe_all.user_id=testing_sample.user_id
          where event_name in ('store_offer','offer','store','logged_in','bookmarks','page_viewed','keyword_search','search','search_result_clicked','program') 
          and event_date between '{1}' and '{0}'
          group by testing_sample.user_id)

          select testing_sample.user_id,
                socio_demo,
                (CASE when connected_devices is null then 0 else connected_devices end) as connected_devices,
                (CASE when tot_duration is null then 0 else tot_duration end) as tot_duration,
                (CASE when movie_duration is null then 0 else movie_duration end) as movie_duration,
                (CASE when serie_duration is null then 0 else serie_duration end) as serie_duration,
                (CASE when sport_duration is null then 0 else sport_duration end) as sport_duration,
                (CASE when information_duration is null then 0 else information_duration end) as information_duration,
                (CASE when entertainement_duration is null then 0 else entertainement_duration end) as entertainement_duration,
                (CASE when child_duration is null then 0 else child_duration end) as child_duration,
                (CASE when documentary_duration is null then 0 else documentary_duration end) as documentary_duration,
                (CASE when culture_duration is null then 0 else culture_duration end) as culture_duration,
                (CASE when tv_duration is null then 0 else tv_duration end) as tv_duration,
                (CASE when desktop_duration is null then 0 else desktop_duration end) as desktop_duration,
                (CASE when phone_duration is null then 0 else phone_duration end) as phone_duration,
                (CASE when tablet_duration is null then 0 else tablet_duration end) as tablet_duration,
                (CASE when page_offer is null then 0 else page_offer end) as page_offer,
                (CASE when program is null then 0 else program end) as program,
                (CASE when search is null then 0 else search end) as search,
                (CASE when bookmark_page is null then 0 else bookmark_page end) as bookmark_page,
                (CASE when app_login is null then 0 else app_login end) as app_login,
                tv_Sport_duration,tv_Documentaires_duration, tablet_Films_duration,desktop_Documentaires_duration, 
                desktop_Informations_duration, tv_Enfants_duration, phone_Sport_duration, 
                tv_Informations_duration, desktop_Series_duration, desktop_Films_duration, tablet_Informations_duration, 
                tablet_Series_duration,phone_Culture_duration, phone_Series_duration, phone_Films_duration, tablet_Enfants_duration, 
                tablet_Sport_duration, phone_Divertissement_duration, tv_Films_duration, tablet_Divertissement_duration, 
                tv_Culture_duration,phone_Documentaires_duration, phone_Informations_duration, desktop_Culture_duration,
                desktop_Divertissement_duration, phone_Enfants_duration, tablet_Documentaires_duration, 
                tablet_Culture_duration, desktop_Sport_duration, tv_Divertissement_duration, 
                tv_Series_duration, desktop_Enfants_duration
                from testing_sample
                left join cte_devices
                on cte_devices.user_id=testing_sample.user_id
                left join cte_sub_watch
                on cte_sub_watch.user_id=testing_sample.user_id
                left join cte_event
                on cte_event.user_id=testing_sample.user_id
                left join cte_cross_features
                on cte_cross_features.user_id=testing_sample.user_id

                order by user_id desc
          """.format(date_1_format, date_15_format, now_format)

        users_data_df = load_snowflake_query_df(self.spark, self.read_options, query_data)
        users_data_df = users_data_df.fillna(0)

        # Filter for negative duration

        duration_column = [col for col in users_data_df.columns if 'duration' in col.lower().split('_')]

        for column in duration_column:
            users_data_df = users_data_df. \
                withColumn(column, F.when(F.col(column) < 0, F.lit(0)).otherwise(F.col(column)))

        return users_data_df

    @staticmethod
    def update_model_version(model_name):
        # We check if the version in Production stage is the latest version,
        # if not the latest version is transited to Production stage

        def get_model_version(env_info):
            return max(map(int, map(lambda x: x.version, env_info))) if len(env_info) > 0 else 0

        client = MlflowClient()
        stage_none_info = client.get_latest_versions(model_name, stages=["None"])
        latest_none_version = get_model_version(stage_none_info)

        stage_prod_info = client.get_latest_versions(model_name, stages=["Production"])
        latest_prod_version = get_model_version(stage_prod_info)

        if latest_none_version > latest_prod_version:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_none_version,
                stage="Production"
            )

    def prediction(self, users_data_df, model_name):
        non_features = ['user_id']

        # Setting model acces uri, we load last version in Production stage
        model_stage = "Production"
        uri = f"models:/{model_name}/{model_stage}"

        users_data_df = users_data_df.select([F.col(x).alias(x.lower()) for x in users_data_df.columns])
        udf_inputs = F.struct(*users_data_df.drop(*non_features).columns)

        # Prediction with spark.
        loaded_model = mlflow.pyfunc.spark_udf(self.spark, model_uri=uri)

        predicted_data_df = users_data_df.withColumn('psub', loaded_model(udf_inputs))

        return predicted_data_df

    def prepare_data_offer(self):
        # Select users information on 15 last days for offer score computation:
        #      -page_viewed
        #      -affinities
        #      -category watch
        #      -socio_demo
        date_1 = self.now - timedelta(days=1)
        date_15 = self.now - timedelta(days=self.interval)
        now_format = '{}-{}-{}'.format(self.now.year, self.now.month, self.now.day)
        date_1_format = '{}-{}-{}'.format(date_1.year, date_1.month, date_1.day)
        date_15_format = '{}-{}-{}'.format(date_15.year, date_15.month, date_15.day)
        query_offer = """
        with testing_sample as (with sub_set as (select user_id, is_sub_active, status_day_date, sub_real_end_at
            from dw.fact_user_subscription_day_bill 
            where status_day_date = '{2}'
            and platform not in ('digital_virgo')
            and ((IS_SUB_ACTIVE = TRUE AND ((status_day_date < sub_real_end_at) OR sub_real_end_at is NULL))
            OR 
            (status_day_date = sub_real_end_at))
            )
          select cte_free.user_id,socio_demo
          from
          (select dw.fact_audience_aff_only_flag.user_id,dw.fact_audience_aff_only_flag."date"
          from dw.fact_audience_aff_only_flag
          left join sub_set
          on sub_set.user_id=dw.fact_audience_aff_only_flag.user_id
          where "date" = '{2}'
          and sub_set.is_sub_active is null
          ) as cte_free
          join backend.user
          on backend.user.id=cte_free.user_id
          where not email like any ('%@molotov%', '%@temp-molotov%')
          ),

        cte_prog_type as (select user_id, MOVIE, SERIE, BROADCAST, DOCUMENTARY
        from dw.fact_audience_prog_type_affinity
        where date_day= '{2}'
        ),    
        cte_affinities as (select user_id,"Action & Aventure","Adaptation","Animaux","Art","Banque & Finance",
                                 "Beauté & Bien-être","Biopic","Cinéma","Comics & Animation","Comédies","Consommateurs",
                                 "Courts Métrages","Cuisine","Culture","Divertissement","Drames & Sentiments","Enfants",
                                 "Économie","Éducation","Famille","Histoire","Horreur","Information & Politique",
                                 "Investigation & Reportage","Jeux","Loisirs","Maison & Jardin","Mode","Musique",
                                 "Médicale","Nature","Santé & Médecine","Science Fiction & Fantastique",
                                 "Spectacles & Concerts","Sport & Fitness","Talk Show","Technologie & Science",
                                 "Thrillers & Policiers","Téléréalité","Voyage","Véhicules & Transports","Web & Gaming",
                                 "Westerns"
        from dw.fact_audience_aff_only_flag
        where "date"='{2}'),
        cte_event as (select testing_sample.user_id,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' and properties_json:product_equivalence_code::STRING='OCS' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='store_offer' and properties_json:product_equivalence_code::STRING='OCS' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='offer' and properties_json:product_equivalence_code::STRING='OCS' THEN properties_json:product_equivalence_code::STRING
              ELSE null END )as page_offer_ocs,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' and properties_json:product_equivalence_code::STRING='CINE_PLUS' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='store_offer' and properties_json:product_equivalence_code::STRING='CINE_PLUS' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='offer' and properties_json:product_equivalence_code::STRING='CINE_PLUS' THEN properties_json:product_equivalence_code::STRING
              ELSE null END )as page_offer_cine,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' and properties_json:product_equivalence_code::STRING='ADULT_SWIM' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='store_offer' and properties_json:product_equivalence_code::STRING='ADULT_SWIM' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='offer' and properties_json:product_equivalence_code::STRING='ADULT_SWIM' THEN properties_json:product_equivalence_code::STRING
              ELSE null END )as page_offer_adsw,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' and properties_json:product_equivalence_code::STRING='EXTENDED' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='store_offer' and properties_json:product_equivalence_code::STRING='EXTENDED' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='offer' and properties_json:product_equivalence_code::STRING='EXTENDED' THEN properties_json:product_equivalence_code::STRING
              ELSE null END )as page_offer_ext,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='store_offer' and properties_json:product_equivalence_code::STRING='OPTION_100H' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='store_offer' and properties_json:product_equivalence_code::STRING='OPTION_100H' THEN properties_json:product_equivalence_code::STRING
                   WHEN event_name='offer' and properties_json:product_equivalence_code::STRING='OPTION_100H' THEN properties_json:product_equivalence_code::STRING
              ELSE null END )as page_offer_mtv,


       count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT in (13,15,17,37,149) THEN properties_json:channel_name::STRING
                   WHEN event_name='program' and properties_json:channel_id::INT in (13,15,17,37,149) THEN properties_json:channel_name::STRING
                   ELSE null END) as program_ocs,
        count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT in (96,97,98,99,104,105) THEN properties_json:channel_name::STRING
                   WHEN event_name='program' and properties_json:channel_id::INT in (96,97,98,99,104,105) THEN properties_json:channel_name::STRING
                   ELSE null END) as program_cine,
        count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT in (139, 236) THEN properties_json:channel_name::STRING
                   WHEN event_name='program' and properties_json:channel_id::INT in (139, 236) THEN properties_json:channel_name::STRING
                   ELSE null END) as program_adsw,
        count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT in (select distinct channel_id
        from backend.channel 
        join backend.rel_tvbundle_channel
        on backend.channel.id=backend.rel_tvbundle_channel.channel_id
        where tvbundle_id in (26,32,85)) THEN properties_json:channel_name::STRING
                 WHEN event_name='program' and properties_json:channel_id::INT in (select distinct channel_id
        from backend.channel 
        join backend.rel_tvbundle_channel
        on backend.channel.id=backend.rel_tvbundle_channel.channel_id
        where tvbundle_id in (26,32,85)) THEN properties_json:channel_name::STRING
              ELSE null END) as program_ext,
        count(CASE WHEN event_name='page_viewed' and properties_json:channel_id::INT in (select channel_id
        from backend.channel 
        join backend.rel_tvbundle_channel
        on backend.channel.id=backend.rel_tvbundle_channel.channel_id
        where tvbundle_id in (90, 142, 146)) THEN properties_json:channel_name::STRING
                 WHEN event_name='program' and properties_json:channel_id::INT in (select channel_id
        from backend.channel 
        join backend.rel_tvbundle_channel
        on backend.channel.id=backend.rel_tvbundle_channel.channel_id
        where tvbundle_id in (90, 142, 146)) THEN properties_json:channel_name::STRING
              ELSE null END) as program_mtv,    

        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('ocs%') THEN properties_json:query::STRING
                   WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('ocs%') THEN properties_json:search_query::STRING
                   WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('ocs%') THEN properties_json:keyword::STRING
             ELSE null END) as search_ocs,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('cine%') THEN properties_json:query::STRING
                   WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('cine%') THEN properties_json:search_query::STRING
                   WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('cine%') THEN properties_json:keyword::STRING
             ELSE null END) as search_cine,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('rick %','%morty','adult%','%swim','toon%') THEN properties_json:query::STRING
                   WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('rick %','%morty','adult%','%swim','toon%') THEN properties_json:search_query::STRING
                   WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('rick %','%morty','adult%','%swim','toon%') THEN properties_json:keyword::STRING
             ELSE null END) as search_adsw,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('%molotov%','%extended%') THEN properties_json:query::STRING
                   WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('%molotov%','%extended%') THEN properties_json:search_query::STRING
                   WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('%molotov%','%extended%') THEN properties_json:keyword::STRING
             ELSE null END) as search_ext,
        count(CASE WHEN event_name='page_viewed' and properties_json:page_name::STRING='search' and lower(properties_json:query::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','%molotov%') THEN properties_json:query::STRING
                   WHEN event_name='search' and lower(properties_json:search_query::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','%molotov%') THEN properties_json:search_query::STRING
                   WHEN event_name='keyword_search' and lower(properties_json:keyword::STRING) like any ('%tf1%','%tmc%','%tfx%','%lci%','%m6%','%w9%','%6ter%','%gulli%','%molotov%') THEN properties_json:keyword::STRING
             ELSE null END) as search_mtv

        from testing_sample
        left join segment.snowpipe_all
        on segment.snowpipe_all.user_id=testing_sample.user_id
        where event_name in ('store_offer','offer','store','logged_in','bookmarks','page_viewed','keyword_search','search','search_result_clicked','program') 
        and event_date between '{1}' and '{0}'
        group by testing_sample.user_id)

        select testing_sample.user_id,
               socio_demo,
               MOVIE, 
               SERIE, 
               BROADCAST, 
               DOCUMENTARY,
               "Action & Aventure","Adaptation","Animaux","Art","Banque & Finance","Beauté & Bien-être","Biopic",
               "Cinéma","Comics & Animation","Comédies","Consommateurs","Courts Métrages","Cuisine","Culture","Divertissement",
               "Drames & Sentiments","Enfants","Économie","Éducation","Famille","Histoire","Horreur","Information & Politique",
               "Investigation & Reportage","Jeux","Loisirs","Maison & Jardin","Mode","Musique","Médicale","Nature","Santé & Médecine",
               "Science Fiction & Fantastique","Spectacles & Concerts","Sport & Fitness","Talk Show","Technologie & Science",
               "Thrillers & Policiers","Téléréalité","Voyage","Véhicules & Transports","Web & Gaming","Westerns",
               (CASE when page_offer_ocs is null then 0 else page_offer_ocs end) as page_offer_ocs,
               (CASE when page_offer_cine is null then 0 else page_offer_cine end) as page_offer_cine,
               (CASE when page_offer_adsw is null then 0 else page_offer_adsw end) as page_offer_adsw,
               (CASE when page_offer_ext is null then 0 else page_offer_ext end) as page_offer_ext,
               (CASE when page_offer_mtv is null then 0 else page_offer_mtv end) as page_offer_mtv,
               (CASE when program_ocs is null then 0 else program_ocs end) as program_ocs,
               (CASE when program_cine is null then 0 else program_cine end) as program_cine,
               (CASE when program_ext is null then 0 else program_ext end) as program_ext,
               (CASE when program_adsw is null then 0 else program_adsw end) as program_adsw,
               (CASE when program_mtv is null then 0 else program_mtv end) as program_mtv,
               (CASE when search_ocs is null then 0 else search_ocs end) as search_ocs,
               (CASE when search_cine is null then 0 else search_cine end) as search_cine,
               (CASE when search_adsw is null then 0 else search_adsw end) as search_adsw,
               (CASE when search_ext is null then 0 else search_ext end) as search_ext,
               (CASE when search_mtv is null then 0 else search_mtv end) as search_mtv
               from testing_sample
               left join cte_event
               on cte_event.user_id=testing_sample.user_id
               left join cte_prog_type
               on cte_prog_type.user_id=testing_sample.user_id
               left join cte_affinities
               on cte_affinities.user_id=testing_sample.user_id
               order by user_id desc
               """.format(date_1_format, date_15_format, now_format)

        users_offer_data_df = load_snowflake_query_df(self.spark, self.read_options, query_offer)
        users_offer_data_df = users_offer_data_df.fillna(0)

        return users_offer_data_df

    def better_offer(self, users_offer_data_df):
        list_offer = ["MTV", "CINE", "OCS", "ADSW", "EXT"]

        # Compute score for each offer using user information
        def score_offer(offer):
            if offer == "MTV":
                return F.col("broadcast") + F.col("page_offer_mtv") + F.col("program_mtv") + F.col("search_mtv")

            if offer == "CINE":
                return F.col("movie") + F.col("page_offer_cine") + F.col("program_cine") + F.col("search_cine")

            if offer == "OCS":
                return F.lit(0.75) * F.col("movie") + F.lit(0.75) * F.col("serie") + F.col("page_offer_ocs") + \
                    F.col("program_ocs") + F.col("search_ocs")

            if offer == "ADSW":
                return F.when((F.col("socio_demo") == "M_15-24") | (F.col("socio_demo") == "M_25-34"), F.lit(1)). \
                    otherwise(F.lit(0)) + F.col("page_offer_adsw") + F.col("program_adsw") + F.col(
                    "search_adsw") + \
                    F.col('"Comics & Animation"')

            if offer == "EXT":
                return F.col("broadcast") + F.col("page_offer_ext") + F.col("program_ext") + F.col("search_ext")

        for offer in list_offer:
            users_offer_data_df = users_offer_data_df.withColumn(offer, score_offer(offer))

        offer_with_score_df = users_offer_data_df.select("user_id", "MTV", "CINE", "OCS", "ADSW", "EXT")

        return offer_with_score_df

    def load_data_psub(self, predicted_data_df, offer_with_score_df):
        psub_offer_df = predicted_data_df. \
            where(F.col("VERSION") == 1). \
            join(offer_with_score_df, offer_with_score_df.user_id == predicted_data_df.user_id). \
            drop(offer_with_score_df.user_id). \
            select("user_id", "psub", "MTV",
                   "CINE", "OCS", "ADSW",
                   "EXT")

        no_offer_col = ["user_id", "psub"]
        self.list_offer = [x for x in psub_offer_df.columns if x not in no_offer_col]

        psub_offer_df = psub_offer_df.withColumn("user_id", F.col("user_id").cast(T.IntegerType()))
        for col in self.list_offer:
            psub_offer_df = psub_offer_df.withColumn(col, F.col(col).cast(T.DoubleType()))

        return psub_offer_df

    def psub_cluster(self, psub_offer_df):
        thresh_low = 0.2
        thresh_medium = 0.3
        thresh_high = 0.75
        psub_df_with_cluster = psub_offer_df.withColumn(
            "sub_cluster", F.when(F.col("psub") < thresh_low, "psub_low").
            when((F.col("psub") >= thresh_low) & (F.col("psub") < thresh_medium), "psub_low_medium").
            when((F.col("psub") >= thresh_medium) & (F.col("psub") < thresh_high), "psub_medium_high").
            otherwise("psub_high"))
        return psub_df_with_cluster

    def psub_offer_vect(self, psub_df_with_cluster):

        psub_df_with_cluster_pd = psub_df_with_cluster.toPandas()

        def better_offer():
            return lambda x: 'MTV' if (x == 0).all() else x.idxmax()

        def offer_vector(best_offer):
            vector = [0] * len(self.list_offer)
            index = self.list_offer.index(best_offer)
            vector[index] = 1

            return pd.Series(vector)

        def vectorize():
            return lambda x: offer_vector(x['offer'])

        psub_df_with_cluster_pd['offer'] = psub_df_with_cluster_pd[self.list_offer].apply(better_offer(), axis=1)
        psub_df_with_cluster_pd[self.list_offer] = psub_df_with_cluster_pd.apply(vectorize(), axis=1)

        psub_df_with_cluster_vect = self.spark.createDataFrame(psub_df_with_cluster_pd)
        return psub_df_with_cluster_vect

    def write_pred_to_snowflake(self, predicted_data_df, offer_with_score_df, psub_df_with_cluster_vect):
        final_user_pred_df = predicted_data_df. \
            join(offer_with_score_df, offer_with_score_df.user_id == predicted_data_df.user_id). \
            drop(offer_with_score_df.user_id). \
            withColumn("PRED_DATE", F.lit(self.now))

        list_columns = final_user_pred_df.columns
        list_columns.remove("VERSION")

        final_user_pred_df = final_user_pred_df.select(*list_columns, "VERSION")

        psub_df_with_cluster_vect = psub_df_with_cluster_vect.withColumn("PRED_DATE", F.lit(self.now))

        write_df_to_snowflake(final_user_pred_df, self.user_sub_pred_options, self.daily_sub_pred_table, "append")
        write_df_to_snowflake(psub_df_with_cluster_vect, self.user_sub_pred_options, self.daily_sub_pred_flag_table,
                              "overwrite")


if __name__ == "__main__":
    job = PsubJob()
    job.launch()
