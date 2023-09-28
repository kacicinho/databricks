from datetime import timedelta

import mlflow
import numpy as np
import pyspark.sql.types as T
import sklearn
from pyspark.sql import functions as F
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

from databricks_jobs.common import Job
from databricks_jobs.jobs.ml_models_jobs.training_psub_job.pipeline_lib import Bining, MultiColumnLabelEncoder, SklearnModelWrapper
from databricks_jobs.jobs.utils.utils import load_snowflake_query_df


class TrainingPsubJob(Job):
    INTERVAL = 365
    OFFSET = 30

    def __init__(self, *args, **kwargs):
        super(TrainingPsubJob, self).__init__(*args, **kwargs)

        self.snowflake_user = self.conf.get("user_name", "")
        self.snowflake_password = self.conf.get("password", "")

        self.sf_database = self.conf.get("env", "DEV")

        self.now = self.parse_date_args()
        self.end_date = self.now - timedelta(days=self.OFFSET)
        self.start_date = self.now - timedelta(days=self.INTERVAL)
        self.logger.info(f"Running on date : {self.now}")

        if self.sf_database == 'DEV':
            self.sf_wharehouse = 'PROD_WH'
            self.sf_role = 'DEV'

        elif self.sf_database == 'PROD':
            self.sf_wharehouse = 'PROD_WH'
            self.sf_role = 'PROD'

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

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        training_data_df_pd = self.prepare_training_data()
        self.train_model(training_data_df_pd)

    def prepare_training_data(self):

        start_date_format = '{}-{}-{}'.format(self.start_date.year, self.start_date.month, self.start_date.day)
        end_date_format = '{}-{}-{}'.format(self.end_date.year, self.end_date.month, self.end_date.day)

        query_data = """with cte as (select user_id,subscription_date,subscription_datetime,equivalence_code,reg_date,row_number() over(partition by user_id order by subscription_date) as rang
        from backend.subscribers
        where reg_date<=subscription_date-15
          and equivalence_code in ('OPTION_100H','EXTENDED','CINE_PLUS','OCS','ADULT_SWIM')
          and platform not in ('molotov','digital_virgo')
          and extract(year from subscription_date)>=2019
          and subscription_date between '{0}' and '{1}'
        order by user_id desc),

        cte_sub as (select user_id,subscription_date,subscription_datetime,equivalence_code,reg_date,
        CASE when equivalence_code is not null then 1
        END as is_sub
        from cte 
        where rang=1),

        cte_free_all as ( with cte_free as(select user_id, Null as subscription_date,Null as subscription_datetime,'FREE' as equivalence_code,reg_date,0 as is_sub 
        from
        (select dw.fact_registered.user_id,dw.fact_registered.date_day as reg_date
        from dw.fact_registered
        left join backend.subscribers
        on backend.subscribers.user_id=dw.fact_registered.user_id
        join backend.user
        on backend.user.id=dw.fact_registered.user_id
        where event_name='registered'
        and dw.fact_registered.reg_type = 'B2C'
        and dw.fact_registered.active_reg = True
        and backend.subscribers.equivalence_code is null
        and backend.user.email not like '%molotov%'
        and dw.fact_registered.date_day <= current_date()-15
        and extract(year from dw.fact_registered.date_day) >=2019
        and dw.fact_registered.date_day between '{0}' and '{1}')

        )
        select user_id,subscription_date,subscription_datetime,equivalence_code,reg_date,is_sub
        from(
        select cte_free.user_id,rfm_end_date as subscription_date,rfm_end_date as subscription_datetime,equivalence_code,reg_date,is_sub,
        row_number() over(partition by cte_free.user_id order by F desc,M desc,R asc) as rang
        from cte_free
        inner join dw.fact_rfm_window_slide_01d_size_28d
        on dw.fact_rfm_window_slide_01d_size_28d.user_id=cte_free.user_id
        and rfm_end_date >= dateadd(day, 15,reg_date))
        where rang=1),

        cte_free_date as (
            select * 
            from cte_free_all
            ),

        cte_sub_socio as (with temp as (select * from cte_sub
        UNION
        select * from cte_free_date)

        select temp.user_id,subscription_date,subscription_datetime,equivalence_code,reg_date,is_sub,socio_demo
        from temp
        inner join backend.user
        on temp.user_id=backend.user.id),


        cte_devices as (select user_id,count(*) as connected_devices
        from (select distinct cte_sub_socio.user_id,action_device_type,device_type,device_id 
              from dw.fact_watch 
              join cte_sub_socio 
              on cte_sub_socio.user_id=dw.fact_watch.user_id
              where date_day between dateadd(day,-15,subscription_date) and subscription_date)
             group by user_id),

        cte_sub_watch as (select cte_sub_socio.user_id,
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
        join cte_sub_socio
        on dw.fact_watch.user_id=cte_sub_socio.user_id
        where date_day between dateadd(day,-15,subscription_date) and subscription_date
          and duration_over_30_sec
        group by cte_sub_socio.user_id
                         ),

        cte_cross_features as ( with cte_device_cat as (select cte_sub_socio.user_id,duration,concat(action_device_type,'_',name) as device_cat
        from dw.fact_watch
        join backend.ref_program_category
        on backend.ref_program_category.id=dw.fact_watch.category_id
        join cte_sub_socio
        on dw.fact_watch.user_id=cte_sub_socio.user_id
        where date_day between dateadd(day,-15,subscription_date) and subscription_date
        and duration_over_30_sec
        and action_device_type in('tv','desktop','tablet','phone')
        and category_id in (1,2,3,4,5,6,8,9)
        )

        select *
        from cte_device_cat
        pivot(sum(duration) for device_cat in ('tv_Sport','tv_Documentaires', 'tablet_Films', 'desktop_Documentaires', 'desktop_Informations', 'tv_Enfants', 'phone_Sport', 'tv_Informations', 'desktop_Séries', 'desktop_Films', 'tablet_Informations', 'tablet_Séries', 'phone_Culture', 'phone_Séries', 'phone_Films', 'tablet_Enfants', 'tablet_Sport', 'phone_Divertissement', 'tv_Films', 'tablet_Divertissement', 'tv_Culture', 'phone_Documentaires', 'phone_Informations', 'desktop_Culture', 'desktop_Divertissement', 'phone_Enfants', 'tablet_Documentaires', 'tablet_Culture', 'desktop_Sport', 'tv_Divertissement', 'tv_Séries', 'desktop_Enfants')) as p(user_id,tv_Sport_duration,tv_Documentaires_duration, tablet_Films_duration,desktop_Documentaires_duration, desktop_Informations_duration, tv_Enfants_duration, phone_Sport_duration, tv_Informations_duration, desktop_Series_duration, desktop_Films_duration, tablet_Informations_duration, tablet_Series_duration, phone_Culture_duration, phone_Series_duration, phone_Films_duration, tablet_Enfants_duration, tablet_Sport_duration, phone_Divertissement_duration, tv_Films_duration, tablet_Divertissement_duration, tv_Culture_duration,phone_Documentaires_duration, phone_Informations_duration, desktop_Culture_duration, desktop_Divertissement_duration, phone_Enfants_duration, tablet_Documentaires_duration, tablet_Culture_duration, desktop_Sport_duration, tv_Divertissement_duration, tv_Series_duration, desktop_Enfants_duration)
        order by user_id),

        cte_event as (select cte_sub_socio.user_id,
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
        from cte_sub_socio
        join segment.snowpipe_all
        on segment.snowpipe_all.user_id=cte_sub_socio.user_id
        where event_name in ('store_offer','offer','store','logged_in','bookmarks','page_viewed','keyword_search','search','search_result_clicked','program') 
        and event_date between dateadd(day,-15,subscription_date) and subscription_date
        and timestamp<dateadd(minute,-10,subscription_datetime)
        group by cte_sub_socio.user_id)

        select cte_sub_socio.user_id,
               socio_demo,
               subscription_date,
               subscription_datetime,
               equivalence_code,
               reg_date,
               is_sub,
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
               tv_Sport_duration,tv_Documentaires_duration, tablet_Films_duration,desktop_Documentaires_duration, desktop_Informations_duration, tv_Enfants_duration, phone_Sport_duration,                  tv_Informations_duration, desktop_Series_duration, desktop_Films_duration, tablet_Informations_duration, tablet_Series_duration,
               phone_Culture_duration, phone_Series_duration, phone_Films_duration, tablet_Enfants_duration, tablet_Sport_duration, phone_Divertissement_duration, tv_Films_duration,                        tablet_Divertissement_duration, tv_Culture_duration,phone_Documentaires_duration, phone_Informations_duration, desktop_Culture_duration, desktop_Divertissement_duration,                    phone_Enfants_duration, tablet_Documentaires_duration, tablet_Culture_duration, desktop_Sport_duration, tv_Divertissement_duration, 
               tv_Series_duration, desktop_Enfants_duration,
               (CASE when page_offer is null then 0 else page_offer end) as page_offer,
               (CASE when program is null then 0 else program end) as program,
               (CASE when search is null then 0 else search end) as search,
               (CASE when bookmark_page is null then 0 else bookmark_page end) as bookmark_page,
               (CASE when app_login is null then 0 else app_login end) as app_login
               from cte_sub_socio
               left join cte_devices
               on cte_devices.user_id=cte_sub_socio.user_id
               left join cte_sub_watch
               on cte_sub_watch.user_id=cte_sub_socio.user_id
               left join cte_event
               on cte_event.user_id=cte_sub_socio.user_id
               left join cte_cross_features
               on cte_cross_features.user_id=cte_sub_socio.user_id
               order by cte_sub_socio.user_id desc

               """.format(start_date_format, end_date_format)

        train_data_df = load_snowflake_query_df(self.spark, self.read_options, query_data)

        decimal_col = ["USER_ID", "IS_SUB", "CONNECTED_DEVICES", "TOT_DURATION", "MOVIE_DURATION", "SERIE_DURATION",
                       "SPORT_DURATION", "INFORMATION_DURATION", "ENTERTAINEMENT_DURATION", "CHILD_DURATION",
                       "DOCUMENTARY_DURATION", "CULTURE_DURATION", "TV_DURATION", "DESKTOP_DURATION",
                       "PHONE_DURATION",
                       "TABLET_DURATION", "TV_SPORT_DURATION", "TV_DOCUMENTAIRES_DURATION", "TABLET_FILMS_DURATION",
                       "DESKTOP_DOCUMENTAIRES_DURATION", "DESKTOP_INFORMATIONS_DURATION", "TV_ENFANTS_DURATION",
                       "PHONE_SPORT_DURATION", "TV_INFORMATIONS_DURATION", "DESKTOP_SERIES_DURATION",
                       "DESKTOP_FILMS_DURATION",
                       "TABLET_INFORMATIONS_DURATION", "TABLET_SERIES_DURATION", "PHONE_CULTURE_DURATION",
                       "PHONE_SERIES_DURATION", "PHONE_FILMS_DURATION", "TABLET_ENFANTS_DURATION",
                       "TABLET_SPORT_DURATION",
                       "PHONE_DIVERTISSEMENT_DURATION", "TV_FILMS_DURATION", "TABLET_DIVERTISSEMENT_DURATION",
                       "TV_CULTURE_DURATION", "PHONE_DOCUMENTAIRES_DURATION", "PHONE_INFORMATIONS_DURATION",
                       "DESKTOP_CULTURE_DURATION", "DESKTOP_DIVERTISSEMENT_DURATION", "PHONE_ENFANTS_DURATION",
                       "TABLET_DOCUMENTAIRES_DURATION", "TABLET_CULTURE_DURATION", "DESKTOP_SPORT_DURATION",
                       "TV_DIVERTISSEMENT_DURATION", "TV_SERIES_DURATION", "DESKTOP_ENFANTS_DURATION", "PAGE_OFFER",
                       "PROGRAM",
                       "SEARCH", "BOOKMARK_PAGE", "APP_LOGIN"]

        for col in decimal_col:
            train_data_df = train_data_df.withColumn(col, F.col(col).cast(T.DoubleType()))
        train_data_df_pd = train_data_df.toPandas()
        train_data_df_pd.columns = train_data_df_pd.columns.str.lower()

        return train_data_df_pd

    def data_processing(self, data):
        data_process = data.copy()
        duration_column = [col for col in data.columns if 'duration' in col.split('_')]
        column_drop = ['equivalence_code', 'user_id', 'reg_date', 'subscription_date', 'subscription_datetime',
                       'seniority']
        categorical_column = ['socio_demo'] + duration_column
        numerical_column = ['connected_devices', 'page_offer', 'program', 'search', 'bookmark_page', 'app_login']

        data_process[duration_column] = data_process[duration_column].fillna(0)
        data_process = data_process.drop(columns=column_drop, errors="ignore")

        X = data_process.drop(columns=['is_sub'])
        y = data_process['is_sub']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        return X_train, y_train, X_test, y_test, duration_column, categorical_column, numerical_column

    def train_model(self, train_data_df_pd):
        seed = 2
        np.random.seed(seed)
        X_train, y_train, X_test, y_test, duration_column, categorical_column, numerical_column = self.data_processing(
            train_data_df_pd)
        ratio_train = y_train.value_counts()[0] / y_train.value_counts()[1]
        weight_train = {0: 1, 1: ratio_train}

        ratio_test = y_test.value_counts()[0] / y_test.value_counts()[1]
        weight_test = {0: 1, 1: ratio_test}
        class_weight_test = weight_test
        weights_by_class_test = [1 if y == 0 else class_weight_test[1] for y in y_test]

        mlflow.set_experiment("/Users/mgayas@molotov.tv/Psub_training")
        with mlflow.start_run(run_name='MODELV2'):
            loss = 'log'
            penalty = 'elasticnet'
            alpha = 0.000001
            numeric_transformer = Pipeline(
                steps=[('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                       ('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('binarizer', Bining(duration_column)),
                                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_column),
                    ('cat', categorical_transformer, categorical_column)])
            model_v2 = Pipeline([('preprocessor', preprocessor),
                                 ('log_reg',
                                  SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, class_weight=weight_train))])

            model_v2.fit(X_train, y_train)

            # summarize feature importance
            log_reg = model_v2.named_steps['log_reg']
            importance = log_reg.coef_[0]
            feature_list = model_v2[:-1].get_feature_names_out()
            feat_importance = {k: v for k, v in zip(feature_list, importance)}

            # Model predictions
            predictions_test = model_v2.predict(X_test)

            # Log model parameters
            mlflow.log_param("loss", loss)
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("random_seed", seed)
            mlflow.log_param("start", self.start_date)
            mlflow.log_param("end", self.end_date)
            mlflow.log_dict(feat_importance, "feature_importance.json")

            # Create metrics
            accuracy_test = sklearn.metrics.accuracy_score(y_test, predictions_test,
                                                           sample_weight=weights_by_class_test)
            recall_test = sklearn.metrics.recall_score(y_test, predictions_test, sample_weight=weights_by_class_test)

            # Log metrics
            mlflow.log_metric("accuracy_test", accuracy_test)
            mlflow.log_metric("recall_test", recall_test)

            wrappedModel = SklearnModelWrapper(model_v2)
            # Log model
            mlflow.sklearn.log_model(
                sk_model=wrappedModel,
                artifact_path="psub-model",
                registered_model_name="Psub-v2"
            )

            with mlflow.start_run(run_name='MODELV1', nested=True):
                loss = 'log'
                penalty = 'l1'
                alpha = 0.0001
                model_v1 = Pipeline([('binarizer', Bining(duration_column)),
                                     ('labelencoding', MultiColumnLabelEncoder(categorical_column)),
                                     ('scaler', StandardScaler()),
                                     ('log_reg', SGDClassifier(loss=loss, penalty=penalty, alpha=alpha,
                                                               class_weight=weight_train))])

                model_v1.fit(X_train, y_train)

                # summarize feature importance
                log_reg = model_v1.named_steps['log_reg']
                importance = log_reg.coef_[0]
                feature_list = model_v1[:-1].get_feature_names_out()
                feat_importance = {k: v for k, v in zip(feature_list, importance)}

                # Model predictions
                predictions_test = model_v1.predict(X_test)

                # Log model parameters
                mlflow.log_param("loss", loss)
                mlflow.log_param("penalty", penalty)
                mlflow.log_param("random_seed", seed)
                mlflow.log_param("start", self.start_date)
                mlflow.log_param("end", self.end_date)
                mlflow.log_dict(feat_importance, "feature_importance.json")

                # Create metrics
                accuracy_test = sklearn.metrics.accuracy_score(y_test, predictions_test,
                                                               sample_weight=weights_by_class_test)
                recall_test = sklearn.metrics.recall_score(y_test, predictions_test,
                                                           sample_weight=weights_by_class_test)

                # Log metrics
                mlflow.log_metric("accuracy_test", accuracy_test)
                mlflow.log_metric("recall_test", recall_test)

                wrappedModel = SklearnModelWrapper(model_v1)
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=wrappedModel,
                    artifact_path="psub-model",
                    registered_model_name="Psub"
                )


if __name__ == "__main__":
    job = TrainingPsubJob()
    job.launch()
