import os
import math
from datetime import timedelta, datetime

from matplotlib import pyplot as plt
import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import functions as F
import pyspark.sql.types as T

from databricks_jobs.common import Job
from databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.helpers import \
    BucketizedFeature, IndexedFeature, RawFeature, CrossFeature, IdentityFeature, build_pipeline, \
    load_or_train_and_transform, threshold_parse, MatchingSetFeature
from databricks_jobs.jobs.utils.utils import get_snowflake_options, load_snowflake_table, write_df_to_snowflake, \
    load_snowflake_query_df


class PClickTrainingJob(Job):

    MAX_ITER = 150
    REG_PARAM = 0.1

    def __init__(self, *args, **kwargs):
        super(PClickTrainingJob, self).__init__(*args, **kwargs)

        self.model_directory = "dbfs:/Models"
        self.now = self.parse_date_args() - timedelta(days=2)
        self.lookback = timedelta(days=180)
        self.valid_lookback = timedelta(days=30)
        self.options = get_snowflake_options(self.conf, "PROD", **{"keep_column_case": "on"})
        self.write_options = get_snowflake_options(self.conf, "PROD", sf_schema="PUBLIC")

        mlflow.set_experiment("/Users/amorvan@molotov.tv/pclick")

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")

    def launch(self):
        """
        1 - Load the different tables (user, prog, channel)
        2 - Join the preprocessed elements (each kind of table has its own preprocessing)
        3 - Model cross feature are computed and the model is learned
        """
        with mlflow.start_run(tags={"date": str(self.now)}):
            # 1 - Get base tables
            user_df, prog_df, channel_df = self.initial_data_load()

            # 2 - Featurize the data
            # Sampling is required as we have ~2M lines per day
            association_index_df = load_snowflake_table(self.spark, self.options, "ML.PCLICK_TRAINING_LOG"). \
                where((F.col("DATE_DAY") >= self.now - self.lookback) &
                      (F.col("DATE_DAY") <= self.now - self.valid_lookback)). \
                sample(fraction=0.5)
            train_feature_df = self.combine_learning_log(association_index_df, user_df, prog_df, channel_df). \
                repartition(1500). \
                cache()
            test_association_index_df = load_snowflake_table(self.spark, self.options, "ML.PCLICK_TRAINING_LOG"). \
                where((F.col("DATE_DAY") >= self.now - self.valid_lookback) &
                      (F.col("DATE_DAY") <= self.now)). \
                repartition(1000). \
                sample(fraction=0.1)
            test_feature_df = self.combine_learning_log(test_association_index_df, user_df, prog_df, channel_df,
                                                        train=False). \
                cache()

            # 3 - Cross feature and model pipeline
            pipeline = self.featurize_dataset(train_feature_df)
            model = pipeline.fit(train_feature_df.withColumn("label", F.col("LABEL")))

            # 4 - Finally save the model
            self.mlflow_registration(model, "pclick_model")

            # Post training stats
            lr_model = model.stages[-1]
            self.log_training_stats(lr_model)

            out = model.transform(train_feature_df). \
                withColumnRenamed("LABEL", "label"). \
                limit(100000)  # We need a limited amount of info to compute feat importance
            self.log_feature_indexing(lr_model, out, compute_feature_count=False)

            # Compute raw scores on the test set
            output = model.transform(test_feature_df). \
                withColumnRenamed("LABEL", "label")
            predictions_and_labels = output.rdd.map(lambda lp: (lp.prediction, float(lp.label)))
            # Instantiate metrics object
            metrics = BinaryClassificationMetrics(predictions_and_labels)

            # Area under ROC curve
            mlflow.log_metric("Test Area under ROC", metrics.areaUnderROC)

    def inference(self):
        """
        Not used in the launch() of the job as it is training only

        This part of the job take cares of predicting a score per (user, prog_id, channel_id)
        Everything is then written in a table
        """
        pipeline_stage_name = "pclick_model"
        # Parse model paths and take the latest model date
        model_dates = filter(None,
                             map(lambda x: datetime.strptime(x.name.replace("/", ""), "%Y-%m-%d")
                                 if "-" in x.name else None,
                                 self.dbutils.fs.ls(os.path.join(self.model_directory, pipeline_stage_name))))
        latest_model_date = sorted(list(model_dates), reverse=True)[0].date()

        # 1 - Get base tables
        user_df, prog_df, channel_df = self.initial_data_load()

        query = f"""
            select USER_ID, parse_json(value):PROGRAM_ID as PROGRAM_ID, parse_json(value):CHANNEL_ID as CHANNEL_ID, UPDATE_DATE as DATE_DAY
            from "ML"."RECO_USER_PROGS_THIS_WEEK_LATEST", lateral flatten(recommendations)
            where UPDATE_DATE = '{self.now}'
            """
        interaction = load_snowflake_query_df(self.spark, self.options, query). \
            withColumn("RANK", F.lit(1)). \
            withColumn("LABEL", F.lit(0))

        # 2 - Featurize the data
        feature_df = self.combine_learning_log(interaction, user_df, prog_df, channel_df,
                                               train=False, model_date=latest_model_date)

        # 3 - Load model and add prediction column
        model_save_path = os.path.join(self.model_directory, pipeline_stage_name, str(latest_model_date))
        model = PipelineModel.load(model_save_path)
        # Does not work as mentioned with mlflow.pyfunc : https://issues.apache.org/jira/browse/SPARK-29952
        prediction_per_user = model.transform(feature_df)
        prediction = F.udf(lambda v: float(v[1]), T.FloatType())

        # 4 write the final result
        write_df_to_snowflake(prediction_per_user.select("USER_ID", "PROGRAM_ID", prediction("probability")),
                              self.write_options, "ML.PCLICK_USER_PROGRAM_PREDICTION", "overwrite")

    def combine_learning_log(self, association_index_df, user_df, program_df, channel_df, train=True, model_date=None):
        """
        We separate the feature pipeline for each type of feature as it saves on the volume of data in buckets

        feature_Pipeline are saved once learned.
        """
        # SPECIFICITY for user, convert the personality watch into watched personality list that can be used as a
        # BoW feature
        if model_date is None:
            model_date = self.now
        schema = T.ArrayType(T.StringType())
        user_df = user_df. \
            withColumn("WATCHED_PERSON_ID_1H+", threshold_parse("DURATION_PERSON_ID", F.lit(360))). \
            withColumn("WATCHED_PERSON_ID_10H+", threshold_parse("DURATION_PERSON_ID", F.lit(3600)))
        program_df = program_df. \
            withColumn("FAMOUS_CAST", F.coalesce(F.from_json("FAMOUS_CAST", schema), F.expr("array('')")))

        # a - user feature
        pipeline_stage_name = "user_feature"
        user_pipeline = self.featurize_user_features(user_df)
        user_feature_save_path = os.path.join(self.model_directory, pipeline_stage_name, str(model_date))
        featurized_user_df, user_model = load_or_train_and_transform(user_pipeline, user_df, user_feature_save_path,
                                                                     train=train)
        if train:
            self.mlflow_registration(user_model, pipeline_stage_name)

        # b - program feature
        pipeline_stage_name = "program_feature"
        program_pipeline = self.featurize_program_features(program_df)
        program_feature_save_path = os.path.join(self.model_directory, pipeline_stage_name, str(model_date))
        featurized_program_df, prog_model = load_or_train_and_transform(program_pipeline, program_df,
                                                                        program_feature_save_path, train=train)
        if train:
            self.mlflow_registration(prog_model, pipeline_stage_name)

        # c - channel feature
        pipeline_stage_name = "channel_feature"
        channel_pipeline = self.featurize_channel_features(channel_df)
        channel_feature_save_path = os.path.join(self.model_directory, pipeline_stage_name, str(model_date))
        featurized_channel_df, cha_model = load_or_train_and_transform(channel_pipeline, channel_df,
                                                                       channel_feature_save_path, train=train)
        if train:
            self.mlflow_registration(cha_model, pipeline_stage_name)

        return association_index_df. \
            join(featurized_user_df, on=["USER_ID", "DATE_DAY"]). \
            join(featurized_channel_df, on=["CHANNEL_ID", "DATE_DAY"]). \
            join(featurized_program_df, on=["PROGRAM_ID", "DATE_DAY"])

    def featurize_dataset(self, df):
        """
        input_df format

        program side
        "ID", "PRODUCTION_YEAR", "REF_PROGRAM_CATEGORY_ID", "REF_PROGRAM_KIND_ID", "PROGRAM_DURATION", "AFFINITY"
        + channel_feature : sociodemo watch, affinity watch, categoriy watch

        user side
        GENDER, AGE, DURATION x [ACTION_DEVICE_TYPE", "AFFINITY", "CATEGORY_ID", "PERSON_ID"]

        List of crosses :
        https://confluence-atlassian.molotov.net/display/~amorvan/Pclick+cross-features
        """

        # 0 - We define all the columns type, this will change how their are treated
        def filter_func(col_name):
            return [col for col in df.columns
                    if col.startswith(col_name) and "index" not in col and "vector" not in col]

        person_user_watch = filter_func("WATCHED_PERSON_ID_")
        affinity_user_watch = filter_func("DURATION_AFFINITY")
        category_user_watch = filter_func("DURATION_CATEGORY")
        device_user_watch = filter_func("DURATION_ACTION")
        watch_columns = affinity_user_watch + category_user_watch + device_user_watch

        socio_channel_watch = filter_func("channel_DURATION_socio")
        affinity_channel_watch = filter_func("channel_DURATION_AFFINITY")
        channel_watch_columns = affinity_channel_watch + socio_channel_watch

        # Sanity check on the filters
        for i, col_listing in enumerate(
                [affinity_user_watch, category_user_watch, device_user_watch, socio_channel_watch,
                 affinity_channel_watch, person_user_watch]):
            assert len(col_listing) > 0, f"The {i}th set of feature is empty"

        continuous_features = {
            col_name: RawFeature(col_name)
            for col_name in ["AGE", "PRODUCTION_YEAR", "PROGRAM_DURATION", "TOTAL_CELEB_POINTS", "EXTERNAL_RATING"] +
            watch_columns + channel_watch_columns + person_user_watch
        }
        string_features = {col_name: RawFeature(col_name)
                           for col_name in ["GENDER", "AFFINITY"]}
        index_features = {col_name: IndexedFeature(col_name)
                          for col_name in ["REF_PROGRAM_CATEGORY_ID", "RANK"]}
        all_features = {}
        list(map(lambda x: all_features.update(x),
                 [continuous_features, string_features, index_features]))

        feature_rank = index_features["RANK"]
        # 2 - One hot encoding of all indexed features
        one_hot_encoder = OneHotEncoder(dropLast=False)
        one_hot_encoder.setInputCols([feature.indexed_column for feature in [feature_rank]])
        one_hot_encoder.setOutputCols([feature.vectorized_column for feature in [feature_rank]])

        # 3 - Cross feature génération based on vectorized representations
        # 3.a - some crosses are used afterward, so build it first
        age_gender_cross = CrossFeature("AGE", "GENDER")
        socio_demo_name = age_gender_cross.name
        all_features[socio_demo_name] = IdentityFeature(socio_demo_name)

        prog_quality_cross = CrossFeature("EXTERNAL_RATING", "TOTAL_CELEB_POINTS")
        prog_quality_name = prog_quality_cross.name
        all_features[prog_quality_name] = IdentityFeature(prog_quality_name)

        # 3.b - Generic cross features
        user_prog_cross_features = \
            [CrossFeature(socio_demo_name, "REF_PROGRAM_CATEGORY_ID"),
             CrossFeature(socio_demo_name, "PRODUCTION_YEAR"),
             CrossFeature(socio_demo_name, "AFFINITY")] + \
            [CrossFeature("AFFINITY", col) for col in watch_columns] + \
            [CrossFeature("REF_PROGRAM_CATEGORY_ID", col) for col in watch_columns]

        user_channel_cross_features = \
            [CrossFeature(col_1, col_2) for col_1 in affinity_user_watch for col_2 in affinity_channel_watch] + \
            [CrossFeature(socio_demo_name, col_2) for col_2 in socio_channel_watch]

        prog_prog_cross_features = \
            [CrossFeature(prog_quality_name, "PRODUCTION_YEAR"),
             ]

        cross_features = [age_gender_cross, prog_quality_cross] + user_prog_cross_features + \
            user_channel_cross_features + prog_prog_cross_features

        # 3.c - Specific cross feature : matching casting and personality watch
        matching_cast_features = [
            MatchingSetFeature(f, "FAMOUS_CAST") for f in person_user_watch
        ]

        # 4 - Recombine everything into a single vector
        assembler = VectorAssembler(
            inputCols=[cf.name for cf in cross_features] + [f.vectorized_column for f in all_features.values()] +
                      [op.name for op in matching_cast_features],
            outputCol="features", handleInvalid="keep")

        # 5 -Define log reg
        lr = LogisticRegression(maxIter=self.MAX_ITER, regParam=self.REG_PARAM, labelCol="label", fitIntercept=False,
                                maxBlockSizeInMB=128)
        mlflow.log_params({"maxIter": self.MAX_ITER, "regParam": self.REG_PARAM})

        pipeline = Pipeline(stages=[one_hot_encoder] +
                                   [cross_feature.build_interaction(all_features) for cross_feature in cross_features] +
                                   [op.build_interaction() for op in matching_cast_features] +
                                   [assembler, lr])

        mlflow.log_text("\n".join(sorted([f.vectorized_column for f in all_features.values()])), "basic features.txt")
        mlflow.log_text("\n".join(sorted([cf.mlflow_name for cf in cross_features + matching_cast_features])),
                        "cross features.txt")

        return pipeline

    ####################
    #     Helpers
    ####################

    def mlflow_registration(self, model: PipelineModel, name: str):
        save_model_path = os.path.join(self.model_directory, name, str(self.now))
        model.write().overwrite().save(save_model_path)
        # The true path is not used to save the model, making it unavailable for a simple retrieval
        mlflow.spark.save_model(model, save_model_path)
        try:
            mlflow.spark.log_model(model, save_model_path)
        except mlflow.exceptions.MlflowException:
            self.logger.info(f"Mlflow model logging failed for model {name}")

    @staticmethod
    def featurize_user_features(df):
        # We define all the columns type, this will change how their are treated
        affinity_user_watch = [col for col in df.columns if col.startswith("DURATION_AFFINITY")]
        category_user_watch = [col for col in df.columns if col.startswith("DURATION_CATEGORY")]
        device_user_watch = [col for col in df.columns if col.startswith("DURATION_ACTION")]
        personality_user_watch = [col for col in df.columns if col.startswith("WATCHED_PERSON_ID_")]
        assert len(personality_user_watch) == 2
        # Retired as the format is a dictionary, not yet supported
        # person_user_watch = [col for col in df.columns if col.startswith("DURATION_PERSON")]
        watch_columns = affinity_user_watch + category_user_watch + device_user_watch

        # Sanity check on the filters
        for i, col_listing in enumerate(
                [affinity_user_watch, category_user_watch, device_user_watch]):
            assert len(col_listing) > 0, f"The {i}th set of feature is empty"

        bucketized_features = {}
        continuous_features = {
            col_name: RawFeature(col_name)
            for col_name in ["AGE"] + watch_columns
        }
        string_features = {col_name: RawFeature(col_name)
                           for col_name in ["GENDER"]}
        index_features = {}
        bow_features = {col_name: RawFeature(col_name)
                        for col_name in personality_user_watch}
        user_pipeline = build_pipeline(bucketized_features, continuous_features, string_features,
                                       index_features, bow_features)

        mlflow.log_dict({
            "user_bucketized_features": list(bucketized_features.keys()),
            "user_continuous_features": list(continuous_features.keys()),
            "user_string_features": list(string_features.keys()),
            "user_index_features": list(index_features.keys()),
            "user_bow_features": list(bow_features.keys())
        }, "user_feature.json")

        return user_pipeline

    @staticmethod
    def featurize_program_features(df):
        bucketized_features = {
            "PRODUCTION_YEAR": BucketizedFeature("PRODUCTION_YEAR", (-math.inf, 0, 1970, 1985, 2000, 2010, 2018,
                                                                     math.inf)),
            "EXTERNAL_RATING": BucketizedFeature("EXTERNAL_RATING", (-math.inf, 0, 2, 3.3, 4, math.inf)),
            "TOTAL_CELEB_POINTS": BucketizedFeature("TOTAL_CELEB_POINTS", (-math.inf, 0, 10000, 30000, 50000,
                                                                           math.inf)),
        }
        continuous_features = {
            col_name: RawFeature(col_name)
            for col_name in ["PROGRAM_DURATION"]
        }
        string_features = {col_name: RawFeature(col_name)
                           for col_name in ["AFFINITY"]}
        index_features = {col_name: IndexedFeature(col_name)
                          for col_name in ["REF_PROGRAM_CATEGORY_ID", "REF_PROGRAM_KIND_ID"]}
        bow_features = {col_name: RawFeature(col_name) for col_name in ["FAMOUS_CAST"]}

        program_pipeline = build_pipeline(bucketized_features, continuous_features, string_features, index_features,
                                          bow_features)

        mlflow.log_dict({
            "program_bucketized_features": list(bucketized_features.keys()),
            "program_continuous_features": list(continuous_features.keys()),
            "program_string_features": list(string_features.keys()),
            "program_index_features": list(index_features.keys()),
            "program_bow_features": list(bow_features.keys())
        }, "program_feature.json")

        return program_pipeline

    @staticmethod
    def featurize_channel_features(df):
        socio_channel_watch = [col for col in df.columns if col.startswith("channel_DURATION_socio")]
        affinity_channel_watch = [col for col in df.columns if col.startswith("channel_DURATION_AFFINITY")]
        channel_watch_columns = affinity_channel_watch + socio_channel_watch

        # Sanity check on the filters
        for i, col_listing in enumerate([socio_channel_watch, affinity_channel_watch]):
            assert len(col_listing) > 0, f"The {i}th set of feature is empty"

        bucketized_features = {}
        continuous_features = {
            col_name: RawFeature(col_name)
            for col_name in channel_watch_columns
        }
        string_features = {}
        index_features = {}
        bow_features = {}

        channel_pipeline = build_pipeline(bucketized_features, continuous_features, string_features,
                                          index_features, bow_features)

        mlflow.log_dict({
            "channel_bucketized_features": list(bucketized_features.keys()),
            "channel_continuous_features": list(continuous_features.keys()),
            "channel_string_features": list(string_features.keys()),
            "channel_index_features": list(index_features.keys()),
            "channel_bow_features": list(bow_features.keys())
        }, "channel_feature.json")

        return channel_pipeline

    @staticmethod
    def log_training_stats(model):
        """
        We extract from the LR model the following :
        - ROC curve
        - Training loss summary
        - Weighted precision and recall
        """
        training_summary = model.summary

        roc = training_summary.roc.toPandas()
        fig, ax = plt.subplots()
        ax.plot(roc['FPR'], roc['TPR'])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_title('ROC Curve')
        mlflow.log_figure(fig, "training_roc_summary.png")

        fig, ax = plt.subplots()
        metric_history = training_summary.objectiveHistory
        ax.plot(range(len(metric_history)), metric_history)
        ax.set_ylabel('objective')
        ax.set_xlabel('epoch')
        ax.set_title('metric_history')
        mlflow.log_figure(fig, "metric_history.png")

        mlflow.log_metric("areaUnderROC", training_summary.areaUnderROC)
        mlflow.log_metric("weightedPrecision", training_summary.weightedPrecision)
        mlflow.log_metric("weightedRecall", training_summary.weightedRecall)

    def log_feature_indexing(self, model, output, compute_feature_count=True):
        """
        Small util to create a rather large dict with the model weights attached to the feature name
        Very useful to understand if a specific feature or cross feature is used or not by the model
        """
        metadata = output.select("features"). \
            schema[0].metadata.get('ml_attr').get('attrs')

        numeric_metadata = metadata.get('numeric')
        binary_metadata = metadata.get('binary')
        nominal_metadata = metadata.get('nominal')
        # Given the kind of feature engineering we are doing, we should not have this kind of feature left
        assert nominal_metadata is None or len(nominal_metadata) == 0

        ordered_identity = list(map(lambda x: x["name"], sorted(numeric_metadata + binary_metadata,
                                                                key=lambda k: k["idx"])))
        coefs = model.coefficients.toArray().tolist()
        feat_importance = {k: v for k, v in zip(ordered_identity, coefs)}
        mlflow.log_dict(feat_importance, "feature_importance.json")

        if compute_feature_count:
            # We sum all the feature vector (all indexes) in order to get the count of each singular feature
            # A final count of 0 would mean its occurence < 1 / 1e5
            total = output.rdd.map(lambda x: DenseVector(x.features)).sum()
            metadata = (binary_metadata if binary_metadata else []) + (numeric_metadata if numeric_metadata else []) + \
                       (nominal_metadata if nominal_metadata else [])
            ordered_identity = list(map(lambda x: x["name"], sorted(metadata, key=lambda k: k["idx"])))
            features_count = {k: v for k, v in zip(ordered_identity, total)}

            mlflow.log_dict(features_count, "feature_count.json")

    def initial_data_load(self):
        user_df = load_snowflake_table(self.spark, self.options, "ML.USER_FEATURE_LOG"). \
            fillna(value="U", subset=["GENDER"]). \
            fillna(value=0)
        prog_df = load_snowflake_table(self.spark, self.options, "ML.PROGRAM_FEATURE_LOG"). \
            fillna(value="[]", subset=["FAMOUS_CAST"]). \
            fillna(value=-1, subset=["EXTERNAL_RATING", "TOTAL_CELEB_POINTS", "PRODUCTION_YEAR"]).\
            fillna(value=0)
        channel_df = load_snowflake_table(self.spark, self.options, "ML.CHANNEL_FEATURE_LOG"). \
            fillna(value=0)
        return user_df, prog_df, channel_df


if __name__ == "__main__":
    job = PClickTrainingJob()
    job.launch()
