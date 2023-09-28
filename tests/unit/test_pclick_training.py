import os
import datetime
import shutil
from unittest import TestCase, mock
import mlflow

from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pyspark.sql.functions as F
import pyspark.sql.types as T

from databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint import PClickTrainingJob
from databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.helpers import load_or_train_and_transform, \
    build_interaction_log_from_broadcast, threshold_parse, MatchingSetFeature
from databricks_jobs.jobs.misc_jobs.sample.entrypoint import SampleJob
from tests.unit.utils.mocks import create_spark_df_from_data, multiplex_mock


class TestPrePclickTrainingJob(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        It could be necessary to have a local mlflow and add the following to the setupClass
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("pclick-training")
        """
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = PClickTrainingJob()
        cls.job.MAX_ITER = 1
        cls.job.model_directory = "models"

    def tearDown(self) -> None:
        for dir_ in [self.job.model_directory, "my_pclick"]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)

    def setUp(self) -> None:
        if not os.path.exists(self.job.model_directory):
            os.mkdir(self.job.model_directory)
        now = self.job.now
        user_data = {
            "USER_ID": [1, 2, 3],
            "GENDER": ["F", "M", "U"],
            "AGE": [12, 34, 4],
            "DURATION_AFFINITY=Action": [2340, 0, 0],
            "DURATION_ACTION_DEVICE_TYPE=tv": [234, 1200, 0],
            "DURATION_CATEGORY=1": [34, 1, 90],
            "DURATION_PERSON_ID": ['{"23": 32452, "2": 1}', None, ""],
            "DATE_DAY": [now, now, now]
        }
        prog_data = {
            "PROGRAM_ID": [1, 2, 0],
            "PRODUCTION_YEAR": [1990, 2008, 1970],
            "REF_PROGRAM_CATEGORY_ID": [1, 2, 3],
            "REF_PROGRAM_KIND_ID": [1, 2, 3],
            "TOTAL_CELEB_POINTS": [33, 1245, None],
            "EXTERNAL_RATING": [None, 3.4, None],
            "PROGRAM_DURATION": [1200, 2340, 600],
            "AFFINITY": ["Action", "Cuisine", "Horreur"],
            "FAMOUS_CAST": ['["1"]', '["3"]', ''],
            "DATE_DAY": [now, now, now]
        }
        channel_data = {
            "CHANNEL_ID": [0, 1, 2],
            "channel_DURATION_socio=F_4": [1, 0, 4],
            "channel_DURATION_socio=H_4": [1, 0, 4],
            "channel_DURATION_AFFINITY=Action": [20, 0, 33],
            "DATE_DAY": [now, now, now]
        }
        self.user_df = create_spark_df_from_data(self.job.spark, user_data)
        self.prog_df = create_spark_df_from_data(self.job.spark, prog_data)
        self.channel_df = create_spark_df_from_data(self.job.spark, channel_data)

    def test_user_featurizer(self):
        user_df = self.user_df. \
            withColumn("WATCHED_PERSON_ID_1H+", threshold_parse("DURATION_PERSON_ID", F.lit(3600))). \
            withColumn("WATCHED_PERSON_ID_10H+", threshold_parse("DURATION_PERSON_ID", F.lit(36000)))
        pipeline = self.job.featurize_user_features(user_df)
        model = pipeline.fit(user_df)
        featurized_user_df = model.transform(user_df)
        featurized_user_df.collect()

    def test_prog_featurizer(self):
        schema = T.ArrayType(T.StringType())
        prog_df = self.prog_df. \
            withColumn("FAMOUS_CAST", F.coalesce(F.from_json("FAMOUS_CAST", schema), F.expr('array("")')))
        pipeline = self.job.featurize_program_features(prog_df)
        model = pipeline.fit(prog_df)
        featurized_prog_df = model.transform(prog_df)
        featurized_prog_df.collect()

    def test_channel_featurizer(self):
        pipeline = self.job.featurize_channel_features(self.channel_df)
        model = pipeline.fit(self.channel_df)
        featurized_chan_df = model.transform(self.channel_df)
        featurized_chan_df.collect()

    def test_feature_creation(self):
        now = self.job.now
        data = {"USER_ID": [1, 2, 3], "PROGRAM_ID": [0, 1, 2], "CHANNEL_ID": [1, 2, 0],
                "label": [0, 0, 1], "RANK": [1, 4, 5],
                "DATE_DAY": [now, now, now]}

        association_index_df = create_spark_df_from_data(self.job.spark, data)
        with mock.patch.multiple("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint",
                                 mlflow=mock.MagicMock(mlflow)):
            df = self.job.combine_learning_log(association_index_df, self.user_df, self.prog_df, self.channel_df)
            df.show()
            pipeline = self.job.featurize_dataset(df)

        model = pipeline.fit(df)
        rez = model.transform(df)
        cols = rez.columns
        self.assertIn("cross_AGExGENDER", cols)
        self.assertIn("cross_AFFINITYxDURATION_AFFINITY=Action", cols)
        self.assertIn("cross_EXTERNAL_RATINGxTOTAL_CELEB_POINTS", cols)

    def test_feature_explanation(self):
        now = self.job.now
        interaction_data = create_spark_df_from_data(
            self.job.spark,
            {"USER_ID": [0], "PROGRAM_ID": [0], "CHANNEL_ID": [0], "LABEL": [0], "RANK": [1],
             "DATE_DAY": [now]}
        )
        user_data = create_spark_df_from_data(
            self.job.spark,
            {"USER_ID": [0, 1],
             "GENDER": ["U", "F"],
             "AGE": [0, 1],
             "DURATION_AFFINITY=Action": [0, 1],
             "DURATION_ACTION_DEVICE_TYPE=tv": [0, 1],
             "DURATION_CATEGORY=1": [0, 1],
             "DURATION_PERSON_ID": ['{"23": 32452, "2": 1}', ""],
             "DATE_DAY": [now, now]}
        )
        prog_data = create_spark_df_from_data(
            self.job.spark,
            {"PROGRAM_ID": [0, 1],
             "PRODUCTION_YEAR": [0, 1999],
             "REF_PROGRAM_CATEGORY_ID": [1, 0],
             "REF_PROGRAM_KIND_ID": [1, 0],
             "TOTAL_CELEB_POINTS": [None, 1],
             "EXTERNAL_RATING": [None, 1],
             "PROGRAM_DURATION": [1200, 0],
             "AFFINITY": ["Action", "Action"],
             "DATE_DAY": [now, now],
             "FAMOUS_CAST": ['["33"]', '["3"]']
             }
        )
        channel_data = create_spark_df_from_data(
            self.job.spark,
            {"CHANNEL_ID": [0, 1],
             "channel_DURATION_socio=F_4": [0, 1],
             "channel_DURATION_socio=H_4": [0, 1],
             "channel_DURATION_AFFINITY=Action": [0, 1],
             "DATE_DAY": [now, now]}
        )

        # Feat the pipeline
        with mock.patch.multiple("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint",
                                 mlflow=mock.MagicMock(mlflow)):
            self.job.combine_learning_log(interaction_data, user_data, prog_data, channel_data)
        # Do the inference
        rez = self.job.combine_learning_log(interaction_data, user_data.limit(1), prog_data.limit(1),
                                            channel_data.limit(1),
                                            train=False)
        rez.collect()
        self.assertIn("FAMOUS_CAST_vectorized", rez.columns)

    def test_end_to_end(self):
        with mock.patch.multiple("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint",
                                 BinaryClassificationMetrics=mock.MagicMock(BinaryClassificationMetrics),
                                 load_snowflake_table=multiplex_mock,
                                 mlflow=mock.MagicMock(mlflow)):
            with mock.patch.multiple("databricks_jobs.db_common",
                                     load_snowflake_table=multiplex_mock):
                self.job.valid_lookback = datetime.timedelta(days=0)
                self.job.launch()


class TestPostTrainingPclick(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        It could be necessary to have a local mlflow and add the following to the setupClass
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("pclick-training")
        """
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        spark.conf.set("spark.default.parallelism", "1")
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        cls.job = PClickTrainingJob()
        cls.job.MAX_ITER = 1
        cls.job.model_directory = "models"

        cls.train_base_model()

    @classmethod
    def train_base_model(cls):
        if not os.path.exists(cls.job.model_directory):
            os.mkdir(cls.job.model_directory)
        now = cls.job.now
        user_data = {
            "USER_ID": [1, 2, 3],
            "GENDER": ["F", "M", "U"],
            "AGE": [12, 34, 4],
            "DURATION_AFFINITY=Action": [2340, 0, 0],
            "DURATION_ACTION_DEVICE_TYPE=tv": [234, 1200, 0],
            "DURATION_CATEGORY=1": [34, 1, 90],
            "DURATION_PERSON_ID": ['{"23": 32452, "2": 1}', None, ""],
            "DATE_DAY": [now, now, now]
        }
        prog_data = {
            "PROGRAM_ID": [1, 2, 0],
            "PRODUCTION_YEAR": [1990, 2008, 1970],
            "REF_PROGRAM_CATEGORY_ID": [1, 2, 3],
            "REF_PROGRAM_KIND_ID": [1, 2, 3],
            "TOTAL_CELEB_POINTS": [33, 0, None],
            "EXTERNAL_RATING": [3.1, 3.4, None],
            "PROGRAM_DURATION": [1200, 2340, 600],
            "AFFINITY": ["Action", "Cuisine", "Horreur"],
            "FAMOUS_CAST": ['["1"]', '["3"]', None],
            "DATE_DAY": [now, now, now],
        }
        channel_data = {
            "CHANNEL_ID": [0, 1, 2],
            "channel_DURATION_socio=F_4": [1, 0, 4],
            "channel_DURATION_socio=H_4": [1, 0, 4],
            "channel_DURATION_AFFINITY=Action": [20, 0, 33],
            "DATE_DAY": [now, now, now]
        }
        cls.user_df = create_spark_df_from_data(cls.job.spark, user_data). \
            withColumn("WATCHED_PERSON_ID_1H+", threshold_parse("DURATION_PERSON_ID", F.lit(3600))). \
            withColumn("WATCHED_PERSON_ID_10H+", threshold_parse("DURATION_PERSON_ID", F.lit(36000)))
        cls.prog_df = create_spark_df_from_data(cls.job.spark, prog_data)
        cls.channel_df = create_spark_df_from_data(cls.job.spark, channel_data)

        now = cls.job.now
        data = {"USER_ID": [1, 2, 3], "PROGRAM_ID": [0, 0, 0], "CHANNEL_ID": [0, 0, 0],
                "label": [0, 0, 1], "RANK": [1, 1, 1], "DATE_DAY": [now, now, now]}
        association_index_df = create_spark_df_from_data(cls.job.spark, data)
        cls.learning_log_df = cls.job.combine_learning_log(association_index_df, cls.user_df, cls.prog_df,
                                                           cls.channel_df)
        pipeline = cls.job.featurize_dataset(cls.learning_log_df)
        cls.model = pipeline.fit(cls.learning_log_df)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.job.model_directory)

    def test_inference_on_model(self):

        uids = create_spark_df_from_data(self.job.spark, {"USER_ID": list(range(10))})
        now = datetime.date.today()

        with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.helpers",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                int_df = build_interaction_log_from_broadcast(self.job.spark, None, now - datetime.timedelta(days=1),
                                                              now + datetime.timedelta(days=14), uids,
                                                              min_duration_in_mins=0).\
                    withColumn("DATE_DAY", F.lit(self.job.now))
        # 2 - Featurize the data
        feature_df = self.job.combine_learning_log(int_df, self.user_df, self.prog_df, self.channel_df)
        feature_df.collect()

        # 3 - Load model and add prediction column
        prediction_per_user = self.model.transform(feature_df)
        prediction_per_user.select("USER_ID", "PROGRAM_ID", "CHANNEL_ID", "probability", "rawPrediction").collect()

    def test_inference_function(self):
        data = {
            "USER_ID": [0, 1, 2, 3, 4],
            "PROGRAM_ID": [0, 1, 2, 3, 4],
            "CHANNEL_ID": [0, 1, 2, 3, 4],
            "DATE_DAY": [self.job.now] * 5
        }
        df = create_spark_df_from_data(self.job.spark, data)
        with mock.patch.multiple("databricks_jobs.db_common",
                                 load_snowflake_table=multiplex_mock):
            with mock.patch.multiple("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint",
                                     load_snowflake_table=multiplex_mock,
                                     load_snowflake_query_df=lambda *args, **kwargs: df,
                                     write_df_to_snowflake=lambda *args, **kwargs: None,
                                     mlflow=mock.MagicMock(mlflow)):
                self.job.mlflow_registration(self.model, "pclick_model")
                self.job.inference()

    def test_mlflow_registration(self):
        mlflow.spark.log_model(self.model, "my_pclick")
        mlflow.spark.save_model(self.model, "my_pclick")

        model = mlflow.spark.load_model("my_pclick")
        assert model is not None

    def test_metric_additions(self):
        # Post training stats
        output = self.model.transform(self.learning_log_df)

        log_reg_model = self.model.stages[-1]
        self.job.log_training_stats(log_reg_model)
        self.job.log_feature_indexing(log_reg_model, output)


class TestTrainingUtils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.job = SampleJob()
        cls.model_path = "./model"
        os.mkdir(cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.model_path)

    def test_duration_person_threshold_indexer(self):
        df = create_spark_df_from_data(self.job.spark, {"watch_per_id": ['{"23": 32452, "2": 1}', '', None]})
        rez = df.select(threshold_parse("watch_per_id", F.lit(2)).alias("watch_more_than_1")).collect()
        self.assertEqual(rez[0].watch_more_than_1, ["23"])

    def test_matching_set_transformer(self):
        df = create_spark_df_from_data(self.job.spark, {"watch_per_id": [["23", "32"], [""], None],
                                                        "FAMOUS_CAST": [['23', '32'], ['23', '32'], ['23', '32']]})
        for indicator, rez in zip([False, True], [2, 1]):
            feat = MatchingSetFeature("watch_per_id", "FAMOUS_CAST", indicator)
            transformer = feat.build_interaction()
            df_2 = transformer.transform(df)
            self.assertIn(feat.name, df_2.columns)
            elements = df_2.select(df_2[feat.name]).collect()
            self.assertEqual(max([e[0] for e in elements]), rez)

    def test_inference_only(self):
        df = create_spark_df_from_data(self.job.spark, {"feature_index": [0, 1, 2], "feature_str": ["0", "1", "2"]})

        q = QuantileDiscretizer(numBuckets=2)
        q.setInputCol("feature_index")
        q.setOutputCol("feature_index_q")
        indexer_encoder = StringIndexer()
        indexer_encoder.setInputCols(["feature_str"])
        indexer_encoder.setOutputCols(["feat_index"])
        one_hot_encoder = OneHotEncoder(dropLast=False)
        one_hot_encoder.setInputCols(["feat_index"])
        one_hot_encoder.setOutputCols(["feature"])
        pipeline = Pipeline(stages=[q, indexer_encoder, one_hot_encoder])

        # Training + inference
        tf_df, model = load_or_train_and_transform(pipeline, df, self.model_path + "/model")
        tf_df.collect()
        self.assertIn("feature", tf_df.columns)
        self.assertIn("feature_index_q", tf_df.columns)

        # Inference only
        tf_df, model = load_or_train_and_transform(pipeline, df, self.model_path + "/model", train=False)
        tf_df.collect()

    def test_build_interaction_log_from_broadcast(self):
        uids = create_spark_df_from_data(self.job.spark, {"USER_ID": list(range(10))})
        now = datetime.date.today()

        with mock.patch("databricks_jobs.db_common.load_snowflake_table",
                        new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
            with mock.patch("databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.helpers",
                            new=lambda spark, options, table_name: multiplex_mock(spark, options, table_name)):
                rez = build_interaction_log_from_broadcast(self.job.spark, None, now - datetime.timedelta(days=1),
                                                           now + datetime.timedelta(days=14), uids,
                                                           min_duration_in_mins=0)
                rows = rez.collect()
                # crossJoin creates n_users * n_progs
                self.assertEqual(len(rows), 10 * 5)
