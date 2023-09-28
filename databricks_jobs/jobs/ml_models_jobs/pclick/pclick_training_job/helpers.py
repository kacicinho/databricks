import json
from typing import NamedTuple, Tuple

import pyspark.sql.types as T
from pyspark.ml.feature import Interaction
from pyspark.ml.feature import OneHotEncoder, QuantileDiscretizer, StringIndexer, CountVectorizer, Bucketizer, \
    SQLTransformer
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql import functions as F

from databricks_jobs.jobs.utils.spark_utils import typed_udf
from databricks_jobs.db_common import build_episode_df, build_broadcast_df_with_episode_info


def build_interaction_log_from_broadcast(spark, options, start, end, uids, min_duration_in_mins=15,
                                         tvbundles=(25, 90, 26, 60, 31, 142, 146)):
    episode_info = build_episode_df(spark, options)
    epg = build_broadcast_df_with_episode_info(spark, options, episode_info, start, end,
                                               min_duration_in_mins=min_duration_in_mins). \
        where(F.col("TVBUNDLE_ID").isin(*tvbundles)). \
        select("PROGRAM_ID", "CHANNEL_ID"). \
        distinct()
    return epg. \
        crossJoin(F.broadcast(uids)).\
        select("USER_ID", "PROGRAM_ID", "CHANNEL_ID", F.lit(1).alias("RANK"), F.lit(0).alias("LABEL"))


def load_or_train_and_transform(defined_pipeline, df, save_path, train=True):
    if train:
        model = defined_pipeline.fit(df)
        model.write().overwrite().save(save_path)
        return model.transform(df), model
    else:
        model = PipelineModel.load(save_path)
    return model.transform(df), model


def build_pipeline(bucketized_features, continuous_features, string_features, index_features, bow_features):
    all_features = {}
    list(map(lambda x: all_features.update(x),
             [bucketized_features, continuous_features, string_features, index_features, bow_features]))
    features_to_vectorize = {}
    list(map(lambda x: features_to_vectorize.update(x),
             [bucketized_features, continuous_features, string_features, index_features]))

    # 1 - Raw feature are transformed into indexes
    stages = list()
    for feature in bucketized_features.values():
        qd = Bucketizer(splits=feature.buckets)
        qd.setInputCol(feature.col_name)
        qd.setOutputCol(feature.indexed_column)
        qd.setHandleInvalid("keep")
        stages.append(qd)

    for feature in continuous_features.values():
        qd = QuantileDiscretizer(numBuckets=5)
        qd.setInputCol(feature.col_name)
        qd.setOutputCol(feature.indexed_column)
        qd.setHandleInvalid("keep")
        stages.append(qd)

    if len(string_features) > 0:
        indexer = StringIndexer()
        indexer.setInputCols([feature.col_name for feature in string_features.values()])
        indexer.setOutputCols([feature.indexed_column for feature in string_features.values()])
        indexer.setHandleInvalid("skip")
        stages.append(indexer)

    # 2 - One hot encoding of all indexed features
    if len(features_to_vectorize) > 0:
        one_hot_encoder = OneHotEncoder(dropLast=False)
        one_hot_encoder.setInputCols([feature.indexed_column for feature in features_to_vectorize.values()])
        one_hot_encoder.setOutputCols([feature.vectorized_column for feature in features_to_vectorize.values()])
        one_hot_encoder.setHandleInvalid("keep")
        stages.append(one_hot_encoder)

    if len(bow_features) > 0:
        for col in bow_features.values():
            cv = CountVectorizer(inputCol=col.col_name, outputCol=col.vectorized_column, binary=True)
            # MinTF = n = we need a least n occurrence PER document, inference only
            cv.setMinTF(0)
            # MinDF, same but for training
            cv.setMinDF(0)
            stages.append(cv)

    return Pipeline(stages=stages)


class BucketizedFeature(NamedTuple):
    col_name: str
    buckets: Tuple

    @property
    def indexed_column(self):
        return self.col_name + "_indexed"

    @property
    def vectorized_column(self):
        return self.col_name + "_vectorized"


class RawFeature(NamedTuple):
    col_name: str

    @property
    def indexed_column(self):
        return self.col_name + "_indexed"

    @property
    def vectorized_column(self):
        return self.col_name + "_vectorized"


class IndexedFeature(NamedTuple):
    col_name: str

    @property
    def indexed_column(self):
        return self.col_name

    @property
    def vectorized_column(self):
        return self.col_name + "_vectorized"


class IdentityFeature(NamedTuple):
    col_name: str

    @property
    def indexed_column(self):
        return self.col_name

    @property
    def vectorized_column(self):
        return self.col_name


class CrossFeature(NamedTuple):
    col_a: str
    col_b: str

    @property
    def name(self):
        return f"cross_{self.col_a}x{self.col_b}"

    def build_interaction(self, all_features):
        interaction = Interaction()
        interaction.setInputCols(
            [all_features[self.col_a].vectorized_column, all_features[self.col_b].vectorized_column])
        interaction.setOutputCol(self.name)
        return interaction

    @property
    def mlflow_name(self):
        return f"{self.col_a}x{self.col_b}"


class MatchingSetFeature(NamedTuple):
    """
    Count the intersection between two sets.
    """
    set_a: str
    set_b: str
    indicator: bool = False

    @property
    def name(self):
        return f"matching_set_{self.set_a}x{self.set_b}_i={int(self.indicator)}"

    def build_interaction(self):
        if not self.indicator:
            max_value = 5
        else:
            max_value = 1
        return SQLTransformer(
            statement=f'SELECT *, least({max_value}, greatest(0, size(array_intersect(`{self.set_a}`, `{self.set_b}`)))) as `{self.name}` FROM __THIS__;'
        )

    @property
    def mlflow_name(self):
        return f"{self.set_a}x{self.set_b}_i={int(self.indicator)}"


@typed_udf(T.ArrayType(T.StringType()))
def threshold_parse(x, t):
    if x:
        return [k for k, v in json.loads(x).items() if v > t]
    else:
        return []
