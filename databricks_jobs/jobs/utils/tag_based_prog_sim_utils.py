import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from collections import Counter
import itertools

from pyspark.sql.types import FloatType, IntegerType, StructField, StructType, ArrayType, StringType


tag_program_schema = StructType([
    StructField("CHANNEL_GROUP_ID", IntegerType(), nullable=False),
    StructField("CHANNEL_ID", IntegerType(), nullable=False),
    StructField("PROGRAM_ID", IntegerType(), nullable=False),
    StructField("KIND", StringType(), nullable=False),
    StructField("TAGS", ArrayType(IntegerType()), False)
])

reco_latest_schema = StructType([
    StructField("PROGRAM_ID", IntegerType(), nullable=False),
    StructField("CHANNEL_ID", IntegerType(), nullable=False),
    StructField("CHANNEL_GROUP_ID", IntegerType(), nullable=False),
    StructField("RECOMMENDATIONS", ArrayType(StructType([
        StructField("CHANNEL_ID", IntegerType(), False),
        StructField("PROGRAM_ID", IntegerType(), False),
        StructField("distance", FloatType(), False)
    ])), nullable=False)
])

sim_matrix_schema = StructType([
    StructField('CHANNEL_ID', IntegerType(), nullable=False),
    StructField('PROGRAM_ID', IntegerType(), nullable=False),
    StructField('RECOMMENDED_CHANNEL_ID', IntegerType(), nullable=False),
    StructField('RECOMMENDED_PROGRAM_ID', IntegerType(), nullable=False),
    StructField('score_t', FloatType(), nullable=False),
    StructField('score_k', IntegerType(), nullable=False),
    StructField('score_n', IntegerType(), nullable=False),
    StructField('rank', IntegerType(), nullable=False)
])


def extract_tags(train, x, vectorizer, knn):
    """
    :param train: dataframe with progs with tags
    :param x: a program with no tag
    :param vectorizer: tfidf or count vectorizer
    :return: List[str]
    """
    # 1 - Perform inference
    no_tag_prog = x['info_concat']
    no_tag_prog_vect = vectorizer.transform([no_tag_prog])
    res = knn.kneighbors(no_tag_prog_vect.reshape(1, -1), return_distance=False)

    # 2 - Map from programs to set of tags
    # 2.1 - each prog gives a list of tags
    nearest_prog = train.iloc[list(res[0])]
    all_tags = list(itertools.chain(*nearest_prog['TAGS']))
    # Keep the top 15 most common tags
    c = Counter(all_tags)
    return list(dict(c.most_common(15)).keys())


def similar_prog(train, x, vectorizer, knn):
    """
    Knn kneighors with return_distance=True returns a list of 2 list: 
    - the first one with the scores
    - the second one with the index
    """
    no_tag_prog = x['info_concat']
    no_tag_prog_vect = vectorizer.transform([no_tag_prog])
    data = knn.kneighbors(no_tag_prog_vect.reshape(1, -1), return_distance=True)
    res = pd.DataFrame({"distance": data[0][0]}, index=data[1][0])
    nearest_prog = pd.merge(res, train, left_index=True, right_index=True)
    nearest_prog[['PROGRAM_ID', 'CHANNEL_ID']] = nearest_prog[['PROGRAM_ID', 'CHANNEL_ID']].astype('int64')

    rez = nearest_prog[["CHANNEL_ID", "PROGRAM_ID", "distance"]].to_dict(orient='records')

    return list(
        map(lambda z: (z["CHANNEL_ID"], z["PROGRAM_ID"], z["distance"]),
            filter(lambda z: z["distance"] > 1e-4, rez))
    )


def fit_prog_sim_model(train_data, stop_words, min_n_neighbours):

    all_infos = list(train_data['info_concat'].values)
    size = len(train_data.index)
    max_df = 1 if size < 100 else 0.5
    tfidf = TfidfVectorizer(max_df=max_df, ngram_range=(1, 1), max_features=32768,
                            min_df=1, stop_words=stop_words,
                            use_idf=True)
    train_data_vectorized = tfidf.fit_transform(all_infos)

    n_neighbors = min(min_n_neighbours, size)
    knn = NearestNeighbors(metric='euclidean', n_neighbors=n_neighbors, p=2)
    knn.fit(train_data_vectorized)

    return knn, tfidf


def build_sim_matrix_on_tag(spark, program_no_tags_df, train_data, stop_words, min_n_neighbours):

    knn, tfidf = fit_prog_sim_model(train_data, stop_words, min_n_neighbours)
    program_no_tags_df['TAGS'] = program_no_tags_df.apply(
        lambda x: x['TAGS'] if x['N_TAGS'] > 0 else extract_tags(train_data, x, tfidf, knn), axis=1)
    program_no_tags_df[['PROGRAM_ID', 'CHANNEL_ID', 'CHANNEL_GROUP_ID']] = program_no_tags_df[['PROGRAM_ID', 'CHANNEL_ID', 'CHANNEL_GROUP_ID']].astype('int64')
    program_no_tags_df = program_no_tags_df[["CHANNEL_GROUP_ID", "CHANNEL_ID", "PROGRAM_ID", "KIND", "TAGS"]]

    return spark.createDataFrame(program_no_tags_df, schema=tag_program_schema)


def build_sim_matrix_on_descritpion(spark, program_no_tags_to_reco, train_data, stop_words, min_n_neighbours):

    knn, tfidf = fit_prog_sim_model(train_data, stop_words, min_n_neighbours)

    if len(program_no_tags_to_reco) > 0:
        program_no_tags_to_reco['RECOMMENDATIONS'] = program_no_tags_to_reco.apply(
            lambda x: similar_prog(train_data, x, tfidf, knn), axis=1)
    else:
        program_no_tags_to_reco['RECOMMENDATIONS'] = None

    program_no_tags_to_reco[['PROGRAM_ID', 'CHANNEL_ID', 'CHANNEL_GROUP_ID']] = program_no_tags_to_reco[['PROGRAM_ID', 'CHANNEL_ID', 'CHANNEL_GROUP_ID']].astype('int64')
    program_no_tags_to_reco = program_no_tags_to_reco[["PROGRAM_ID", "CHANNEL_ID", "CHANNEL_GROUP_ID", "RECOMMENDATIONS"]]

    res = spark.createDataFrame(program_no_tags_to_reco, schema=reco_latest_schema)
    res = res. \
        select("CHANNEL_ID", "PROGRAM_ID", F.explode("RECOMMENDATIONS").alias("RECOMMENDATIONS")). \
        withColumn("RECOMMENDED_CHANNEL_ID", F.col("RECOMMENDATIONS.CHANNEL_ID")). \
        withColumn("RECOMMENDED_PROGRAM_ID", F.col("RECOMMENDATIONS.PROGRAM_ID")). \
        withColumn("score_t", F.lit(1) / F.col("RECOMMENDATIONS.distance")). \
        withColumn("score_k", F.lit(0)). \
        withColumn("score_N", F.lit(0)). \
        withColumn('rank', F.row_number().over(Window.partitionBy("PROGRAM_ID").orderBy("score_t"))). \
        drop("RECOMMENDATIONS")

    return res
