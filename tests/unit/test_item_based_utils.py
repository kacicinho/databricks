import os
from databricks_jobs.jobs.utils.item_based_utils import compute_minhash_per_node, collect_minhashes, \
    self_join_collected, \
    hashed_jaccard_similarity, compute_hashes
import pandas as pd
from pyspark.sql import SparkSession
import numpy as np
import pyspark.sql.functions as f

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
spark = SparkSession.builder.master("local[1]").getOrCreate()
spark.conf.set("spark.default.parallelism", "1")
spark.conf.set("spark.sql.shuffle.partitions", "1")
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def spark_hash(n: int):
    int_sdf = spark.createDataFrame(pd.DataFrame([{'n': n}]))
    hashes_sdf = int_sdf.withColumn('hash', f.hash(f.col('n')))
    return hashes_sdf.toPandas()['hash'].values[0]


def neg_modulo(p, q):
    # aims at reproducing spark implementation of mod, when use in selctexpr with symbol '%'
    # ie : p % q is negative when p is negative
    # ex : -3 % 7 = -4
    # ex : -10 % 7 = -4

    return p % q if p > 0 else p % (-q)


def mod_hash(n: int, a: int, b: int, c: int):
    return neg_modulo(a * spark_hash(n) + b, c)


def test_compute_hashes():
    hash_prime = 5
    n_hashes = 10
    coefs = [1] * n_hashes
    shifts = np.arange(n_hashes).tolist()
    num_edges = 6
    edges = range(-hash_prime, hash_prime + num_edges)
    data = pd.DataFrame({'edge': edges})

    def expected_hashes(df: pd.DataFrame) -> pd.DataFrame:
        # we want the hash columns to be of type int32 to mimic the result of spark conversion to pandas
        for i, (a, b) in enumerate(zip(coefs, shifts)):
            df[f"hash_{i}"] = df['edge'].apply(lambda n: mod_hash(n, a, b, hash_prime)).astype('int32')
        return df

    sdf = spark.createDataFrame(data)
    computed_hashes_pdf = compute_hashes(user_item_sdf=sdf, coefs=coefs, shifts=shifts, hash_prime=hash_prime,
                                         edge_basis_col='edge').toPandas()
    expected_hashes_pdf = expected_hashes(data)
    # print(computed_hashes_pdf)
    # print(expected_hashes_pdf)
    assert computed_hashes_pdf.equals(expected_hashes_pdf)
    return


def test_compute_minhash_per_node():
    hashed = spark.createDataFrame(pd.DataFrame([
        {'node': 10, 'edge': 1, 'hash_0': 0, 'hash_1': 1, 'hash_2': 2},
        {'node': 10, 'edge': 2, 'hash_0': -4, 'hash_1': 6, 'hash_2': 8},
        {'node': 20, 'edge': 1, 'hash_0': 0, 'hash_1': 1, 'hash_2': 2},
        {'node': 20, 'edge': 3, 'hash_0': -4, 'hash_1': 0, 'hash_2': 1},
    ]))

    computed_minhash_pdf = compute_minhash_per_node(sdf_with_hashes=hashed, node_col='node').toPandas()
    # following minHash standard designation "signature"
    computed_signature_pdf = computed_minhash_pdf.pivot(index='hashIndex', columns='node', values='minHash')
    expected_signature_pdf = pd.DataFrame(data=[{10: -4, 20: -4},
                                                {10: 1, 20: 0},
                                                {10: 2, 20: 1}
                                                ], index=[0, 1, 2])
    expected_signature_pdf.index.name = 'hashIndex'
    expected_signature_pdf.columns.name = 'node'
    # print(computed_signature_pdf)
    # print(expected_signature_pdf)
    assert computed_signature_pdf.equals(expected_signature_pdf)
    return


def test_collect_minhashes():
    min_hashed = spark.createDataFrame(pd.DataFrame([
        {'node': 10, 'hashIndex': 0, 'minHash': 0},
        {'node': 10, 'hashIndex': 1, 'minHash': -2},
        {'node': 20, 'hashIndex': 0, 'minHash': -6},
        {'node': 20, 'hashIndex': 1, 'minHash': -2},
    ]))

    computed_collected_pdf = collect_minhashes(min_hashed, node_col='node').toPandas()
    expected_collected_pdf = pd.DataFrame([
        {'hashIndex': 0, 'minHash': -6, 'nodeSet': [20]},
        {'hashIndex': 0, 'minHash': 0, 'nodeSet': [10]},
        {'hashIndex': 1, 'minHash': -2, 'nodeSet': [20, 10]}])
    # print(computed_collected_pdf)
    # print(expected_collected_pdf)
    # row order can be different, so sorting and dropping index is necessary bedore asserting equality
    equality = (
        expected_collected_pdf
        .sort_values('minHash')
        .reset_index(drop=True)
        .equals(
            computed_collected_pdf
            .sort_values('minHash')
            .reset_index(drop=True)
        )
    )
    assert equality


def test_self_join_collected():
    collected_sdf = spark.createDataFrame(pd.DataFrame([
        {'hashIndex': 0, 'minHash': -6, 'nodeSet': [20]},
        {'hashIndex': 0, 'minHash': 0, 'nodeSet': [10]},
        {'hashIndex': 1, 'minHash': -2, 'nodeSet': [20, 10]},
    ]))
    expected_self_joined_pdf = pd.DataFrame([
        {'minHash': -6, 'node_I': 20, 'node_J': 20},
        {'minHash': 0, 'node_I': 10, 'node_J': 10},
        {'minHash': -2, 'node_I': 20, 'node_J': 20},
        {'minHash': -2, 'node_I': 20, 'node_J': 10},
        {'minHash': -2, 'node_I': 10, 'node_J': 20},
        {'minHash': -2, 'node_I': 10, 'node_J': 10}
    ])
    # print(expected_self_joined_pdf)
    computed_self_joined_pdf = self_join_collected(sdf=collected_sdf, node_col='node').toPandas()
    # print(computed_self_joined_pdf)
    assert computed_self_joined_pdf.equals(expected_self_joined_pdf)


def test_hashed_jaccard_similarity():
    joined_sdf = spark.createDataFrame(pd.DataFrame([
        {'minHash': -6, 'node_I': 20, 'node_J': 20},
        {'minHash': 0, 'node_I': 10, 'node_J': 10},
        {'minHash': -2, 'node_I': 20, 'node_J': 20},
        {'minHash': -2, 'node_I': 20, 'node_J': 10},
        {'minHash': -2, 'node_I': 10, 'node_J': 20},
        {'minHash': -2, 'node_I': 10, 'node_J': 10}
    ]))

    computed_sim_pdf = hashed_jaccard_similarity(joined_sdf, node_col='node', n_hashes=2).toPandas()
    # print(computed_sim_pdf)

    expected_sim_pdf = pd.DataFrame([
        {'node_I': 10, 'node_J': 20, 'SIMILARITY_SCORE': 0.5},
        {'node_I': 10, 'node_J': 10, 'SIMILARITY_SCORE': 1.0},
        {'node_I': 20, 'node_J': 20, 'SIMILARITY_SCORE': 1.0},
        {'node_I': 20, 'node_J': 10, 'SIMILARITY_SCORE': 0.5}
    ])

    # print(expected_sim_pdf)
    # row order can be different, so sorting and dropping index is necessary bedore asserting equality
    equality = (
        computed_sim_pdf
        .sort_values(['node_I', 'node_J'])
        .reset_index(drop=True)
        .equals(
            expected_sim_pdf
            .sort_values(['node_I', 'node_J'])
            .reset_index(drop=True)
        )
    )
    assert equality
