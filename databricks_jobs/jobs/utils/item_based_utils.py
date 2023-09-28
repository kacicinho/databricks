from typing import List
from numpy.random import default_rng
import pyspark.sql.functions as f
from pyspark.sql import DataFrame as SparkDataFrame
from numpy.random._generator import Generator as RandGenerator


class JaccardHasher:
    """
   Jaccard Hasher aims at finding similarities between nodes (nodes can be programs) using edge info (edge can be user)
   The computed similarity is an approximation of Jaccard Similarity (edges being elements of node thught as ensembles)
   This approximation is made using the minHash technique :

   With a set of n hash functions, we compute the n hashes of each edge of each node.
   For each node, and each function, we keep the min value of this function across edges.
   The similarity of two nodes is the proportion of the n min values they have in common
   """

    def __init__(self, user_item_sdf: SparkDataFrame, node_col: str, edge_basis_col: str, n_hashes: int,
                 seed: int = 42, hash_prime: int = 2038074743):
        self.user_item_sdf = user_item_sdf
        self.node_col = node_col
        self.edge_basis_col = edge_basis_col
        self.n_hashes = n_hashes
        self.seed = seed
        self.hash_prime = hash_prime

    def compute_and_get_similarities(self) -> SparkDataFrame:
        rng = default_rng(self.seed)
        # hash functions a's and b's (coefs and shifts)
        coefs = hash_func_coeffs(size=self.n_hashes, max_value=self.hash_prime, rng=rng)
        shifts = hash_func_shifts(size=self.n_hashes, max_value=self.hash_prime, rng=rng)

        # hash computations
        hashed = compute_hashes(user_item_sdf=self.user_item_sdf,
                                coefs=coefs,
                                shifts=shifts,
                                hash_prime=self.hash_prime,
                                edge_basis_col=self.edge_basis_col)

        # min hash computation
        min_hashed = compute_minhash_per_node(sdf_with_hashes=hashed,
                                              node_col=self.node_col)

        # (node , hash_funcs) minHash collection
        collected_min_hashes = collect_minhashes(min_hashed, node_col=self.node_col)

        # all pairs sharing minHashes
        self_joined = self_join_collected(sdf=collected_min_hashes, node_col=self.node_col)

        # similarities
        similarities_sdf = hashed_jaccard_similarity(sdf=self_joined,
                                                     node_col=self.node_col,
                                                     n_hashes=self.n_hashes)
        return similarities_sdf


def hash_func_coeffs(size: int, max_value: int, rng: RandGenerator) -> List[int]:
    return (rng.choice(max_value, size=size, replace=False) + 1).tolist()


def hash_func_shifts(size: int, max_value: int, rng: RandGenerator) -> List[int]:
    return rng.choice(max_value, size=size, replace=False).tolist()


def compute_hashes(user_item_sdf, coefs: List[int], shifts: List[int],
                   hash_prime: int, edge_basis_col: str, hash_func_prefix: str = "hash_") -> SparkDataFrame:
    """
   (coefs and shifts needs to be lists to avoid 'numpy.int64' object has no attribute '_get_object_id' error)
   this function hashes each element of edge_basis_col with hashes function like h(x) = (a * hash(x) + b) % c
   the a are stores in coeffs
   the b are stored in shifts
   c is hash_prime
   there are n_hashes such functions
   """
    hash_calculations = [
        ((f.hash(f.col(edge_basis_col)) * a + b) % hash_prime)
        .alias(f'{hash_func_prefix}{n}')
        for n, (a, b) in enumerate(zip(coefs, shifts))
    ]
    return user_item_sdf.select('*', *hash_calculations)


def compute_minhash_per_node(sdf_with_hashes: SparkDataFrame, node_col: str, hash_func_prefix: str = "hash_",
                             min_hash_col: str = "minHash",
                             hash_index_col: str = "hashIndex") -> SparkDataFrame:
    """
   for each node:
    - finds the minHash (across every edge element of node, and every hash function)
    - and the corresponding hash function index
   """
    hash_cols = [c for c in sdf_with_hashes.columns if c.startswith(hash_func_prefix)]

    hash_sdf = (
        sdf_with_hashes
        .groupBy(node_col)
        .agg(f.array(*[f.min(col) for col in hash_cols]).alias(min_hash_col))
        .select(
            node_col,
            f.posexplode(f.col(min_hash_col)).alias(hash_index_col, min_hash_col)
        )
    )
    return hash_sdf


def collect_minhashes(sdf: SparkDataFrame, node_col: str, min_hash_col: str = "minHash",
                      hash_index_col: str = "hashIndex", node_set_col: str = "nodeSet") -> SparkDataFrame:
    """
   for each pair hash_index, min_hash_value, collects all nodes that have a matching entry in sdf
   """
    return (
        sdf
        .groupby(hash_index_col, min_hash_col)
        .agg(f.collect_set(f.col(node_col)).alias(node_set_col))
    )


def self_join_collected(sdf: SparkDataFrame, node_col, left_node_suffix="_I", right_node_suffix: str = "_J",
                        min_hash_col: str = "minHash", hash_index_col: str = "hashIndex",
                        node_set_col: str = "nodeSet"):
    """
   for each minHash present in sdf min_hash_col, compute all pairs of nodes that are both
   present in corresponding node_set_col
   """
    left_name = node_col + left_node_suffix
    right_name = node_col + right_node_suffix
    joined_sdf = (
        sdf.alias('a')
        .join(sdf.alias('b'), [hash_index_col, min_hash_col], 'inner')
        .select(
            f.col(min_hash_col),
            f.explode(f.col(f'a.{node_set_col}')).alias(left_name),
            f.col(f'b.{node_set_col}')
        )
    )
    exploded_sdf = (
        joined_sdf
        .select(
            f.col(min_hash_col),
            f.col(left_name),
            f.explode(f.col(node_set_col)).alias(right_name),
        )
    )
    return exploded_sdf


def hashed_jaccard_similarity(sdf: SparkDataFrame, node_col: str, n_hashes: int,
                              left_node_suffix: str = "_I", right_node_suffix: str = "_J",
                              similarity_col_name: str = "SIMILARITY_SCORE"):
    """
   computes approximate jaccard similarity for a pair of nodes as the number of minHash they share
   divided by the number of hash functions
   """
    return (
        sdf
        .groupby(node_col + left_node_suffix, node_col + right_node_suffix)
        .agg((f.count('*') / n_hashes).alias(similarity_col_name))
    )
