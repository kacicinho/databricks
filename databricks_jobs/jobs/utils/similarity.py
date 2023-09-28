from faiss.swigfaiss import IndexIDMap as faissIndex
import pandas as pd
import pickle
import numpy as np
from numpy.linalg import norm
import faiss


def get_embeddings_array(df: pd.DataFrame) -> np.array:
    """
    takes EMBEDDING_BYTES column of df, and :
    * deserializes vectors
    * converts float64 to float32
    * normalizes
    * returns embeddings as array
    """
    df['VECTOR'] = df['EMBEDDING_BYTES'].apply(lambda x: pickle.loads(x).astype('float32'))
    df['NORMALIZED_VECTOR'] = df['VECTOR'].apply(lambda x: (x / norm(x)))
    array = np.stack(df['NORMALIZED_VECTOR'].values)
    return array


def compute_index(needs_comparison_df: pd.DataFrame) -> faissIndex:
    """
    * extracts embeddings of needs_comparison_df
    * uses ID column to build index map
    * returns faiss search index
    """
    needs_comparison_embeddings_array = get_embeddings_array(needs_comparison_df)
    # build index with IDs map
    d = needs_comparison_embeddings_array.shape[1]  # d=768 with transformers model
    ids = needs_comparison_df['ID'].values.astype(np.int64)  # format required by faiss algo
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))  # IP stands for Inner Product
    index.add_with_ids(needs_comparison_embeddings_array, ids)

    return index


def get_n_nearest_embeddings(search_index: faissIndex, to_compare_to_df: pd.DataFrame, n_nearest: int = 100) \
        -> pd.DataFrame:
    """
    for each embedding of to_compare_to_df, gets n_nearest vectors of search_index
    column returned are : id, n_rearest ids as list of int, n_scores as list of float
    """
    to_compare_to_embeddings_array = get_embeddings_array(to_compare_to_df)
    similarities = search_index.search(to_compare_to_embeddings_array, n_nearest)
    ids = to_compare_to_df['ID'].values

    # format data
    scores = similarities[0]
    reco_ids = similarities[1]

    res = pd.DataFrame(
        {'PROGRAM_ID': np.repeat(ids, n_nearest),
         'SIMILAR_PROGRAM_ID': reco_ids.flatten(),
         'SIMILARITY_SCORE': scores.flatten()}
    )
    return res
