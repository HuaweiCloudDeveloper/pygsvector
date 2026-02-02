"""A extension for SQLAlchemy for vector storage related schema definition.

* ARRAY             An extended data type in SQLAlchemy for GsVecClient
* FLOATVECTOR            An extended data type in SQLAlchemy for GsVecClient
* VectorIndex       An extended index type in SQLAlchemy for GsVecClient
* CreateVectorIndex Vector Index Creation statement clause
* l2_distance       New system function to calculate l2 distance between vectors
* cosine_distance   New system function to calculate cosine distance between vectors
* hamming_bool_distance     New system function to calculate inner distance between vectors

* ReplaceStmt       Replace into statement based on the extension of SQLAlchemy.Insert
* BM25Index          Full Text Search Index
* CreateBM25Index    Full Text Search Index Creation statement clause
* MatchAgainst      Full Text Search clause
"""
from .floatvector import FLOATVECTOR
from .vector_index import VectorIndex, CreateVectorIndex
from .vec_dist_func import l2_distance, cosine_distance, hamming_bool_distance
from .replace_stmt import ReplaceStmt
from .dialect import GaussDBDialect, AsyncGaussDBDialect
from .bm25_index import BM25Index, CreateBM25Index

__all__ = [
    "FLOATVECTOR",
    "VectorIndex",
    "CreateVectorIndex",
    "l2_distance",
    "cosine_distance",
    "hamming_bool_distance",
    "ReplaceStmt",
    "GaussDBDialect",
    "AsyncGaussDBDialect",
    "BM25Index",
    "CreateBM25Index",
]
