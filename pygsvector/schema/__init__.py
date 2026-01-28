"""A extension for SQLAlchemy for vector storage related schema definition.

* ARRAY             An extended data type in SQLAlchemy for GsVecClient
* FLOATVECTOR            An extended data type in SQLAlchemy for GsVecClient
* SPARSE_VECTOR     An extended data type in SQLAlchemy for GsVecClient
* VectorIndex       An extended index type in SQLAlchemy for GsVecClient
* CreateVectorIndex Vector Index Creation statement clause
* GsDBTable           Extension to Table for creating table with vector index
* l2_distance       New system function to calculate l2 distance between vectors
* cosine_distance   New system function to calculate cosine distance between vectors
* inner_product     New system function to calculate inner distance between vectors

* ReplaceStmt       Replace into statement based on the extension of SQLAlchemy.Insert
* BM25Index          Full Text Search Index
* CreateBM25Index    Full Text Search Index Creation statement clause
* MatchAgainst      Full Text Search clause
"""
from .floatvector import FLOATVECTOR
from .sparse_vector import SPARSE_VECTOR
from .vector_index import VectorIndex, CreateVectorIndex
from .gs_table import GsDBTable
from .vec_dist_func import l2_distance, cosine_distance, inner_product, negative_inner_product
from .replace_stmt import ReplaceStmt
from .dialect import GaussDBDialect, AsyncGaussDBDialect
from .bm25_index import BM25Index, CreateBM25Index
from .match_against_func import MatchAgainst

__all__ = [
    "FLOATVECTOR",
    "SPARSE_VECTOR",
    "VectorIndex",
    "CreateVectorIndex",
    "GsDBTable",
    "l2_distance",
    "cosine_distance",
    "inner_product",
    "negative_inner_product",
    "ReplaceStmt",
    "GaussDBDialect",
    "AsyncGaussDBDialect",
    "BM25Index",
    "CreateBM25Index",
    "MatchAgainst",
]
