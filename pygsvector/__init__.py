"""A python SDK for GaussDB Vector Store, based on SQLAlchemy, compatible with Milvus API.

`pygsvector` supports two modes:
1. `Milvus compatible mode`: You can use the `MilvusLikeClient` class to use vector storage 
in a way similar to the Milvus API.
2. `SQLAlchemy hybrid mode`: You can use the vector storage function provided by the 
`GsVecClient` class and execute the relational database statement with the SQLAlchemy library.
In this mode, you can regard `pygsvector` as an extension of SQLAlchemy.

* GsVecClient           MySQL client in SQLAlchemy hybrid mode
* MilvusLikeClient      Milvus compatible client
* VecIndexType          VecIndexType is used to specify vector index type for MilvusLikeClient
* IndexParam            Specify vector index parameters for MilvusLikeClient
* IndexParams           A list of IndexParam to create vector index in batch
* DataType              Specify field type in collection schema for MilvusLikeClient
* FLOATVECTOR                An extended data type in SQLAlchemy for GsVecClient
* SPARSE_VECTOR         An extended data type in SQLAlchemy for GsVecClient
* VectorIndex           An extended index type in SQLAlchemy for GsVecClient
* BM25Index              Full Text Search Index
* FieldSchema           Clas to define field schema in collection for MilvusLikeClient
* CollectionSchema      Class to define collection schema for MilvusLikeClient
* PartType              Specify partition type of table or collection 
                        for both GsVecClient and MilvusLikeClient
* GsPartition           Abstract type class of all kind of Partition strategy
* RangeListPartInfo     Specify Range/RangeColumns/List/ListColumns partition info
                        for each partition
* GsRangePartition      Specify Range/RangeColumns partition info
* GsListPartition       Specify List partition info
* GsHashPartition       Specify Hash partition info
* GsKeyPartition        Specify Key partition info
* BM25IndexParam         Full Text Search index parameter
* MatchAgainst          Full Text Search clause
"""
from .client import *
from .schema import (
    FLOATVECTOR,
    SPARSE_VECTOR,
    VectorIndex,
    l2_distance,
    cosine_distance,
    inner_product,
    negative_inner_product,
    BM25Index,
    MatchAgainst,
    GaussDBDialect,
    AsyncGaussDBDialect,
)

__all__ = [
    "GsVecClient",
    "MilvusLikeClient",
    "VecIndexType",
    "IndexParam",
    "IndexParams",
    "DataType",
    "FLOATVECTOR",
    "SPARSE_VECTOR",
    "VectorIndex",
    "BM25Index",
    "FieldSchema",
    "CollectionSchema",
    "PartType",
    "GsPartition",
    "RangeListPartInfo",
    "GsRangePartition",
    "GsSubRangePartition",
    "GsListPartition",
    "GsSubListPartition",
    "GsHashPartition",
    "GsSubHashPartition",
    "GsKeyPartition",
    "GsSubKeyPartition",
    "l2_distance",
    "cosine_distance",
    "inner_product",
    "negative_inner_product",
    "BM25IndexParam",
    "MatchAgainst",
    "GaussDBDialect",
    "AsyncGaussDBDialect",
]
