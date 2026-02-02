"""Multi-type Vector Store Client:

1. `Milvus compatible mode`: You can use the `MilvusCompatClient` class to use vector storage
in a way similar to the Milvus API.
2. `SQLAlchemy hybrid mode`: You can use the vector storage function provided by the 
`GsVecClient` class and execute the relational database statement with the SQLAlchemy library.
In this mode, you can regard `pygsvector` as an extension of SQLAlchemy.

* GsVecClient           MySQL client in SQLAlchemy hybrid mode
* MilvusCompatClient      Milvus compatible client
* IndexType          IndexType is used to specify vector index type for MilvusCompatClient
* VecIndexParam            Specify vector index parameters for MilvusCompatClient
* IndexParams           A list of VecIndexParam to create vector index in batch
* DataType              Specify field type in collection schema for MilvusCompatClient
* FieldSchema           Clas to define field schema in collection for MilvusCompatClient
* CollectionSchema      Class to define collection schema for MilvusCompatClient
* PartType              Specify partition type of table or collection 
                        for both GsVecClient and MilvusCompatClient
* GsPartition           Abstract type class of all kind of Partition strategy
* RangeListPartInfo     Specify Range/RangeColumns/List/ListColumns partition info
                        for each partition
* GsRangePartition      Specify Range/RangeColumns partition info
* GsListPartition       Specify List partition info
* GsHashPartition       Specify Hash partition info
* GsSubHashPartition    Specify Hash subpartition info
* GsKeyPartition        Specify Key partition info
* BM25IndexParam         Full Text Search index parameter
"""
from .gs_vec_client import GsVecClient
from .milvus_compat_client import MilvusCompatClient
from .async_gs_client import AsyncGsClient
from .async_gs_vec_client import AsyncGsVecClient
from .index_param import IndexType, VecIndexParam, BM25IndexParam, IndexParams
from .schema_type import DataType
from .collection_schema import FieldSchema, CollectionSchema
from .partitions import *

__all__ = [
    "GsVecClient",
    "AsyncGsVecClient",
    "MilvusCompatClient",
    "IndexType",
    "VecIndexParam",
    "IndexParams",
    "DataType",
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
    "BM25IndexParam",
]
