"""Data type module that compatible with Milvus."""
from sqlalchemy import (
    Boolean,
    SmallInteger,
    Integer,
    BigInteger,
    String,
    Float,
    JSON,
    Double,
    ARRAY
)
from sqlalchemy.dialects.postgresql import TEXT

from .enum import IntEnum
from ..schema import FLOATVECTOR


class DataType(IntEnum):
    """Data type definition that compatible with Milvus."""
    # NONE = 0
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5

    FLOAT = 10
    DOUBLE = 11

    STRING = 20
    VARCHAR = 21
    ARRAY = 22
    JSON = 23

    FLOAT_VECTOR = 101
    BOOL_VECTOR = 102


def convert_datatype_to_sqltype(datatype: DataType):
    """Convert Milvus data type to SQL type.
    
    Args:
        datatype (DataType) : Milvus data type.
    """
    if datatype in (DataType.BOOL, DataType.INT8):
        return Boolean
    if datatype == DataType.INT16:
        return SmallInteger
    if datatype == DataType.INT32:
        return Integer
    if datatype == DataType.INT64:
        return BigInteger
    if datatype == DataType.FLOAT:
        return Float
    if datatype == DataType.DOUBLE:
        return Double
    if datatype == DataType.STRING:
        return TEXT
    if datatype == DataType.VARCHAR:
        return String
    if datatype == DataType.ARRAY:
        return ARRAY
    if datatype == DataType.JSON:
        return JSON
    if datatype == DataType.FLOAT_VECTOR:
        return FLOATVECTOR
    raise ValueError(f"Invalid DataType: {datatype}")
