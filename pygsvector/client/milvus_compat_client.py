"""Milvus Like Client."""
import logging
import json
from typing import Optional, Union, Dict, List

from sqlalchemy.exc import NoSuchTableError
from sqlalchemy import (
    Column,
    Integer,
    String,
    text,
    Table,
    select,
)
from sqlalchemy.sql import func
import numpy as np

from .gs_vec_client import GsVecClient as Client
from .schema_type import DataType
from .collection_schema import CollectionSchema
from .index_param import IndexParams
from .exceptions import *
from ..schema import FLOATVECTOR, VectorIndex
from ..util import Vector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _parse_metric_type_str_to_dist_func(metric_type: str):
    if metric_type == "l2":
        return func.l2_distance
    if metric_type == "cosine":
        return func.cosine_distance
    if metric_type == "hamming":
        return func.hamming_bool_distance
    raise VectorMetricTypeException(
        code=ErrorCode.INVALID_ARGUMENT,
        message=ExceptionsMessage.MetricTypeValueInvalid,
    )


class MilvusCompatClient(Client):
    """Milvus Compat Vector Database Client"""

    def __init__(
        self,
        uri: str = "127.0.0.1:8000",
        user: str = "usr",
        password: str = " ",
        db_name: str = "gaussdb",
        **kwargs,
    ):
        super().__init__(uri, user, password, db_name, **kwargs)

    # Collection & Schema API

    def create_schema(self, **kwargs) -> CollectionSchema:
        """Create a CollectionSchema object."""
        return CollectionSchema(**kwargs)

    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        primary_field_name: str = "id",
        id_type: Union[DataType, str] = DataType.INT64,
        vector_field_name: str = "vector",
        metric_type: str = "l2",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        schema: Optional[CollectionSchema] = None,  # Used for custom setup
        index_params: Optional[IndexParams] = None,  # Used for custom setup
        max_length: int = 16384,
        **kwargs,
    ): # pylint: disable=unused-argument
        """Create a collection. 
        If `schema` is not None, `dimension`, `primary_field_name`, `id_type`, `vector_field_name`,
        `metric_type`, `auto_id` will be ignored.
        
        Args:
            collection_name (string) : collection name
            dimension (Optional[int]) : vector data dimension
            primary_field_name (string) : primary field name
            id_type (Union[DataType, str]) :
                primary field data type(Only VARCHAR and INT type supported)
            vector_field_name (string) : vector field name
            metric_type (str) : l2 or cosine (for default, l2 distance)
            auto_id (bool) : whether primary field is auto incremented
            timeout (Optional[float]) : not used in GaussDB
            schema (Optional[CollectionSchema]) :
                customed collection schema, when `schema` is not None 
                the above argument will be ignored
            index_params (Optional[IndexParams]) : customed vector index parameters
            max_length (int) :
                when primary field data type is VARCHAR and `schema` is not None 
            the max varchar length is `max_length`
        """
        if isinstance(id_type, str):
            if id_type not in ("str", "string", "int", "integer"):
                raise PrimaryKeyException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.PrimaryFieldType,
                )
            if id_type in ("str", "string"):
                id_type = DataType.VARCHAR
            else:
                id_type = DataType.INT64

        if id_type not in (
            DataType.VARCHAR,
            DataType.INT64,
            DataType.INT32,
            DataType.INT16,
            DataType.INT8,
            DataType.BOOL,
        ):
            raise PrimaryKeyException(
                code=ErrorCode.INVALID_ARGUMENT,
                message=ExceptionsMessage.PrimaryFieldType,
            )

        if schema is None:
            if dimension is None:
                raise VectorFieldParamException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.VectorFieldMissingDimParam,
                )
            id_column = (
                Column(
                    primary_field_name,
                    Integer(),
                    primary_key=True,
                    autoincrement=auto_id,
                )
                if id_type == DataType.INT64
                else Column(
                    primary_field_name,
                    String(max_length),
                    primary_key=True,
                    autoincrement=auto_id,
                )
            )
            vector_column = Column(vector_field_name, FLOATVECTOR(dimension))
            columns = [id_column, vector_column]
            self.create_table_with_index_params(
                table_name=collection_name,
                columns=columns,
                index_params=index_params,
            )
        else:
            columns = [field.column_schema for field in schema.fields]
            self.create_table_with_index_params(
                table_name=collection_name,
                columns=columns,
                index_params=index_params,
                partitions=schema.partitions,
            )

    def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None # pylint: disable=unused-argument
    ) -> Dict:
        """Get collection row count.
        
        Args:
            collection_name (string): collection name
            timeout (Optional[float]): not used in GaussDB
        Returns:
            dict: {'row_count': count}
        """
        with self.engine.connect() as conn:
            with conn.begin():
                res = conn.execute(
                    text(f"SELECT COUNT(*) as row_count FROM {collection_name}")
                )
                cnt = [r[0] for r in res][0]
                return {"row_count": cnt}

    def has_collection(
        self, collection_name: str, timeout: Optional[float] = None # pylint: disable=unused-argument
    ) -> bool: # pylint: disable=unused-argument
        """Check if collection exists.

        Args:
            collection_name (string): collection name
            timeout (Optional[float]): not used in GaussDB
        Returns:
            bool: True if collection exists else False
        """
        return self.check_table_exists(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        """drop collection if exists.
        
        Args:
            collection_name (string) : collection name
        """
        self.drop_table_if_exist(collection_name)

    def rename_collection(
        self, old_name: str, new_name: str, timeout: Optional[float] = None # pylint: disable=unused-argument
    ) -> None:
        """rename collection.
        
        Args:
            old_name (string) : old collection name
            new_name (string) : new collection name
            timeout (Optional[float]): not used
        """
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(text(f"ALTER TABLE {old_name} RENAME TO {new_name}"))

    def load_collection(
        self,
        collection_name: str,
    ):
        """Load table into SQLAlchemy metadata.
        
        Args:
            collection_name (string): which collection to load
        Returns:
            sqlalchemy.Table: table object
        """
        try:
            table = Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e
        return table

    # Index API

    def create_index(
            self,
            collection_name: str,
            index_params: IndexParams,
            timeout: Optional[float] = None,
            **kwargs,
    ):  # pylint: disable=unused-argument
        """Create vector index with index params.

        Args:
            collection_name (string): which collection to create vector index
            index_params (IndexParams): the vector index parameters
            timeout (Optional[float]): not used in GaussDB
            **kwargs: different args for different vector index type
        """
        try:
            Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

        for index_param in index_params:
            super().create_index(
                table_name=collection_name,
                index_name=index_param.index_name,
                column_names=[index_param.field_name],
                index_type=index_param.index_type,
                metric_type=index_param.metric_type,
                idx_params=index_param.param_str(),
                **kwargs,
            )

    def drop_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ): # pylint: disable=unused-argument
        """Drop index on specified collection.
        
        If the index not exists, SQL ERROR will raise.

        Args:
            collection_name (string): which collection the index belongs to
            index_name (string): which index
            timeout (Optional[float]): not used in GaussDB
            **kwargs: additional arguments
        """
        super().drop_index(collection_name, index_name)

    def rebuild_index(
        self,
        collection_name: str,
        index_name: str,
        trigger_threshold: float = 0.2,
    ):
        """Rebuild vector index for performance.
        
        Args:
            collection_name (string) : collection name
            index_name (string) : vector index name
            trigger_threshold (float)
        """
        super().rebuild_index(
            table_name=collection_name,
            index_name=index_name,
        )

    # Insert & Search

    def _parse_value_for_text_sql(
        self, need_parse: bool, table, column_name: str, value
    ):
        if not need_parse:
            return value
        try:
            type_str = str(table.c[column_name].type)
            if type_str.startswith("floatvector") and value is not None:
                return Vector._from_db(value)
        except KeyError:
            return value
        return value

    def search(
        self,
        collection_name: str,
        data: Union[list, dict],
        anns_field: Optional[str] = "vector",
        with_dist: bool = False,
        filter=None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None, # pylint: disable=unused-argument
        partition_names: Optional[List[str]] = None,
        **kwargs, # pylint: disable=unused-argument
    ) -> List[dict]:
        """Perform ann search.
        Note: GaussDB does not support batch search now. `data` & the return value is not a batch.
        
        Args:
            collection_name (string): collection name
            data (list): the vector/sparse_vector data to search
            anns_field (string): which vector field to search
            with_dist (bool): return result with distance
            filter: do ann search with filter (note: parameter name is intentionally 'filter' to distinguish it from the built-in function)
            limit (int): top K
            output_fields (Optional[List[str]]): output fields
            search_params (Optional[dict]): Only `metric_type` with value `l2`/`neg_ip` supported
            timeout (Optional[float]): not used in GaussDB
            partition_names (Optional[List[str]]): limit the query to certain partitions
            **kwargs: additional arguments
        Returns:
            List[dict]: A list of records, each record is a dict indicating a mapping from
                column_name to column value.
        """
        if not isinstance(data, (list, dict)):
            raise ValueError("'data' type must be in 'list'/'dict'")

        lower_metric_type_str = "l2"
        if search_params and "metric_type" in search_params:
            mt = search_params["metric_type"]
            if not isinstance(mt, str):
                raise VectorMetricTypeException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.MetricTypeParamTypeInvalid,
                )
            lower_metric_type_str = mt.lower()
            if lower_metric_type_str not in ("l2", "cosine", "hamming"):
                raise VectorMetricTypeException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.MetricTypeValueInvalid,
                )
        distance_func = _parse_metric_type_str_to_dist_func(lower_metric_type_str)

        try:
            Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

        result_proxy = super().ann_search(
            table_name=collection_name,
            vec_data=data,
            vec_column_name=anns_field,
            distance_func=distance_func,
            with_dist=with_dist,
            topk=limit,
            output_column_names=output_fields,
            where_clause=filter,
            partition_names=partition_names,
        )

        data_res = result_proxy.fetchall()
        columns = list(result_proxy.keys())

        res = [
            {
                col_name: self._parse_value_for_text_sql(True,
                                                         Table(collection_name, self.metadata_obj,
                                                               autoload_with=self.engine),
                                                         col_name, value)
                for col_name, value in zip(columns, row)
            }
            for row in data_res
        ]

        if with_dist:
            res.sort(key=lambda x: x[columns[-1]])

        return res

    def query(
        self,
        collection_name: str,
        filter=None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None, # pylint: disable=unused-argument
        partition_names: Optional[List[str]] = None,
        **kwargs, # pylint: disable=unused-argument
    ) -> List[dict]:
        """query records.
        
        Args:
            collection_name (string): collection name
            filter: do ann search with filter (note: parameter name is intentionally 'filter' to distinguish it from the built-in function)
            output_fields (Optional[List[str]]): output fields
            timeout (Optional[float]): not used in GaussDB
            partition_names (Optional[List[str]]): limit the query to certain partitions
            **kwargs: additional arguments
        Returns:
            List[dict]: A list of records, each record is a dict indicating a mapping from
                column_name to column value.
        """
        try:
            table = Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

        if output_fields is not None:
            columns = [table.c[column_name] for column_name in output_fields]
            stmt = select(*columns)
        else:
            stmt = select(table)

        if filter is not None:
            stmt = stmt.where(*filter)

        with self.engine.connect() as conn:
            with conn.begin():
                if partition_names is None:
                    execute_res = conn.execute(stmt)
                else:
                    stmt_str = str(stmt.compile(
                        dialect=self.engine.dialect,
                        compile_kwargs={"literal_binds": True}
                    ))
                    stmt_str = self._insert_partition_hint_for_query_sql(
                        stmt_str, f"PARTITION({', '.join(partition_names)})"
                    )
                    logging.debug(stmt_str)
                    execute_res = conn.execute(text(stmt_str))
                data_res = execute_res.fetchall()
                columns = list(execute_res.keys())
                return [
                    {
                        columns[i]: self._parse_value_for_text_sql(
                            partition_names is not None, table, columns[i], value
                        )
                        for i, value in enumerate(row)
                    }
                    for row in data_res
                ]

    def get(
        self,
        collection_name: str,
        ids: Union[list, str, int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None, # pylint: disable=unused-argument
        partition_names: Optional[List[str]] = None,
        **kwargs, # pylint: disable=unused-argument
    ) -> List[dict]:
        """Get records with specified primary field `ids`.
        
        Args:
            collection_name (string): collection name
            ids (Union[list, str, int]): specified primary field values
            output_fields (Optional[List[str]]): output fields
            timeout (Optional[float]): not used in GaussDB
            partition_names (Optional[List[str]]): limit the query to certain partitions
            **kwargs: additional arguments
        Returns:
            List[dict]: A list of records, each record is a dict indicating a mapping from
                column_name to column value.
        """
        try:
            table = Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

        primary_keys = table.primary_key
        pkey_names = [column.name for column in primary_keys]
        if len(pkey_names) > 1:
            raise MilvusCompatibilityException(
                code=ErrorCode.INVALID_ARGUMENT,
                message=ExceptionsMessage.UsingInIDsWhenMultiPrimaryKey,
            )
        if len(pkey_names) == 0:
            raise ValueError(f"Table '{collection_name}' has no primary key.")

        result_proxy = super().get(
            table_name=collection_name,
            ids=ids,
            where_clause=None,
            output_column_names=output_fields,
            partition_names=partition_names,
            n_limits=None,
            idx_name_hint=None,
        )

        data_res = result_proxy.fetchall()
        columns = list(result_proxy.keys())

        return [
            {
                columns[i]: self._parse_value_for_text_sql(
                    partition_names is not None, table, columns[i], value
                )
                for i, value in enumerate(row)
            }
            for row in data_res
        ]

    def delete(
        self,
        collection_name: str,
        ids: Optional[Union[list, str, int]] = None,
        timeout: Optional[float] = None, # pylint: disable=unused-argument
        filter=None,
        partition_name: Optional[str] = "",
        **kwargs, # pylint: disable=unused-argument
    ) -> dict:
        """Delete data in collection.

        Args:
            collection_name (string): collection name
            ids (Optional[Union[list, str, int]]): a list of primary keys value
            timeout (Optional[float]): not used in GaussDB
            filter: delete with filter (note: parameter name is intentionally 'filter' to distinguish it from the built-in function)
            partition_name (Optional[str]): limit the query to certain partition
            **kwargs: additional arguments
        Returns:
            dict: deletion result
        """
        try:
            super().delete(
                table_name=collection_name, ids=ids, where_clause=filter, partition_name=partition_name
            )
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

    def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
    ) -> (
        None
    ):  # pylint: disable=unused-argument
        """Insert data into collection.
        
        Args:
            collection_name (string): collection name
            data (Union[Dict, List[Dict]]): data that will be inserted
            timeout (Optional[float]): not used in GaussDB
            partition_name (Optional[str]): limit the query to certain partition
        """
        try:
            super().insert(
                table_name=collection_name, data=data, partition_name=partition_name
            )
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

    def upsert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None, # pylint: disable=unused-argument
        partition_name: Optional[str] = "",
    ) -> List[Union[str, int]]:
        """Update data in table. If primary key is duplicated, replace it.
        
        Args:
            collection_name (string): collection name
            data (Union[Dict, List[Dict]]): data that will be upserted
            timeout (Optional[float]): not used in GaussDB
            partition_name (Optional[str]): limit the query to certain partition
        Returns:
            List[Union[str, int]]: list of primary keys
        """
        try:
            super().upsert(
                table_name=collection_name, data=data, partition_name=partition_name
            )
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists,
            ) from e

    def perform_raw_text_sql(self, text_sql: str):
        return super().perform_raw_text_sql(text_sql)
