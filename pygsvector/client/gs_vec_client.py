"""GaussDB Vector Store Client."""

import logging
from typing import List, Optional, Union

import numpy as np
from sqlalchemy import Table, Column, Index, select, text, ColumnElement

from .exceptions import ClusterVersionException, ErrorCode, ExceptionsMessage
from .gs_client import GsClient as Client
from .index_param import IndexParams, VecIndexParam, BM25IndexParam, IndexType
from .partitions import GsPartition
from ..schema import VectorIndex, BM25Index
from ..util import GsDBVersion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GsVecClient(Client):
    """The GaussDB Vector Client"""

    def __init__(
            self,
            uri: str = "127.0.0.1:8000",
            user: str = "usr",
            password: str = " ",
            db_name: str = "gaussdb",
            **kwargs,
    ):
        super().__init__(uri, user, password, db_name, **kwargs)

        if self.gs_version < GsDBVersion.from_db_version_nums(505, 1, 0):
            raise ClusterVersionException(
                code=ErrorCode.NOT_SUPPORTED,
                message=ExceptionsMessage.ClusterVersionIsLow % ("Vector Store", "505.1.0"),
            )

    def create_table_with_index_params(
            self,
            table_name: str,
            columns: List[Column],
            indexes: Optional[List[Index]] = None,
            index_params: Optional[IndexParams] = None,
            partitions: Optional[GsPartition] = None,
    ):
        """Create table with optional index_params.

        Args:
            table_name (string): table name
            columns (List[Column]): column schema
            indexes (Optional[List[Index]]): optional common index schema
            index_params (Optional[IndexParams]): optional vector and BM25 searchindex schema
            partitions (Optional[ObPartition]): optional partition strategy
        """
        with self.engine.connect() as conn:
            with conn.begin():
                # do partition
                partition_kwargs = {}
                local_index = False
                if partitions is not None:
                    partition_kwargs["postgresql_partition_by"] = partitions.do_compile()
                    local_index = True

                # create table with common index
                if indexes is not None:
                    table = Table(table_name, self.metadata_obj, *columns, *indexes, extend_existing=True,
                                  **partition_kwargs,
                                  )
                else:
                    table = Table(table_name, self.metadata_obj, *columns, extend_existing=True, **partition_kwargs,
                                  )

                table.create(self.engine, checkfirst=True)

                if index_params is not None:
                    for idx in index_params:
                        if isinstance(idx, VecIndexParam):
                            # create vector indexes
                            idx = VectorIndex(idx.index_name, table.c[idx.field_name],
                                              index_type=idx.index_type, metric_type=idx.metric_type,
                                              local_index=local_index, params=idx.param_str(),
                                              )
                        elif isinstance(idx, BM25IndexParam):
                            # create fts indexes
                            idx = BM25Index(idx.index_name, table.c[idx.field_name],
                                            local_index=local_index, params=idx.param_str(),
                                            )
                        idx.create(self.engine, checkfirst=True)

    def __is_partitioned_table(self, table_name: str, schema: str = "public") -> bool:
        """
        Check if a GaussDB table is a partitioned table (relkind = 'p').
        """
        query = text("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_partition p
                JOIN pg_class c ON p.parentid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = :table_name
                  AND n.nspname = :schema
                  AND p.parttype = 'p'  -- 'p' means partitioned table (parent)
            );
        """)
        with self.engine.connect() as conn:
            return conn.execute(query, {"table_name": table_name, "schema": schema}).scalar()

    def create_index(
            self,
            table_name: str,
            index_name: str,
            column_names: List[str],
            index_type: str,
            metric_type: str,
            idx_params: Optional[str] = None,
            **kw,
    ):
        """Create common index or vector index.

        Args:
            metric_type:  l2 or cosine
            index_type: GsDiskANN or GsIVFFLAT
            table_name (string): table name
            index_name (string): index name
            column_names (List[string]): create index on which columns
            idx_params (Optional[str]): vector index params, for example 'distance=l2, type=hnsw, lib=vsag'
            **kw: additional keyword arguments
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        columns = [table.c[column_name] for column_name in column_names]

        # 判断是否为分区表
        local_index = self.__is_partitioned_table(table.name, schema=table.schema or "public")

        with self.engine.connect() as conn:
            with conn.begin():
                if index_type == IndexType.BM25:
                    idx = BM25Index(index_name, *columns, local_index=local_index, params=idx_params, **kw)
                elif index_type == IndexType.GSDISKANN or IndexType.GSIVFFLAT:
                    idx = VectorIndex(index_name, *columns, index_type=index_type, metric_type=metric_type,
                                      local_index=local_index, params=idx_params, **kw)
                else:
                    idx = Index(index_name, *columns, **kw)

                idx.create(self.engine, checkfirst=True)

    def create_vidx_with_vec_index_param(
            self,
            table_name: str,
            vidx_param: VecIndexParam,
    ):
        """Create vector index with vector index parameter.

        Args:
            table_name (string): table name
            vidx_param (VecIndexParam): vector index parameter
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        local_index = self.__is_partitioned_table(table.name, schema=table.schema or "public")

        with self.engine.connect() as conn:
            with conn.begin():
                idx = VectorIndex(vidx_param.index_name, table.c[vidx_param.field_name],
                                  index_type=vidx_param.index_type, metric_type=vidx_param.metric_type,
                                  local_index=local_index, params=vidx_param.param_str(),
                                  )
                idx.create(self.engine, checkfirst=True)

    def create_bm25_idx_with_bm25_index_param(
            self,
            table_name: str,
            bm25_idx_param: BM25IndexParam,
    ):
        """Create BM25 index with BM25 index parameter.
        
        Args:
            table_name (string) : table name
            bm25_idx_param (BM25IndexParam) : BM25 index parameter
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        local_index = self.__is_partitioned_table(table.name, schema=table.schema or "public")

        with self.engine.connect() as conn:
            with conn.begin():
                idx_cols = [table.c[bm25_idx_param.field_name]]
                bm25_idx = BM25Index(bm25_idx_param.index_name, *idx_cols, local_index=local_index,
                                    params=bm25_idx_param.param_str(),
                                    )
                bm25_idx.create(self.engine, checkfirst=True)

    def ann_search(
            self,
            table_name: str,
            vec_data: Union[list, dict],
            vec_column_name: str,
            distance_func,
            with_dist: bool = False,
            topk: int = 10,
            output_column_names: Optional[List[str]] = None,
            output_columns: Optional[Union[List, tuple]] = None,
            extra_output_cols: Optional[List] = None,
            where_clause=None,
            partition_names: Optional[List[str]] = None,
            idx_name_hint: Optional[List[str]] = None,
            distance_threshold: Optional[float] = None,
            **kwargs,
    ):  # pylint: disable=unused-argument
        """perform ann search.

        Args:
            table_name (string): table name
            vec_data (Union[list, dict]): the vector data to search
            vec_column_name (string): which vector field to search
            distance_func: function to calculate distance between vectors
            with_dist (bool): return result with distance
            topk (int): top K
            output_column_names (Optional[List[str]]): output fields
            output_columns (Optional[Union[List, tuple]]): output columns as SQLAlchemy Column objects
            extra_output_cols (Optional[List]): additional output columns
            where_clause: do ann search with filter
            partition_names (Optional[List[str]]): limit the query to certain partitions
            idx_name_hint (Optional[List[str]]): post-filtering enabled if vector index name is specified
                Or pre-filtering enabled
            distance_threshold (Optional[float]): filter results where distance <= threshold.
            **kwargs: additional arguments
        """
        if not (isinstance(vec_data, list) or isinstance(vec_data, dict)):
            raise ValueError("'vec_data' type must be in 'list'/'dict'")

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        columns: List[ColumnElement] = []

        # Handle output_columns
        if output_columns:
            columns = list(output_columns) if isinstance(output_columns, (list, tuple)) else [output_columns]
        elif output_column_names:
            columns = [table.c[column_name] for column_name in output_column_names]
        else:
            columns = [table.c[column.name] for column in table.columns]

        # Add extra output columns if provided
        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        # Construct distance expression based on vec_data type
        dist_expr = distance_func(
            table.c[vec_column_name],
            "[" + ",".join([str(np.float32(v)) for v in vec_data]) + "]" if isinstance(vec_data, list) else f"{vec_data}"
        )

        if with_dist:
            columns.append(dist_expr.label("score"))

        stmt = select(*columns)

        if where_clause is not None:
            stmt = stmt.where(*where_clause)

        # Add distance threshold filter in SQL WHERE clause
        if distance_threshold is not None:
            stmt = stmt.where(dist_expr <= distance_threshold)

        order_by_expr = text("score") if with_dist else dist_expr
        stmt = stmt.order_by(order_by_expr)

        stmt_str = (
                str(stmt.compile(
                    dialect=self.engine.dialect,
                    compile_kwargs={"literal_binds": True}
                ))
                + f" limit {topk}"
        )
        with self.engine.connect() as conn:
            with conn.begin():
                if idx_name_hint is not None:
                    select_idx = stmt_str.find("SELECT ")
                    stmt_str = (
                            f"SELECT /*+ indexscan({table_name} {idx_name_hint}) */ "
                            + stmt_str[select_idx + len("SELECT "):]
                    )

                if partition_names is None:
                    return conn.execute(text(stmt_str))

                partition_hint = f"PARTITION({', '.join(partition_names)})"
                stmt_str = self._insert_partition_hint_for_query_sql(stmt_str, partition_hint)

                return conn.execute(text(stmt_str))

    def post_ann_search(
            self,
            table_name: str,
            vec_data: list,
            vec_column_name: str,
            distance_func,
            with_dist: bool = False,
            topk: int = 10,
            output_column_names: Optional[List[str]] = None,
            extra_output_cols: Optional[List] = None,
            where_clause=None,
            partition_names: Optional[List[str]] = None,
            str_list: Optional[List[str]] = None,
            **kwargs,
    ):  # pylint: disable=unused-argument
        """Perform post ann search.

        Args:
            table_name (string): table name
            vec_data (list): the vector data to search
            vec_column_name (string): which vector field to search
            distance_func: function to calculate distance between vectors
            with_dist (bool): return result with distance
            topk (int): top K
            output_column_names (Optional[List[str]]): output fields
            extra_output_cols (Optional[List]): additional output columns
            where_clause: do ann search with filter
            partition_names (Optional[List[str]]): limit the query to certain partitions
            str_list (Optional[List[str]]): list to append SQL string to
            **kwargs: additional arguments
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        columns = []
        if output_column_names is not None:
            columns.extend([table.c[column_name] for column_name in output_column_names])
        else:
            columns.extend([table.c[column.name] for column in table.columns])
        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        if with_dist:
            columns.append(
                distance_func(
                    table.c[vec_column_name],
                    "[" + ",".join([str(np.float32(v)) for v in vec_data]) + "]",
                )
            )

        stmt = select(*columns)
        if where_clause is not None:
            stmt = stmt.where(*where_clause)
        stmt = stmt.order_by(
            distance_func(
                table.c[vec_column_name],
                "[" + ",".join([str(np.float32(v)) for v in vec_data]) + "]",
            )
        ).limit(topk)

        with self.engine.connect() as conn:
            with conn.begin():
                if partition_names is None:
                    if str_list is not None:
                        str_list.append(
                            str(stmt.compile(
                                dialect=self.engine.dialect,
                                compile_kwargs={"literal_binds": True}
                            ))
                        )
                    return conn.execute(stmt)
                stmt_str = str(stmt.compile(
                    dialect=self.engine.dialect,
                    compile_kwargs={"literal_binds": True}
                ))
                stmt_str = self._insert_partition_hint_for_query_sql(
                    stmt_str, f"PARTITION({', '.join(partition_names)})"
                )
                if str_list is not None:
                    str_list.append(stmt_str)
                return conn.execute(text(stmt_str))

    def bm25_search(
            self,
            table_name: str,
            search_text: str,
            column_name: str,
            with_score: bool = False,
            topk: int = 10,
            output_column_names: Optional[List[str]] = None,
            output_columns: Optional[Union[List, tuple]] = None,
            extra_output_cols: Optional[List] = None,
            where_clause=None,
            partition_names: Optional[List[str]] = None,
            idx_name_hint: Optional[str] = None,
            score_threshold: Optional[float] = None,
            **kwargs,
    ):
        """
        Perform BM25 full-text search using the '###' operator.

        Args:
            table_name (str): Table name.
            search_text (str): The query string for BM25 search.
            column_name (str): The text column to search against.
            with_score (bool): Whether to return BM25 score.
            topk (int): Number of top results to return.
            output_column_names (Optional[List[str]]): List of column names to return.
            output_columns (Optional[Union[List, tuple]]): SQLAlchemy Column objects to return.
            extra_output_cols (Optional[List]): Additional columns to include.
            where_clause: Filter condition(s) as SQLAlchemy expression(s).
            partition_names (Optional[List[str]]): Limit query to specific partitions.
            idx_name_hint (Optional[str]): Index name for indexscan hint (e.g., 'st_information_st_email_bm25_index').
            score_threshold (Optional[float]): Only return results with BM25 score >= this value.
        """
        # Validate inputs
        if not isinstance(search_text, str):
            raise ValueError("'query_text' must be a string for BM25 search.")

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        # Build output columns
        columns: List[ColumnElement] = []
        if output_columns:
            columns = list(output_columns) if isinstance(output_columns, (list, tuple)) else [output_columns]
        elif output_column_names:
            columns = [table.c[col] for col in output_column_names]
        else:
            columns = [col for col in table.columns.values()]

        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        # Construct BM25 score expression: "text_col ### 'query'"
        escaped_query = search_text.replace("'", "''")
        score_expr_str = f"{column_name} ### '{escaped_query}'"

        if with_score:
            columns.append(text(score_expr_str).label("score"))

        stmt = select(*columns)

        if where_clause is not None:
            if isinstance(where_clause, (list, tuple)):
                stmt = stmt.where(*where_clause)
            else:
                stmt = stmt.where(where_clause)

        if score_threshold is not None:
            stmt = stmt.where(text(f"({score_expr_str}) >= {float(score_threshold)}"))

        stmt = stmt.order_by(text(f"({score_expr_str}) DESC"))

        stmt = stmt.limit(topk)

        # Compile to raw SQL with literal binds
        stmt_str = str(stmt.compile(
            dialect=self.engine.dialect,
            compile_kwargs={"literal_binds": True}
        ))

        # Inject indexscan hint if provided
        if idx_name_hint:
            # Ensure only one hint; replace SELECT
            if "SELECT /*+" not in stmt_str:
                stmt_str = stmt_str.replace(
                    "SELECT ",
                    f"SELECT /*+ indexscan({table_name} {idx_name_hint}) */ ",
                    1
                )

        # Inject partition hint if needed
        if partition_names:
            stmt_str = self._insert_partition_hint_for_query_sql(
                stmt_str, f"PARTITION({', '.join(partition_names)})"
            )

        # Execute
        with self.engine.connect() as conn:
            return conn.execute(text(stmt_str))
