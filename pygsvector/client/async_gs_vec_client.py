"""Async GaussDB Vector Store Client."""

import asyncio
import logging
from typing import List, Optional, Union, Any, AsyncGenerator

import numpy as np
from sqlalchemy import Table, Column, Index, select, text, ColumnElement
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection, AsyncResult

from .exceptions import ClusterVersionException, ErrorCode, ExceptionsMessage
from .async_gs_client import AsyncGsClient
from .index_param import IndexParams, VecIndexParam, BM25IndexParam, IndexType
from .partitions import GsPartition
from ..schema import VectorIndex, BM25Index
from ..util import GsDBVersion

logger = logging.getLogger(__name__)


class AsyncGsVecClient(AsyncGsClient):
    @classmethod
    async def create(cls, *args, **kwargs):
        instance = await super().create(*args, **kwargs)  # 调用 AsyncGsClient.create
        if instance.gs_version < GsDBVersion.from_db_version_nums(505, 1, 0):
            raise ClusterVersionException(
                code=ErrorCode.NOT_SUPPORTED,
                message=ExceptionsMessage.ClusterVersionIsLow % ("Vector Store", "505.1.0"),
            )
        return instance

    async def create_table_with_index_params(
            self,
            table_name: str,
            columns: List[Column],
            indexes: Optional[List[Index]] = None,
            index_params: Optional[IndexParams] = None,
            partitions: Optional[GsPartition] = None,
    ):
        """异步创建表和索引

        Args:
            table_name: 表名
            columns: 列定义列表
            indexes: 可选的普通索引列表
            index_params: 可选的向量和BM25搜索索引参数
            partitions: 可选的分区策略
        """
        async with self.engine.begin() as conn:  # 推荐直接 begin()
            # 处理分区
            partition_kwargs = {}
            local_index = False
            if partitions is not None:
                partition_kwargs["postgresql_partition_by"] = partitions.do_compile()
                local_index = True

            # 构造表对象
            if indexes is not None:
                table = Table(table_name, self.metadata_obj, *columns, *indexes,
                              extend_existing=True, **partition_kwargs)
            else:
                table = Table(table_name, self.metadata_obj, *columns,
                              extend_existing=True, **partition_kwargs)

            await conn.run_sync(lambda c: table.create(c, checkfirst=True))

            # 创建向量和 BM25 索引
            if index_params is not None:
                for idx_param in index_params:
                    if isinstance(idx_param, VecIndexParam):
                        # 创建向量索引对象
                        idx = VectorIndex(
                            idx_param.index_name,
                            table.c[idx_param.field_name],
                            index_type=idx_param.index_type,
                            metric_type=idx_param.metric_type,
                            local_index=local_index,
                            params=idx_param.param_str(),
                        )
                    elif isinstance(idx_param, BM25IndexParam):
                        # 创建全文搜索索引对象
                        idx = BM25Index(
                            idx_param.index_name,
                            table.c[idx_param.field_name],
                            local_index=local_index,
                            params=idx_param.param_str(),
                        )
                    else:
                        continue  # 或抛出异常

                    await conn.run_sync(lambda c: idx.create(c, checkfirst=True))

    async def __is_partitioned_table(self, table_name: str, schema: str = "public") -> bool:
        """异步检查是否为分区表

        Args:
            table_name: 表名
            schema: 模式名，默认为public

        Returns:
            bool: 如果是分区表返回True，否则返回False
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
        async with self.engine.connect() as conn:
            result = await conn.execute(query, {"table_name": table_name, "schema": schema})
            return result.scalar()

    async def create_index(
        self,
        table_name: str,
        index_name: str,
        column_names: List[str],
        index_type: str,
        metric_type: str,
        idx_params: Optional[str] = None,
        **kw,
    ):
        """异步创建普通索引或向量索引

        Args:
            table_name: 表名
            index_name: 索引名
            column_names: 创建索引的列名列表
            index_type: 索引类型（BM25, GsDiskANN, GSIVFFLAT等）
            metric_type: 距离类型（l2 或 cosine）
            idx_params: 向量索引参数，例如 'distance=l2, type=hnsw, lib=vsag'
            **kw: 额外的关键字参数
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        columns = [table.c[column_name] for column_name in column_names]

        # 判断是否为分区表
        local_index = await self.__is_partitioned_table(table.name, schema=table.schema or "public")

        async with self.engine.connect() as conn:
            async with conn.begin():
                if index_type == IndexType.BM25:
                    idx = BM25Index(index_name, *columns, local_index=local_index, params=idx_params, **kw)
                elif index_type == IndexType.GSDISKANN or index_type == IndexType.GSIVFFLAT:
                    idx = VectorIndex(index_name, *columns, index_type=index_type, metric_type=metric_type,
                                      local_index=local_index, params=idx_params, **kw)
                else:
                    idx = Index(index_name, *columns, **kw)

                await idx.create(self.engine, checkfirst=True)

    async def create_vidx_with_vec_index_param(
        self,
        table_name: str,
        vidx_param: VecIndexParam,
    ):
        """异步使用向量索引参数创建向量索引

        Args:
            table_name: 表名
            vidx_param: 向量索引参数
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        local_index = await self.__is_partitioned_table(table.name, schema=table.schema or "public")

        async with self.engine.connect() as conn:
            async with conn.begin():
                idx = VectorIndex(vidx_param.index_name, table.c[vidx_param.field_name],
                                  index_type=vidx_param.index_type, metric_type=vidx_param.metric_type,
                                  local_index=local_index, params=vidx_param.param_str(),
                                  )
                await idx.create(self.engine, checkfirst=True)

    async def create_bm25_idx_with_bm25_index_param(
        self,
        table_name: str,
        bm25_idx_param: BM25IndexParam,
    ):
        """异步使用BM25索引参数创建BM25索引

        Args:
            table_name: 表名
            bm25_idx_param: BM25索引参数
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        local_index = await self.__is_partitioned_table(table.name, schema=table.schema or "public")

        async with self.engine.connect() as conn:
            async with conn.begin():
                idx_cols = [table.c[bm25_idx_param.field_name]]
                bm25_idx = BM25Index(bm25_idx_param.index_name, *idx_cols, local_index=local_index,
                                    params=bm25_idx_param.param_str(),
                                    )
                await bm25_idx.create(self.engine, checkfirst=True)

    async def ann_search(
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
    ):
        """异步执行近似最近邻搜索（ANN搜索）

        Args:
            table_name: 表名
            vec_data: 要搜索的向量数据，可以是列表或字典
            vec_column_name: 要搜索的向量字段名
            distance_func: 计算向量间距离的函数
            with_dist: 是否返回距离分数
            topk: 返回前K个结果
            output_column_names: 输出字段名列表
            output_columns: SQLAlchemy Column对象作为输出列
            extra_output_cols: 额外的输出列
            where_clause: 搜索条件过滤
            partition_names: 限制查询到特定分区列表
            idx_name_hint: 指定向量索引名称用于post-filtering或pre-filtering
            distance_threshold: 距离阈值过滤
            **kwargs: 额参数

        Returns:
            异步查询结果
        """
        if not (isinstance(vec_data, list) or isinstance(vec_data, dict)):
            raise ValueError("'vec_data'类型必须是'list'/'dict'")

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        columns: List[ColumnElement] = []

        # 处理输出列
        if output_columns:
            columns = list(output_columns) if isinstance(output_columns, (list, tuple)) else [output_columns]
        elif output_column_names:
            columns = [table.c[column_name] for column_name in output_column_names]
        else:
            columns = [table.c[column.name] for column in table.columns]

        # 添加额外的输出列
        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        # 构建距离表达式
        dist_expr = distance_func(
            table.c[vec_column_name],
            "[" + ",".join([str(np.float32(v)) for v in vec_data]) + "]" if isinstance(vec_data, list) else f"{vec_data}"
        )

        if with_dist:
            columns.append(dist_expr.label("score"))

        stmt = select(*columns)

        if where_clause is not None:
            stmt = stmt.where(*where_clause)

        # 添加距离阈值过滤
        if distance_threshold is not None:
            stmt = stmt.where(dist_expr <= distance_threshold)

        order_by_expr = text("score") if with_dist else dist_expr
        stmt = stmt.order_by(order_by_expr).limit(topk)

        stmt_str = str(stmt.compile(
            dialect=self.engine.dialect,
            compile_kwargs={"literal_binds": True}
        ))

        async with self.engine.connect() as conn:
            async with conn.begin():
                if idx_name_hint is not None:
                    select_idx = stmt_str.find("SELECT ")
                    stmt_str = (
                            f"SELECT /*+ indexscan({table_name} {idx_name_hint}) */ "
                            + stmt_str[select_idx + len("SELECT "):]
                    )

                if partition_names is None:
                    return await conn.execute(text(stmt_str))

                partition_hint = f"PARTITION({', '.join(partition_names)})"
                stmt_str = self._insert_partition_hint_for_query_sql(stmt_str, partition_hint)
                return await conn.execute(text(stmt_str))

    async def post_ann_search(
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
    ):
        """异步执行后处理ANN搜索

        Args:
            table_name: 表名
            vec_data: 要搜索的向量数据
            vec_column_name: 要搜索的向量字段名
            distance_func: 计算向量间距离的函数
            with_dist: 是否返回距离分数
            topk: 返回前K个结果
            output_column_names: 输出字段名列表
            extra_output_cols: 额外的输出列
            where_clause: 搜索条件过滤
            partition_names: 限制查询到特定分区列表
            str_list: 可选，用于追加SQL字符串的列表
            **kwargs: 额参数

        Returns:
            异步查询结果
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
                ).label("score")
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

        async with self.engine.connect() as conn:
            async with conn.begin():
                if partition_names is None:
                    if str_list is not None:
                        str_list.append(
                            str(stmt.compile(
                                dialect=self.engine.dialect,
                                compile_kwargs={"literal_binds": True}
                            ))
                        )
                    return await conn.execute(stmt)

                stmt_str = str(stmt.compile(
                    dialect=self.engine.dialect,
                    compile_kwargs={"literal_binds": True}
                ))
                stmt_str = self._insert_partition_hint_for_query_sql(
                    stmt_str, f"PARTITION({', '.join(partition_names)})"
                )
                if str_list is not None:
                    str_list.append(stmt_str)
                return await conn.execute(text(stmt_str))

    async def bm25_search(
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
        """异步执行BM25全文搜索

        Args:
            table_name: 表名
            search_text: BM25搜索的查询字符串
            column_name: 要搜索的文本列名
            with_score: 是否返回BM25分数
            topk: 返回前K个结果
            output_column_names: 输出字段名列表
            output_columns: SQLAlchemy Column对象作为输出列
            extra_output_cols: 额外的输出列
            where_clause: 搜索条件过滤
            partition_names: 限制查询到特定分区列表
            idx_name_hint: 索引提示名称（例如 'st_information_st_email_bm25_index'）
            score_threshold: 只返回BM25分数大于此值的结果
            **kwargs: 额参数

        Returns:
            异步BM25搜索结果
        """
        if not isinstance(search_text, str):
            raise ValueError("'query_text'必须是字符串才能执行BM25搜索")

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        # 构建输出列
        columns: List[ColumnElement] = []
        if output_columns:
            columns = list(output_columns) if isinstance(output_columns, (list, tuple)) else [output_columns]
        elif output_column_names:
            columns = [table.c[col] for col in output_column_names]
        else:
            columns = [col for col in table.columns.values()]

        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        # 构建BM25分数表达式
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

        stmt = stmt.order_by(text(f"({score_expr_str}) DESC")).limit(topk)

        stmt_str = str(stmt.compile(
            dialect=self.engine.dialect,
            compile_kwargs={"literal_binds": True}
        ))

        # 注入索引扫描提示
        if idx_name_hint:
            if "SELECT /*+" not in stmt_str:
                stmt_str = stmt_str.replace(
                    "SELECT ",
                    f"SELECT /*+ indexscan({table_name} {idx_name_hint}) */ ",
                    1
                )

        # 注入分区提示
        if partition_names:
            stmt_str = self._insert_partition_hint_for_query_sql(
                stmt_str, f"PARTITION({', '.join(partition_names)})"
            )

        async with self.engine.connect() as conn:
            return await conn.execute(text(stmt_str))

    async def search_hybrid(
        self,
        table_name: str,
        vec_data: Union[list, dict],
        vec_column_name: str,
        distance_func,
        search_text: str,
        text_column_name: str,
        hybrid_ratio: float = 0.5,
        with_dist: bool = False,
        with_bm25_score: bool = False,
        topk: int = 10,
        alpha: float = 0.5,
        output_column_names: Optional[List[str]] = None,
        output_columns: Optional[Union[List, tuple]] = None,
        extra_output_cols: Optional[List] = None,
        where_clause=None,
        partition_names: Optional[List[str]] = None,
        idx_name_hint: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        bm25_score_threshold: Optional[float] = None,
        **kwargs,
    ):
        """异步执行混合搜索（向量 + BM25）

        Args:
            table_name: 表名
            vec_data: 要搜索的向量数据
            vec_column_name: 向量字段名
            distance_func: 向量距离函数
            search_text: BM25搜索文本
            text_column_name: 文本字段名
            hybrid_ratio: 混合搜索比例，默认为0.5
            with_dist: 是否返回向量距离
            with_bm25_score: 是否返回BM25分数
            topk: 返回前K个结果
            alpha: 混合权重，0.0为纯BM25，1.0为纯向量
            output_column_names: 输出字段名列表
            output_columns: SQLAlchemy Column对象作为输出列
            extra_output_cols: 额外的输出列
            where_clause: 搜索条件过滤
            partition_names: 限制查询到特定分区列表
            idx_name_hint: 索引提示列表
            distance_threshold: 向量距离阈值
            bm25_score_threshold: BM25分数阈值
            **kwargs: 额参数

        Returns:
            异步混合搜索结果
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        # 构建输出列
        columns: List[ColumnElement] = []
        if output_columns:
            columns = list(output_columns) if isinstance(output_columns, (list, tuple)) else [output_columns]
        elif output_column_names:
            columns = [table.c[column_name] for column_name in output_column_names]
        else:
            columns = [table.c[column.name] for column in table.columns]

        if extra_output_cols is not None:
            columns.extend(extra_output_cols)

        # 构建向量距离表达式
        vec_dist_expr = distance_func(
            table.c[vec_column_name],
            "[" + ",".join([str(np.float32(v)) for v in vec_data]) + "]" if isinstance(vec_data, list) else f"{vec_data}"
        )

        # 构建BM25分数表达式
        escaped_query = search_text.replace("'", "''")
        bm25_score_expr = text(f"{text_column_name} ### '{escaped_query}'")

        # 添加到输出列
        if with_dist:
            columns.append(vec_dist_expr.label("vec_distance"))
        if with_bm25_score:
            columns.append(bm25_score_expr.label("bm25_score"))

        # 混合排序：alpha * vec_distance + (1-alpha) * normalized_bm25_score
        order_by_expr = text(
            f"({alpha} * CASE WHEN vec_distance IS NOT NULL THEN vec_distance END "
            f"+ {(1-alpha)} * CASE WHEN bm25_score IS NOT NULL THEN bm25_score END) ASC"
        )

        stmt = select(*columns)

        # 添加where条件
        if where_clause is not None:
            if isinstance(where_clause, (list, tuple)):
                stmt = stmt.where(*where_clause)
            else:
                stmt = stmt.where(where_clause)

        # 添加距离阈值
        if distance_threshold is not None:
            stmt = stmt.where(vec_dist_expr <= distance_threshold)

        # 添加BM25分数阈值
        if bm25_score_threshold is not None:
            stmt = stmt.where(bm25_score_expr >= bm25_score_threshold)

        stmt = stmt.order_by(order_by_expr).limit(topk)

        stmt_str = str(stmt.compile(
            dialect=self.engine.dialect,
            compile_kwargs={"literal_binds": True}
        ))

        async with self.engine.connect() as conn:
            async with conn.begin():
                if idx_name_hint is not None:
                    select_idx = stmt_str.find("SELECT ")
                    stmt_str = (
                            f"SELECT /*+ indexscan({table_name} {idx_name_hint}) */ "
                            + stmt_str[select_idx + len("SELECT "):]
                    )

                if partition_names:
                    stmt_str = self._insert_partition_hint_for_query_sql(
                        stmt_str, f"PARTITION({', '.join(partition_names)})"
                    )

                return await conn.execute(text(stmt_str))