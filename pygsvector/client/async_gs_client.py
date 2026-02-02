"""Async GaussDB Client."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Union
from urllib.parse import quote

import sqlalchemy.sql.functions as func_mod
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Index,
    select,
    delete,
    update,
    insert,
    text,
    inspect,
    and_,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects import registry
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine
from asyncio import events

from .index_param import IndexParams
from .partitions import GsPartition
from ..schema import (
    l2_distance,
    cosine_distance,
    hamming_bool_distance,
    ReplaceStmt,
)
from ..util import GsDBVersion, Vector

logger = logging.getLogger(__name__)

async def register_codec(conn):
    """手动注册 floatvector 编解码器"""
    await conn.set_type_codec(
        "floatvector",
        schema="pg_catalog",
        encoder=str,
        decoder=str,
        format="text"
    )


class AsyncGsClient:
    def __init__(
            self,
            uri: str = "127.0.0.1:8000",
            user: str = "usr",
            password: str = " ",
            db_name: str = "gaussdb",
            **kwargs,
    ):
        registry.register("gaussdb.asyncpg", "pygsvector", "AsyncGaussDBDialect")

        setattr(func_mod, "l2_distance", l2_distance)
        setattr(func_mod, "cosine_distance", cosine_distance)
        setattr(func_mod, "hamming_bool_distance", hamming_bool_distance)

        user = quote(user, safe="")
        password = quote(password, safe="")

        connection_str = f"gaussdb+asyncpg://{user}:{password}@{uri}/{db_name}"
        self.engine: AsyncEngine = create_async_engine(connection_str,
                                                       **kwargs)

        
        self.metadata_obj = MetaData()
        self.gs_version = None  # 占位，避免 AttributeError
        
        # 设置 codec 自动注册拦截器
        self._setup_codec_interceptor()

    @classmethod
    async def create(cls, *args, **kwargs):
        """异步工厂方法"""
        instance = cls(*args, **kwargs)
        await instance._init_version()
        return instance

    async def _init_version(self):
        async with self.engine.connect() as conn:
            # 确保在当前连接上注册 floatvector codec
            await self._ensure_codec_registered(conn)
            async with conn.begin():
                # 获取底层的 asyncpg 连接
                raw_conn = await conn.get_raw_connection()
                asyncpg_conn = raw_conn.driver_connection
                await register_codec(asyncpg_conn)
                res = await conn.execute(text("SELECT VERSION()"))
                version = [r[0] for r in res][0]
                self.gs_version = GsDBVersion.from_db_version_string(version)
                print(self.gs_version)

    

    def _setup_codec_interceptor(self):
        """设置 codec 自动注册机制"""
        # 使用字典来跟踪每个连接的 codec 注册状态
        from weakref import WeakKeyDictionary
        self._codec_connection_registry = WeakKeyDictionary()

    async def _ensure_codec_registered(self, conn):
        """确保在当前连接上已注册 floatvector codec"""
        try:
            # 检查是否已经在当前连接上注册过
            if conn not in self._codec_connection_registry:
                # 获取底层的 asyncpg 连接并注册 codec
                raw_conn = await conn.get_raw_connection()
                asyncpg_conn = raw_conn.driver_connection
                await register_codec(asyncpg_conn)
                self._codec_connection_registry[conn] = True
        except Exception as e:
            logger.warning(f"Failed to register floatvector codec: {e}")
            # 不抛出异常，避免影响主要操作

    async def refresh_metadata(self, tables: Optional[List[str]] = None):
        async with self.engine.connect() as conn:
            await conn.run_sync(
                lambda sync_conn: self.metadata_obj.reflect(
                    bind=sync_conn, only=tables, extend_existing=True
                )
            )

    def _insert_partition_hint_for_query_sql(self, sql: str, partition_hint: str) -> str:
        """为SQL查询插入分区提示"""
        from_index = sql.find("FROM")
        assert from_index != -1
        first_space_after_from = sql.find(" ", from_index + len("FROM") + 1)
        if first_space_after_from == -1:
            return sql + " " + partition_hint
        return (
                sql[:first_space_after_from]
                + " "
                + partition_hint
                + sql[first_space_after_from:]
        )

    async def check_table_exists(self, table_name: str) -> bool:
        """异步检查表是否存在

        Args:
            table_name: 表名

        Returns:
            bool: 如果表存在返回True，否则返回False
        """
        async with self.engine.connect() as conn:
            # 使用 run_sync 在同步上下文中执行检查
            return await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).has_table(table_name)
            )

    async def create_table(
            self,
            table_name: str,
            columns: List[Column],
            indexes: Optional[List[Index]] = None,
            partitions: Optional[GsPartition] = None,
            **kwargs,
    ):
        """异步创建表"""
        partition_kwargs = {}
        if partitions is not None:
            partition_kwargs["postgresql_partition_by"] = partitions.do_compile()

        table = Table(
            table_name,
            self.metadata_obj,
            *columns,
            **partition_kwargs,
            **kwargs,
        )

        async with self.engine.connect() as conn:
            async with conn.begin():
                # 创建表
                await conn.run_sync(lambda sync_conn: table.create(sync_conn, checkfirst=True))

                # 创建索引
                if indexes:
                    for idx in indexes:
                        if idx.table is not table:
                            # 绑定索引到表（如果尚未绑定）
                            idx._set_table(table)
                        await conn.run_sync(lambda sync_conn, i=idx: i.create(sync_conn, checkfirst=True))

    @classmethod
    def prepare_index_params(cls):
        """创建索引参数容器"""
        return IndexParams()

    async def drop_table_if_exist(self, table_name: str):
        """异步删除表（如果存在）

        Args:
            table_name: 表名
        """
        async with self.engine.begin() as conn:
            await conn.execute(
                text(f'DROP TABLE IF EXISTS "{table_name}"')
            )
            # 清除 SQLAlchemy MetaData 中的表定义
            if table_name in self.metadata_obj.tables:
                self.metadata_obj.remove(self.metadata_obj.tables[table_name])

    async def rebuild_index(
            self,
            table_name: str,
            index_name: str,
    ):
        """异步重建向量索引以提高性能

        Args:
            table_name: 表名
            index_name: 索引名
        """
        async with self.engine.connect() as conn:
            # 使用 run_sync 在同步上下文中执行重建索引
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(f"REINDEX INDEX CONCURRENTLY {index_name}")
                )
            )

    async def drop_index(self, table_name: str, index_name: str):
        """异步删除索引

        Args:
            table_name: 表名
            index_name: 索引名
        """
        async with self.engine.connect() as conn:
            # 使用 run_sync 在同步上下文中执行删除索引
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(text(f"DROP INDEX {index_name}"))
            )

    async def insert(
            self,
            table_name: str,
            data: Union[Dict, List[Dict]],
            partition_name: Optional[str] = "",
    ):
        """异步插入数据

        Args:
            table_name: 表名
            data: 要插入的数据，可以是单个字典或字典列表
            partition_name: 限制查询到特定分区
        """
        if isinstance(data, Dict):
            data = [data]

        if len(data) == 0:
            return

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        async with self.engine.connect() as conn:
            # 确保在当前连接上注册 floatvector codec
            await self._ensure_codec_registered(conn)
            async with conn.begin():
                if partition_name is None or partition_name == "":
                    await conn.execute(insert(table).values(data))
                else:
                    await conn.execute(
                        insert(table)
                        .with_hint(f"PARTITION({partition_name})")
                        .values(data)
                    )

    async def upsert(
            self,
            table_name: str,
            data: Union[Dict, List[Dict]],
            partition_name: Optional[str] = "",
    ):
        """异步更新数据，如果主键存在则替换

        Args:
            table_name: 表名
            data: 要upsert的数据，可以是单个字典或字典列表
            partition_name: 限制查询到特定分区
        """
        if isinstance(data, Dict):
            data = [data]

        if len(data) == 0:
            return

        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        async with self.engine.connect() as conn:
            async with conn.begin():
                upsert_stmt = (
                    ReplaceStmt(table).with_hint(f"PARTITION({partition_name})")
                    if partition_name is not None and partition_name != ""
                    else ReplaceStmt(table)
                )
                upsert_stmt = upsert_stmt.values(data)
                await conn.execute(upsert_stmt)

    async def update(
            self,
            table_name: str,
            values_clause,
            where_clause=None,
            partition_name: Optional[str] = "",
    ):
        """异步更新表中的数据

        Args:
            table_name: 表名
            values_clause: 更新值子句
            where_clause: 更新条件的过滤
            partition_name: 限制查询到特定分区

        Example:
            client.update(
                table_name="my_table",
                values_clause=[{"meta": {"doc": "HHH"}}],
                where_clause=[text("id=112")]
            )
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)

        async with self.engine.connect() as conn:
            async with conn.begin():
                update_stmt = (
                    update(table).with_hint(f"PARTITION({partition_name})")
                    if partition_name is not None and partition_name != ""
                    else update(table)
                )
                if where_clause is not None:
                    update_stmt = update_stmt.where(*where_clause).values(
                        *values_clause
                    )
                else:
                    update_stmt = update_stmt.values(*values_clause)
                await conn.execute(update_stmt)

    async def delete(
            self,
            table_name: str,
            ids: Optional[Union[list, str, int]] = None,
            where_clause=None,
            partition_name: Optional[str] = "",
    ):
        """异步删除表中的数据

        Args:
            table_name: 表名
            ids: 要删除的ID列表或单个ID
            where_clause: 删除条件的过滤
            partition_name: 限制查询到特定分区
        """
        table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        where_in_clause = None

        if ids is not None:
            primary_keys = table.primary_key
            pkey_names = [column.name for column in primary_keys]

            if len(pkey_names) > 1:
                raise TypeError("使用'ids'时表不能有复合主键")

            if isinstance(ids, list):
                where_in_clause = table.c[pkey_names[0]].in_(ids)
            elif isinstance(ids, (str, int)):
                where_in_clause = table.c[pkey_names[0]].in_([ids])
            else:
                raise TypeError("'ids'必须是list/str/int类型")

        async with self.engine.connect() as conn:
            async with conn.begin():
                delete_stmt = (
                    delete(table).with_hint(f"PARTITION({partition_name})")
                    if partition_name is not None and partition_name != ""
                    else delete(table)
                )

                if where_in_clause is None and where_clause is None:
                    await conn.execute(delete_stmt)
                elif where_in_clause is not None and where_clause is None:
                    await conn.execute(delete_stmt.where(where_in_clause))
                elif where_in_clause is None and where_clause is not None:
                    await conn.execute(delete_stmt.where(*where_clause))
                else:
                    await conn.execute(
                        delete_stmt.where(and_(where_in_clause, *where_clause))
                    )

    async def get(
            self,
            table_name: str,
            ids: Optional[Union[list, str, int]] = None,
            where_clause=None,
            output_column_names: Optional[List[str]] = None,
            partition_names: Optional[List[str]] = None,
            n_limits: Optional[int] = None,
            idx_name_hint: Optional[str] = None,
    ):
        """异步获取指定主键的记录

        Args:
            table_name: 表名
            ids: 指定主键字段值
            where_clause: SQL过滤条件
            output_column_names: 输出字段名称列表
            partition_names: 限制查询到特定分区列表
            n_limits: 限制结果数量
            idx_name_hint: 索引提示名称

        Returns:
            SQLAlchemy异步执行结果
        """
        async with self.engine.connect() as conn:
            # 确保在当前连接上注册 floatvector codec
            await self._ensure_codec_registered(conn)
            # 使用 run_sync 在同步上下文中加载表结构
            table = await conn.run_sync(
                lambda sync_conn: Table(table_name, self.metadata_obj, autoload_with=sync_conn)
            )

            if output_column_names is not None:
                columns = [table.c[column_name] for column_name in output_column_names]
                stmt = select(*columns)
            else:
                stmt = select(table)

            primary_keys = table.primary_key
            pkey_names = [column.name for column in primary_keys]
            where_in_clause = None

            if ids is not None and len(pkey_names) == 1:
                if isinstance(ids, list):
                    where_in_clause = table.c[pkey_names[0]].in_(ids)
                elif isinstance(ids, (str, int)):
                    where_in_clause = table.c[pkey_names[0]].in_([ids])
                else:
                    raise TypeError("'ids'必须是list/str/int类型")

            if where_in_clause is not None and where_clause is None:
                stmt = stmt.where(where_in_clause)
            elif where_in_clause is None and where_clause is not None:
                stmt = stmt.where(*where_clause)
            elif where_in_clause is not None and where_clause is not None:
                stmt = stmt.where(and_(where_in_clause, *where_clause))

            if n_limits is not None:
                stmt = stmt.limit(n_limits)

            stmt_str = str(stmt.compile(
                dialect=self.engine.dialect,
                compile_kwargs={"literal_binds": True}
            ))

            if idx_name_hint is not None:
                hint_comment = f"/*+ indexscan({table_name} {idx_name_hint}) */"
                if stmt_str.strip().upper().startswith("SELECT"):
                    stmt_str = "SELECT " + hint_comment + " " + stmt_str[len("SELECT"):].lstrip()
                else:
                    stmt_str = f"{hint_comment} {stmt_str}"

            if partition_names is not None:
                stmt_str = self._insert_partition_hint_for_query_sql(
                    stmt_str, f"PARTITION({', '.join(partition_names)})"
                )

            logging.debug(stmt_str)

            return await conn.execute(text(stmt_str))

    async def perform_raw_text_sql(
            self,
            text_sql: str,
    ):
        """异步执行原始文本SQL

        Args:
            text_sql: 要执行的SQL语句
        """
        async with self.engine.connect() as conn:
            # 确保在当前连接上注册 floatvector codec
            await self._ensure_codec_registered(conn)
            # 使用 run_sync 在同步上下文中执行原始SQL
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(text(text_sql))
            )

    async def add_columns(
            self,
            table_name: str,
            columns: List[Column],
    ):
        """异步向现有表添加多个列

        Args:
            table_name: 表名
            columns: SQLAlchemy Column对象列表，代表新列
        """
        compiler = self.engine.dialect.ddl_compiler(self.engine.dialect, None)
        column_specs = [compiler.get_column_specification(column) for column in columns]
        columns_ddl = ", ".join(f"ADD COLUMN {spec}" for spec in column_specs)

        async with self.engine.connect() as conn:
            # 使用 run_sync 在同步上下文中执行ALTER TABLE
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(f"ALTER TABLE {table_name} {columns_ddl}")
                )
            )

        await self.refresh_metadata([table_name])

    async def drop_columns(
            self,
            table_name: str,
            column_names: list[str],
    ):
        """异步从现有表删除多个列

        Args:
            table_name: 表名
            column_names: 要删除的列名列表
        """
        columns_ddl = ", ".join(f"DROP COLUMN {name}" for name in column_names)

        async with self.engine.connect() as conn:
            # 使用 run_sync 在同步上下文中执行ALTER TABLE
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(f"ALTER TABLE {table_name} {columns_ddl}")
                )
            )

        await self.refresh_metadata([table_name])

    async def close(self):
        """异步关闭数据库连接"""
        # 清理 codec 注册状态跟踪字典
        if hasattr(self, '_codec_connection_registry'):
            self._codec_connection_registry.clear()
        
        await self.engine.dispose()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
