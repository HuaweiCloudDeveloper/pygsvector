"""GaussDB dialect."""
from sqlalchemy import util
from sqlalchemy.dialects.postgresql import psycopg2 as pg_psycopg2
from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg
import re

from .floatvector import FLOATVECTOR
from sqlalchemy.dialects.postgresql.base import PGCompiler
from sqlalchemy import exc

class GaussDBCompiler(PGCompiler):
    def format_from_hint_text(self, sqltext, table, hint, iscrud):
        if hint.upper() == "ONLY":
            return "ONLY " + sqltext
        elif hint.upper().startswith("PARTITION(") and hint.endswith(")"):
            # 生成: table_name PARTITION(p0)
            return f"{sqltext} {hint}"
        else:
            raise exc.CompileError("Unrecognized hint: %r" % hint)

class GaussDBDialect(pg_psycopg2.PGDialect_psycopg2):
    """GaussDB dialect for PostgreSQL-compatible mode."""
    name = "gaussdb"

    statement_compiler = GaussDBCompiler

    supports_native_uuid = False
    supports_sane_multi_rowcount = False
    supports_statement_cache = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ischema_names["floatvector"] = FLOATVECTOR

    def _load_domains(self, connection, schema=None, **kw):
        return []

    def _load_enums(self, connection, schema=None, **kw):
        return []

    def _get_server_version_info(self, connection):
        v = connection.exec_driver_sql("select pg_catalog.version()").scalar()
        m = re.match(
            r".*(?:PostgreSQL|EnterpriseDB) "
            r"(\d+)\.?(\d+)?(?:\.(\d+))?(?:\.\d+)?(?:devel|beta)?",
            "PostgreSQL 9.2.1",
        )
        if not m:
            raise AssertionError(
                "Could not determine version from string '%s'" % v
            )
        return tuple([int(x) for x in m.group(1, 2, 3) if x is not None])


class AsyncGaussDBDialect(pg_asyncpg.PGDialect_asyncpg):
    """GaussDB dialect for PostgreSQL-compatible mode (async, using asyncpg)."""
    name = "gaussdb+asyncpg"  # 注意：URL 中会用这个名称，如 gaussdb+asyncpg://...

    statement_compiler = GaussDBCompiler  # 可复用相同的编译器

    supports_native_uuid = False
    supports_sane_multi_rowcount = False
    supports_statement_cache = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ischema_names["floatvector"] = FLOATVECTOR

    async def _load_domains(self, connection, schema=None, **kw):
        return []

    async def _load_enums(self, connection, schema=None, **kw):
        return []
    async def _get_server_version_info(self, connection):
        v = connection.exec_driver_sql("select pg_catalog.version()").scalar()
        m = re.match(
            r".*(?:PostgreSQL|EnterpriseDB) "
            r"(\d+)\.?(\d+)?(?:\.(\d+))?(?:\.\d+)?(?:devel|beta)?",
            "PostgreSQL 9.2.1",
        )
        if not m:
            raise AssertionError(
                "Could not determine version from string '%s'" % v
            )
        return tuple([int(x) for x in m.group(1, 2, 3) if x is not None])