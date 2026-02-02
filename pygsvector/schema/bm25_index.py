"""FullTextIndex: full text search index type"""
from sqlalchemy import Index
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql.ddl import SchemaGenerator


class CreateBM25Index(DDLElement):
    """A new statement clause to create bm25 index.
    
    Attributes:
    index : bm25 index schema
    """

    def __init__(self, index):
        self.index = index


class GsBM25SchemaGenerator(SchemaGenerator):
    """A new schema generator to handle create bm25 index statement."""

    def visit_fts_index(self, index, create_ok=False):
        """Handle create bm25 index statement compiling.

        Args:
            index: bm25 index schema
            create_ok: the schema is created or not
        """
        if not create_ok and not self._can_create_index(index):
            return
        with self.with_ddl_events(index):
            CreateBM25Index(index)._invoke_with(self.connection)


class BM25Index(Index):
    """BM25 Index schema."""
    __visit_name__ = "fts_index"

    def __init__(self, name, *column_names, local_index: bool, params: str = None, **kw):
        self.params = params
        self.local_index = local_index
        if len(column_names) > 1:
            raise ValueError(
                f"expected single column for bm25 index: {len(column_names)}"
            )
        super().__init__(name, *column_names, **kw)

    def create(self, bind, checkfirst: bool = False) -> None:
        """Create bm25 index.
        
        Args:
            bind: SQL engine or connection.
            checkfirst: check the index exists or not.
        """
        bind._run_ddl_visitor(GsBM25SchemaGenerator, self, checkfirst=checkfirst)


@compiles(CreateBM25Index)
def compile_create_bm25_index(element, compiler, **kw):  # pylint: disable=unused-argument
    """A decorator function to compile create bm25 index statement."""
    index = element.index
    table_name = index.table.name
    column_list = ", ".join([column.name for column in index.columns])
    local_clause = " LOCAL" if index.local_index else ""

    return f"CREATE INDEX {index.name} ON {table_name} USING BM25({column_list}) {local_clause} WITH ({index.params})"
