"""VectorIndex: An extended index type for SQLAlchemy"""
from sqlalchemy import Index
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql.ddl import SchemaGenerator


class CreateVectorIndex(DDLElement):
    """A new statement clause to create vector index.
    
    Attributes:
        index : vector index schema
    """

    def __init__(self, index):
        self.index = index


class GsVecSchemaGenerator(SchemaGenerator):
    """A new schema generator to handle create vector index statement."""

    def visit_vector_index(self, index, create_ok=False):
        """Handle create vector index statement compiling.

        Args:
            index: vector index schema
            create_ok: the schema is created or not
        """
        if not create_ok and not self._can_create_index(index):
            return
        with self.with_ddl_events(index):
            CreateVectorIndex(index)._invoke_with(self.connection)


class VectorIndex(Index):
    """Vector Index schema."""
    __visit_name__ = "vector_index"

    def __init__(self, name, *column_names, index_type: str, metric_type: str, local_index: bool, params: str = None,
                 **kw):
        if len(column_names) > 1:
            raise ValueError(
                f"expected single column for vector index: {len(column_names)}"
            )
        self.params = params
        self.index_type = index_type
        self.metric_type = metric_type
        self.local_index = local_index
        super().__init__(name, *column_names, **kw)

    def create(self, bind, checkfirst: bool = False) -> None:
        """Create vector index.
        
        Args:
            bind: SQL engine or connection.
            checkfirst: check the index exists or not.
        """
        bind._run_ddl_visitor(GsVecSchemaGenerator, self, checkfirst=checkfirst)


@compiles(CreateVectorIndex)
def compile_create_vector_index(element: object, compiler: object, **kw: object) -> object:
    """A decorator function to compile create vector index statement."""
    index = element.index
    table_name = index.table.name
    column_list = ", ".join([column.name for column in index.columns])

    # 构建索引语句
    using_clause = f"USING {index.index_type} ({column_list} {index.metric_type})"
    local_clause = " LOCAL" if index.local_index else ""

    if index.params is not None:
        return f"CREATE INDEX {index.name} ON {table_name} {using_clause}{local_clause} WITH ({index.params})"
    else:
        return f"CREATE INDEX {index.name} ON {table_name} {using_clause}{local_clause}"
