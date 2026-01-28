"""GsDBTable: extension to Table for creating table with vector index."""
from sqlalchemy import Table
from .vector_index import GsSchemaGenerator


class GsDBTable(Table):
    """A class extends SQLAlchemy Table to do table creation with vector index."""
    def create(self, bind, checkfirst: bool = False) -> None:
        bind._run_ddl_visitor(GsSchemaGenerator, self, checkfirst=checkfirst)
