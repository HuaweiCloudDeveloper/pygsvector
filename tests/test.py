import pygsvector
from sqlalchemy.dialects import registry
from sqlalchemy.ext.asyncio import create_async_engine

uri: str = "127.0.0.1:2881"
user: str = "root@test"
password: str = ""
db_name: str = "test"
registry.register("postgresql.asyncgaussdb", "pygsvector", "AsyncGaussDBDialect")
connection_str = (
    f"postgresql+asyncgaussdb://{user}:{password}@{uri}/{db_name}"
)
engine = create_async_engine(connection_str)