import unittest
from pygsvector.client import GsVecClient, VecIndexType, IndexParam
from pygsvector.schema import FLOATVECTOR, VectorIndex
from sqlalchemy import Column, Integer, Table
from sqlalchemy.sql import func
from sqlalchemy.exc import NoSuchTableError


class GsVecClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = GsVecClient(echo=True)


if __name__ == "__main__":
    unittest.main()
