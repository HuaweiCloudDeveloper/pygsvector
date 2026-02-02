import unittest
from pygsvector import *
from sqlalchemy import Column, Integer, text
from sqlalchemy.dialects.postgresql import TEXT
import logging

logger = logging.getLogger(__name__)


class GsFtsIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = GsVecClient(uri="10.25.106.118:36801",
                                  user="test",
                                  password="Gauss_234",
                                  db_name="testdb",
                                  )

    def test_bm25_index(self):
        test_collection_name = "fts_simple_test"
        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("doc", TEXT),
        ]
        self.client.create_table(
            test_collection_name,
            columns=cols,
        )
        fts_index_param = BM25IndexParam(
            index_name="fts_idx4",
            field_name="doc",
        )
        self.client.create_bm25_idx_with_bm25_index_param(
            test_collection_name,
            bm25_idx_param=fts_index_param,
        )

        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("doc", TEXT),
        ]
        fts_index_param = BM25IndexParam(
            index_name="fts_idx2",
            field_name="doc",
        )
        self.client.create_table_with_index_params(
            table_name=test_collection_name,
            columns=cols,
            index_params=[fts_index_param],
        )

    def test_bm25_insert_and_search(self):
        test_collection_name = "fts_data_test"
        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", Integer, primary_key=True),
            Column("doc", TEXT),
        ]
        fts_index_param = BM25IndexParam(
            index_name="fts_idx3",
            field_name="doc",
            num_parallels="16",
        )
        self.client.create_table_with_index_params(
            table_name=test_collection_name,
            columns=cols,
            index_params=[fts_index_param],
        )


        datas = [
            {"id": 1, "doc": "pLease porridge in the pot", },
            {"id": 2, "doc": "please say sorry", },
            {"id": 3, "doc": "nine years old", },
            {"id": 4, "doc": "some like it hot, some like it cold", },
            {"id": 5, "doc": "i like coding", },
            {"id": 6, "doc": "i like my company", },
        ]
        self.client.insert(
            test_collection_name,
            data=datas
        )

        res = self.client.bm25_search(
            test_collection_name,
            search_text="like",
            column_name="doc",
            with_score=False,
            where_clause=[text("id > 4")],
            topk=1,
            output_column_names=["id", "doc"],
        )

        self.assertEqual(set(res.fetchall()), set([(5, 'i like coding'), ]))
