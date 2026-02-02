import logging
import unittest
import time


from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy.dialects.postgresql import TEXT

from pygsvector import FLOATVECTOR, VectorIndex, BM25IndexParam, VecIndexParam, IndexType
from pygsvector.client.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HybridSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = HybridSearch(uri="10.25.106.118:36801",
                                   user="test",
                                   password="Gauss_234",
                                   db_name="testdb",
                                   )

    def _create_test_table(self, test_table_name: str):
        self.client.create_table(
            table_name=test_table_name,
            columns=[
                Column("id", Integer, primary_key=True, autoincrement=False),
                Column("source_id", VARCHAR(32)),
                Column("enabled", Integer),
                Column("vector", FLOATVECTOR(3)),
                Column("title", TEXT),
                Column("content", TEXT),
            ],
            indexes=[
                VectorIndex("vec_idx", "vector", index_type="gsdiskann", metric_type="l2", local_index=False)
            ],
        )

        for col in ["title", "content"]:
            self.client.create_bm25_idx_with_bm25_index_param(
                table_name=test_table_name,
                bm25_idx_param=BM25IndexParam(
                    index_name=f"bm25_idx_{col}",
                    field_name=col,
                ),
            )

        self.client.create_vidx_with_vec_index_param(
            table_name=test_table_name,
            vidx_param=VecIndexParam(
                index_name=f"vec_idx_{col}",
                field_name="vector",
                index_type=IndexType.GSDISKANN,
            ),
        )

        self.client.insert(
            table_name=test_table_name,
            data=[
                {
                    "id": 1,
                    "source_id": "3b767712b57211f09c170242ac130008",
                    "enabled": 1,
                    "vector": [1, 1, 1],
                    "title": "功能差异",
                    "content": "GaussDB 数据库的形态。",
                },
                {
                    "id": 2,
                    "vector": [1, 2, 3],
                    "enabled": 1,
                    "source_id": "3b791472b57211f09c170242ac130008",
                    "title": "快速体验 GaussDB",
                    "content": "本文根据使用场景详细介绍如何快速部署 GaussDB 数据库，旨在帮助您快速掌握并成功使用 GaussDB 数据库。",
                },
                {
                    "id": 3,
                    "source_id": "3b7af31eb57211f09c170242ac130008",
                    "enabled": 1,
                    "vector": [3, 2, 1],
                    "title": "配置最佳实践",
                    "content": "为了确保用户在各种业务场景下，能够基于 GaussDB 数据库获得比较好的性能，GaussDB 基于过往大量真实场景的调优经验总结了各类业务场景下一些核心配置项和变量的推荐配置。",
                },
                {
                    "id": 4,
                    "source_id": "3b7cb9ceb57211f09c170242ac130008",
                    "enabled": 1,
                    "vector": [2, 2, 2],
                    "title": "GaussDB 实时分析能力白皮书",
                    "content": "重点解读 GaussDB 实时分析能力的 8 大核心特性，以及在 HTAP 混合负载场景、实时数据分析场景，和 PL/SQL 批处理场景的应用实践与案例。",
                }
            ]
        )

    def _search_param(self):
        query = {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "fields": [
                                "content"
                            ],
                            "type": "best_fields",
                            "query": "((数据)^0.5106318299637825 (迁移)^0.2651122588583924 (GaussDB)^0.22425591117782506 (\"GaussDB 数据 迁移\"~2)^1.5)",
                            "minimum_should_match": "30%",
                            "boost": 1
                        }
                    }
                ],
                "filter": [
                    {
                        "terms": {
                            "source_id": [
                                "3b791472b57211f09c170242ac130008",
                                "3b7af31eb57211f09c170242ac130008"
                            ]
                        }
                    },
                    {
                        "bool": {
                            "must_not": [
                                {
                                    "range": {
                                        "enabled": {
                                            "lt": 1
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ],
                "boost": 0.7
            }
        }

        return {
            "query": query,
            "knn": {
                "field": "vector",
                "k": 1024,
                "num_candidates": 1024,
                "query_vector": [1, 2, 3],
                "filter": query,
                "similarity": 0.2
            },
            "from": 0,
            "size": 60
        }

    def test_search(self):
        test_table_name = "hybrid_search_test"
        self.client.drop_table_if_exist(test_table_name)

        self._create_test_table(test_table_name)
        body = self._search_param()

        start_time = time.time()
        res = self.client.search(index=test_table_name, body=body)
        end_time = time.time()
        print(f"执行时间: {end_time - start_time:.4f} 秒")
        assert isinstance(res, list)
        assert len(res) > 0
        self.client.drop_table_if_exist(test_table_name)


    def test_get_sql(self):
        test_table_name = "get_sql_test"
        self.client.drop_table_if_exist(test_table_name)

        self._create_test_table(test_table_name)
        body = self._search_param()

        start_time = time.time()
        sql = self.client.get_sql(index=test_table_name, body=body)
        end_time = time.time()
        print(f"执行时间: {end_time - start_time:.4f} 秒")
        res = self.client.perform_raw_text_sql(sql).fetchall()
        assert len(res) > 0
        self.client.drop_table_if_exist(test_table_name)

