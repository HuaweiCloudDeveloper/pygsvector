import asyncio
import logging
import unittest

from sqlalchemy import Column, Integer, text, String
from sqlalchemy.dialects.postgresql import JSONB, TEXT

from pygsvector import *
from pygsvector.client import AsyncGsVecClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AsyncGsVecClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = await AsyncGsVecClient.create(
            uri="10.25.106.118:36801",
            user="test",
            password="Gauss_234",
            db_name="testdb",
        )
        await self.client.refresh_metadata()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_client_initialization(self):
        """测试异步客户端初始化"""
        self.assertIsNotNone(self.client)
        self.assertTrue(hasattr(self.client, 'engine'))
        # self.assertTrue(hasattr(self.client, 'metadata_obj'))
        # self.assertTrue(hasattr(self.client, 'gs_version'))

    async def test_refresh_metadata(self):
        """测试异步刷新元数据"""
        await self.client.refresh_metadata()
        self.assertIsNotNone(self.client.metadata_obj)

    async def test_check_table_exists(self):
        """测试异步检查表是否存在"""
        # 测试不存在的表
        exists = await self.client.check_table_exists("non_existent_table")
        self.assertFalse(exists)

    async def test_create_table(self):
        """测试异步创建表"""
        table_name = "test_async_table"
        await self.client.drop_table_if_exist(table_name)

        # 定义表结构
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
            Column("metadata", JSONB),
            Column("embedding", FLOATVECTOR(3))
        ]

        # 创建表
        await self.client.create_table(table_name, columns)

        # 验证表存在
        exists = await self.client.check_table_exists(table_name)
        self.assertTrue(exists)

    async def test_insert_and_select(self):
        """测试异步插入和查询数据"""
        table_name = "test_async_insert_select"
        await self.client.drop_table_if_exist(table_name)
        print("Actual dialect class:", self.client.engine.dialect.__class__)
        print("Dialect name:", self.client.engine.dialect.name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", TEXT),
            Column("embedding", FLOATVECTOR(3))
        ]
        await self.client.create_table(table_name, columns)

        # 插入数据
        test_data = [
            {"id": 1, "name": "test1", "embedding": [0.1, 0.2, 0.3]},
            {"id": 2, "name": "test2", "embedding": [0.4, 0.5, 0.6]}
        ]

        await self.client.insert(table_name, test_data)

        # 查询数据
        result = await self.client.get(table_name)
        rows = result.fetchall()

        self.assertEqual(len(rows), 2)

    async def test_upsert(self):
        """测试异步Upsert操作"""
        table_name = "test_async_upsert"
        await self.client.drop_table_if_exist(table_name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
            Column("metadata", JSONB)
        ]
        await self.client.create_table(table_name, columns)

        # 插入初始数据
        initial_data = [{"id": 1, "name": "original", "metadata": {"version": 1}}]
        await self.client.insert(table_name, initial_data)

        # Upsert新数据（会更新existing记录）
        upsert_data = [{"id": 1, "name": "updated", "metadata": {"version": 2}}]
        await self.client.upsert(table_name, upsert_data)

        # 验证数据已更新
        result = await self.client.get(table_name)
        row = result.fetchone()
        self.assertEqual(row.name, "updated")
        self.assertEqual(row.metadata["version"], 2)

    async def test_update(self):
        """测试异步Update操作"""
        table_name = "test_async_update"
        await self.client.drop_table_if_exist(table_name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
            Column("metadata", JSONB)
        ]
        await self.client.create_table(table_name, columns)

        # 插入数据
        data = [{"id": 1, "name": "test", "metadata": {"status": "active"}}]
        await self.client.insert(table_name, data)

        # 更新数据
        await self.client.update(
            table_name,
            values_clause=[{"name": "updated_test"}],
            where_clause=[text("id=1")]
        )

        # 验证更新
        result = await self.client.get(table_name)
        row = result.fetchone()
        self.assertEqual(row.name, "updated_test")
        self.assertEqual(row.metadata["status"], "active")  # metadata should remain unchanged

    async def test_delete(self):
        """测试异步Delete操作"""
        table_name = "test_async_delete"
        await self.client.drop_table_if_exist(table_name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", String(50))
        ]
        await self.client.create_table(table_name, columns)

        # 插入数据
        data = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"},
            {"id": 3, "name": "test3"}
        ]
        await self.client.insert(table_name, data)

        # 删除ID为2的记录
        await self.client.delete(table_name, ids=[2])

        # 验证删除
        result = await self.client.get(table_name)
        rows = result.fetchall()
        self.assertEqual(len(rows), 2)

    async def test_ann_search(self):
        """测试异步ANN搜索"""
        table_name = "test_async_ann_search"
        await self.client.drop_table_if_exist(table_name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", String(50)),
            Column("embedding", FLOATVECTOR(3))
        ]
        await self.client.create_table(table_name, columns)

        # 插入数据
        data = [
            {"id": 1, "name": "test1", "embedding": [0.1, 0.2, 0.3]},
            {"id": 2, "name": "test2", "embedding": [0.4, 0.5, 0.6]},
            {"id": 3, "name": "test3", "embedding": [0.7, 0.8, 0.9]}
        ]
        await self.client.insert(table_name, data)

        # 执行ANN搜索
        from pygsvector import l2_distance

        result = await self.client.ann_search(
            table_name=table_name,
            vec_data=[0.1, 0.2, 0.3],  # 查询向量
            vec_column_name="embedding",
            distance_func=l2_distance,
            topk=3
        )

        rows = result.fetchall()
        self.assertEqual(len(rows), 3)

        # 按距离排序，第一条应该是自己
        self.assertEqual(rows[0].id, 1)

    async def test_bm25_search(self):
        """测试异步BM25搜索"""
        table_name = "test_async_bm25_search"
        await self.client.drop_table_if_exist(table_name)


        cols = [
            Column("id", Integer, primary_key=True),
            Column("doc", TEXT),
        ]
        fts_index_param = BM25IndexParam(
            index_name="async_bm25_search_idx",
            field_name="doc",
            num_parallels="16",
        )
        await self.client.create_table_with_index_params(
            table_name=table_name,
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
        await self.client.insert(
            table_name,
            data=datas
        )

        res = await self.client.bm25_search(
            table_name,
            search_text="like",
            column_name="doc",
            with_score=False,
            where_clause=[text("id > 4")],
            topk=1,
            output_column_names=["id", "doc"],
        )

        self.assertEqual(set(res.fetchall()), set([(5, 'i like coding'), ]))


    # async def test_hybrid_search(self):
    #     """测试异步混合搜索"""
    #     table_name = "test_async_hybrid_search"
    #     await self.client.drop_table_if_exist(table_name)
    #
    #     # 创建表
    #     columns = [
    #         Column("id", Integer, primary_key=True),
    #         Column("content", String(255)),
    #         Column("embedding", FLOATVECTOR(3)),
    #         Column("metadata", JSONB)
    #     ]
    #     await self.client.create_table(table_name, columns)
    #
    #     # 插入数据
    #     data = [
    #         {"id": 1, "content": "machine learning AI", "embedding": [0.1, 0.2, 0.3], "metadata": {"category": "tech"}},
    #         {"id": 2, "content": "artificial intelligence ML", "embedding": [0.4, 0.5, 0.6],
    #          "metadata": {"category": "tech"}},
    #         {"id": 3, "content": "daily life routine", "embedding": [0.7, 0.8, 0.9], "metadata": {"category": "life"}}
    #     ]
    #     await self.client.insert(table_name, data)
    #
    #     # 执行混合搜索
    #     result = await self.client.search_hybrid(
    #         table_name=table_name,
    #         vec_data=[0.1, 0.2, 0.3],  # 向量搜索
    #         vec_column_name="embedding",
    #         distance_func=l2_distance,
    #         search_text="machine learning",  # BM25搜索
    #         text_column_name="content",
    #         topk=5
    #     )
    #
    #     rows = result.fetchall()
    #     self.assertGreater(len(rows), 0)

    async def test_table_with_index_params(self):
        """测试异步创建带索引参数的表"""
        table_name = "test_async_table_with_index_params"
        await self.client.drop_table_if_exist(table_name)

        # 定义表结构
        from pygsvector import IndexType
        columns = [
            Column("id", Integer, primary_key=True),
            Column("content", TEXT),
            Column("embedding", FLOATVECTOR(128)),
            Column("metadata", JSONB)
        ]

        # 定义索引参数
        index_params = self.client.prepare_index_params()

        # 添加向量索引
        index_params.add_index(
            field_name="embedding",
            index_type=IndexType.GSDISKANN,
            index_name="vec_idx",
            metric_type="L2",
            params={"pq_nclus": 16, "num_parallels": 50, "enable_pq": True},
        )
        # 添加BM25索引
        index_params.add_index(index_name="bm25_idx",
                               field_name="content",
                               index_type=IndexType.BM25,
                               )

        # 创建表和索引
        await self.client.create_table_with_index_params(
            table_name,
            columns,
            index_params=index_params
        )

        # 验证表存在
        exists = await self.client.check_table_exists(table_name)
        self.assertTrue(exists)

    async def test_parallel_operations(self):
        """测试并行操作"""
        table_name = "test_parallel_operations"
        await self.client.drop_table_if_exist(table_name)

        # 创建表
        columns = [
            Column("id", Integer, primary_key=True),
            Column("name", TEXT),
            Column("embedding", FLOATVECTOR(3))
        ]
        await self.client.create_table(table_name, columns)

        # 定义并行插入的函数
        async def insert_batch(start_id, data):
            for i in range(10):
                data[start_id + i]["id"] = start_id + i
            await self.client.insert(table_name, data[start_id:start_id + 10])

        # 并行插入多批次数据
        data = [{"name": f"test_{i}", "embedding": [0.1, 0.2, 0.3]} for i in range(100)]

        tasks = []
        for i in range(0, 100, 10):
            task = insert_batch(i, data)
            tasks.append(task)

        await asyncio.gather(*tasks)

        # 验证所有数据都已插入
        result = await self.client.get(table_name)
        rows = result.fetchall()
        self.assertEqual(len(rows), 100)

if __name__ == "__main__":
    unittest.main()
