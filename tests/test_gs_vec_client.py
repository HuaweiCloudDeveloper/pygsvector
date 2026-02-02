import json
import unittest
from sqlalchemy.sql.selectable import and_
from pygsvector import *
from sqlalchemy import Column, Integer, JSON, String, text, Text, Table, BLOB
from sqlalchemy import func
import logging
from sqlalchemy.dialects.postgresql import ARRAY

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GsVecClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = GsVecClient(uri="10.25.106.118:36801",
                                  user="test",
                                  password="Gauss_234",
                                  db_name="testdb",
                                  )

    def test_ann_search(self):
        test_collection_name = "gs_ann_test"
        self.client.drop_table_if_exist(test_collection_name)

        # create partitioned table
        range_part = GsRangePartition(
            False,
            range_part_infos=[
                RangeListPartInfo("p0", 100),
                RangeListPartInfo("p1", "maxvalue"),
            ],
            range_expr="id",
        )

        cols = [
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("embedding", FLOATVECTOR(3)),
            Column("meta", JSON),
        ]
        self.client.create_table(
            test_collection_name, columns=cols, partitions=range_part
        )

        # create vector index
        self.client.create_index(
            test_collection_name,
            is_vec_index=True,
            index_name="vidx11",
            column_names=["embedding"],
            index_type="GSDISKANN",
            metric_type="l2",
            idx_params="enable_pq=false",
        )

        # insert data
        vector_value1 = [0.748479, 0.276979, 0.555195]
        vector_value2 = [0, 0, 0]
        data1 = [{"id": i, "embedding": vector_value1} for i in range(10)]
        data1.extend([{"id": i, "embedding": vector_value2} for i in range(10, 13)])
        data1.extend([{"id": i, "embedding": vector_value2} for i in range(111, 113)])
        self.client.insert(test_collection_name, data=data1)

        # perform ann search
        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=5,
            output_column_names=["id"],
        )
        self.assertEqual(set(res.fetchall()), set([(112,0.0), (111,0.0), (10,0.0), (11,0.0), (12,0.0)]))

        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=5,
            output_column_names=["id"],
            partition_names=["p0"],
        )
        self.assertEqual(set([r[0] for r in res.fetchall()]), set([12, 11, 10, 2, 4]))

        # Additional tests: using output_columns parameter
        table = Table(test_collection_name, self.client.metadata_obj, autoload_with=self.client.engine)

        # Test output_columns with Column objects
        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=5,
            output_columns=[table.c.id]
        )
        results = res.fetchall()
        self.assertEqual(len(results), 5)
        self.assertEqual(len(results[0]), 2)  # id, distance
        self.assertEqual(set([r[0] for r in results]), set([112, 111, 10, 11, 12]))

        # Test output_columns with SQLAlchemy expressions
        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=5,
            output_columns=[
                table.c.id,
                (table.c.id + 1000).label('id_plus_1000')
            ]
        )
        results = res.fetchall()
        self.assertEqual(len(results), 5)
        self.assertEqual(len(results[0]), 3)
        # Verify calculation results
        for row in results:
            self.assertEqual(row[1], row[0] + 1000)

        # Test parameter priority - output_columns takes precedence over output_column_names
        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=5,
            output_column_names=["id"],
            output_columns=[table.c.id, table.c.meta]
        )

        results = res.fetchall()
        self.assertEqual(len(results), 5)
        self.assertEqual(len(results[0]), 3)

        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0, 0, 0],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=10,
            output_column_names=["id"],
            distance_threshold=0.0,
        )

        results = res.fetchall()
        self.assertEqual(len(results), 5)
        self.assertEqual(len(results[0]), 2)

        # Verify all results have distance = 0.0
        for row in results:
            distance = row[-1]
            self.assertEqual(distance, 0.0)
            self.assertIn(row[0], [112, 111, 10, 11, 12])

        res = self.client.ann_search(
            test_collection_name,
            vec_data=[0.748479, 0.276979, 0.555195],
            vec_column_name="embedding",
            distance_func=l2_distance,
            with_dist=True,
            topk=15,
            output_column_names=["id"],
            distance_threshold=0.5,
        )
        results = res.fetchall()
        # Verify all results have distance <= 0.5
        for row in results:
            distance = row[-1]
            self.assertLessEqual(distance, 0.5)

    def test_delete_get(self):
        test_collection_name = "gs_delete_get_test"
        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", String(64), primary_key=True, autoincrement=False),
            Column("embedding", FLOATVECTOR(3)),
            Column("meta", JSON),
            Column("msg", BLOB),
        ]
        self.client.create_table(
            test_collection_name, columns=cols
        )

        # create vector index
        self.client.create_index(
            test_collection_name,
            is_vec_index=True,
            index_name="vidx12",
            column_names=["embedding"],
            index_type="GSDISKANN",
            metric_type="l2",
            idx_params="enable_pq=false",
        )

        data = [
            {"id": "abc", "embedding": [0.748479, 0.276979, 0.555195], "meta": {"page": 1},
             "msg": b'\x01\x02\x03\x04\xFF\xFF\xFF\xFF\x00\x00\x00\x0C\xAA\xCB\x00\x00'},
            {"id": "bcd", "embedding": [0.748479, 0.276979, 0.555195], "meta": {"page": 2},
             "msg": b'\x02\x03\x04\x05\xFF\xFF\xFF\xFF\x00\x00\x00\x0D\xBB\xDC\x00\x00'},
            {"id": "cde", "embedding": [0, 0, 0], "meta": {"page": 3},
             "msg": b'\x03\x04\x05\x06\xFF\xFF\xFF\xFF\x00\x00\x00\x0E\xCC\xED\x00\x00'},
            {"id": "def", "embedding": [1, 2, 3], "meta": {"page": 4},
             "msg": b'\x04\x05\x06\x07\xFF\xFF\xFF\xFF\x00\x00\x00\x0F\xDD\xFE\x00\x00'}
        ]
        self.client.insert(test_collection_name, data=data)

        self.client.delete(test_collection_name, ids=["bcd", "def"])
        res = self.client.get(
            test_collection_name,
            ids=["abc", "bcd", "cde", "def"],
            where_clause=[text("(meta->>'page')::int > 1")],
            output_column_names=['id']
        )
        self.assertEqual(set(res.fetchall()), set([('cde',)]))

    def test_create_index_dup(self):
        test_collection_name = "gs_create_index_dup_test"
        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", String(64), primary_key=True, autoincrement=False),
            Column("embedding", FLOATVECTOR(3)),
        ]
        self.client.create_table(
            test_collection_name, columns=cols
        )

        # create vector index
        self.client.create_index(
            test_collection_name,
            is_vec_index=True,
            index_name="vidx13",
            column_names=["embedding"],
            index_type="GSDISKANN",
            metric_type="l2",
            idx_params="enable_pq=false",
        )

        self.client.create_index(
            test_collection_name,
            is_vec_index=True,
            index_name="vidx14",
            column_names=["embedding"],
            index_type="GSDISKANN",
            metric_type="l2",
            idx_params="enable_pq=false",
        )

    def test_array_column(self):
        test_collection_name = "gs_array_column_test"
        self.client.drop_table_if_exist(test_collection_name)

        cols = [
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", String(64)),
            Column("arr_c", ARRAY(Text)),
            Column("arr_nested_c", ARRAY(Integer)),
        ]
        self.client.create_table(
            test_collection_name, columns=cols
        )

        data = [
            {"name": "Alice", "arr_c": ["tag1", "tag2"], "arr_nested_c": [[1, 2, 3, 4, 5]]},
            {"name": "Bob", "arr_c": ["tag2", "tag3"], "arr_nested_c": [[6, 7, 8]]},
            {"name": "Charlie", "arr_c": ["tag1"], "arr_nested_c": [[9]]},
        ]
        self.client.insert(test_collection_name, data=data)

        res = self.client.get(
            test_collection_name,
            ids=None,
            where_clause=[text(f"arr_c @> ARRAY['tag1']")],
            output_column_names=["name", "arr_c", "arr_nested_c"]
        )
        for row in res.fetchall():
            name = row[0]
            if name == "Alice":
                self.assertEqual(row[1], ["tag1", "tag2"])
                self.assertEqual(row[2], [[1, 2, 3, 4, 5]])
            elif name == "Charlie":
                self.assertEqual(row[1], ["tag1"])
                self.assertEqual(row[2], [[9]])
            else:
                self.fail("Unexpected row: {}".format(row))

    def test_refresh_metadata(self):
        test_collection_name = "gs_refresh_metadata_test"
        self.client.drop_table_if_exist(test_collection_name)

        self.client.create_table(
            table_name=test_collection_name,
            columns=[
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("name", String(64), nullable=True),
            ]
        )
        self.client.perform_raw_text_sql(f"ALTER TABLE {test_collection_name} ADD COLUMN age INTEGER")

        err_msg = ""
        try:
            self.client.insert(
                table_name=test_collection_name,
                data={
                    "id": 1,
                    "name": "Alice",
                    "age": 20,
                },
            )
        except Exception as e:
            err_msg = str(e)
        self.assertTrue(len(err_msg) > 0)

        self.client.refresh_metadata([test_collection_name])
        self.client.insert(
            table_name=test_collection_name,
            data={
                "id": 1,
                "name": "Alice",
                "age": 20,
            },
        )
        res = self.client.get(
            table_name=test_collection_name,
            ids=None,
            output_column_names=["id", "name", "age"],
        ).fetchall()
        self.assertTrue(len(res) > 0)

    def test_alter_table(self):
        test_collection_name = "gs_alter_table_test"
        self.client.drop_table_if_exist(test_collection_name)

        self.client.create_table(
            table_name=test_collection_name,
            columns=[
                Column("id", Integer, primary_key=True, autoincrement=True),
            ]
        )
        self.client.add_columns(
            table_name=test_collection_name,
            columns=[
                Column("name", String(64), nullable=True),
                Column("age", Integer, nullable=True),
            ]
        )
        self.client.insert(
            table_name=test_collection_name,
            data={
                "id": 1,
                "name": "Alice",
                "age": 20,
            },
        )

        res = self.client.get(
            table_name=test_collection_name,
            ids=[1],
        ).fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 3)

        self.client.drop_columns(
            table_name=test_collection_name,
            column_names=["age"]
        )

        res = self.client.get(
            table_name=test_collection_name,
            ids=[1],
        ).fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 2)

    def test_update(self):
        test_collection_name = "gs_update_test"
        self.client.drop_table_if_exist(test_collection_name)

        self.client.create_table(
            table_name=test_collection_name,
            columns=[
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("embedding", FLOATVECTOR(3)),
                Column("meta", JSON),
            ]
        )

        data = [
            {"id": 112, "embedding": [1, 2, 3], "meta": {'doc': 'hhh1'}},
            {"id": 190, "embedding": [0.13, 0.123, 1.213], "meta": {'doc': 'hhh2'}},
        ]
        self.client.insert(table_name=test_collection_name, data=data)

        where_clause = [and_(text('id=:id').bindparams(id='112'), text('embedding=:embedding').bindparams(embedding='[1, 2, 3]'))]
        self.client.update(
            table_name=test_collection_name,
            values_clause=[{'meta': {'doc': 'HHH'}}],
            where_clause=where_clause
        )

        res = self.client.get(
            table_name=test_collection_name,
            ids=[112],
        ).fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].meta['doc'], 'HHH')


if __name__ == "__main__":
    unittest.main()
