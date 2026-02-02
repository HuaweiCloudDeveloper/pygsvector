# pygsvector/client/hybrid_search.py
import json
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from .gs_vec_client import GsVecClient as Client
from ..util import DSLToSQLConverter


class HybridSearch(Client):
    _metric_to_func = {
        "l2": "<->",
        "hamming": "<#>",
        "cosine": "<=>",
    }

    def __init__(
            self,
            uri: str,
            user: str,
            password: str,
            db_name: str,
            bm25_index_name: Optional[str] = None,
            default_knn_weight: float = 0.5,
            default_bm25_weight: float = 0.5,
            primary_key: str = "id",  # 显式支持主键配置
            **kwargs
    ):
        super().__init__(uri, user, password, db_name, **kwargs)
        self.dsl_converter = DSLToSQLConverter(
            default_operator="AND",
            enable_identifier_validation=True,
        )
        self.bm25_index_name = bm25_index_name
        self.default_knn_weight = default_knn_weight
        self.default_bm25_weight = default_bm25_weight
        self.primary_key = primary_key

    def get_sql(self, index: str, body: Dict[str, Any]) -> str:
        if not isinstance(body, dict):
            raise ValueError("Search body must be a dictionary")

        knn = body.get("knn")
        if not knn:
            raise ValueError("Missing 'knn' in search body")

        # 1. 解析 KNN
        vector_field, query_vector, distance_func, _ = self._validate_and_extract_knn(knn)
        vector_str = json.dumps(query_vector, separators=(",", ":"))

        # 2. 权重
        knn_weight, bm25_weight = self._get_weights(body)

        # 3. 共享过滤条件
        shared_filter = self._build_shared_filter(body, knn)

        # 4. BM25 打分表达式
        bm25_score_expr = self._build_bm25_score_expr(body)

        # 5. Hint
        bm25_hint = self._build_bm25_hint(index)

        # 6. Top-K
        topk = max(1, int(body.get("size", 10)))

        # 7. 委托给专用 SQL 构建方法
        return self._build_hybrid_sql(
            index=index,
            primary_key=self.primary_key,
            vector_field=vector_field,
            vector_str=vector_str,
            distance_func=distance_func,
            shared_filter=shared_filter,
            bm25_score_expr=bm25_score_expr,
            bm25_hint=bm25_hint,
            knn_weight=knn_weight,
            bm25_weight=bm25_weight,
            topk=topk,
        )

    def search(self, index: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        sql = self.get_sql(index, body)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            raise RuntimeError(f"Search execution failed: {e}") from e

    def _validate_and_extract_knn(self, knn: Dict[str, Any]) -> Tuple[str, List[float], str, int]:
        """提取并验证 KNN 参数"""
        vector_field = knn.get("field")
        query_vector = knn.get("query_vector")
        if not vector_field or query_vector is None:
            raise ValueError("'knn' must contain 'field' and 'query_vector'")
        if not isinstance(query_vector, (list, tuple)) or not all(isinstance(x, (int, float)) for x in query_vector):
            raise ValueError("'query_vector' must be a list of numbers")

        metric = knn.get("metric", "l2").lower()
        distance_func = self._metric_to_func.get(metric)
        if not distance_func:
            supported = ", ".join(self._metric_to_func.keys())
            raise ValueError(f"Unsupported metric '{metric}'. Supported: {supported}")

        return vector_field, list(query_vector), distance_func, metric

    def _get_weights(self, body: Dict[str, Any]) -> Tuple[float, float]:
        """统一获取混合权重"""
        weights = body.get("hybrid_weights", {})
        knn_w = float(weights.get("knn", self.default_knn_weight))
        bm25_w = float(weights.get("bm25", self.default_bm25_weight))
        if knn_w < 0 or bm25_w < 0:
            raise ValueError("Weights must be non-negative")
        return knn_w, bm25_w

    def _build_shared_filter(self, body: Dict[str, Any], knn: Dict[str, Any]) -> str:
        """构建共享的 WHERE 过滤条件（不含 BM25 打分子句）"""
        filters = []

        # 主 query 中的结构化过滤（不含 match 等打分类）
        if "query" in body:
            expr = self.dsl_converter.convert(body["query"])
            if expr != "1=1":
                filters.append(expr)

        # knn.filter 中的额外过滤
        if "filter" in knn:
            expr = self.dsl_converter.convert(knn["filter"])
            if expr != "1=1":
                filters.append(expr)

        return " AND ".join(f"({f})" for f in filters) if filters else "1=1"

    def _build_bm25_score_expr(self, body: Dict[str, Any]) -> str:
        """构建 BM25 打分表达式"""
        if "query" not in body:
            return "0.0"
        expressions = self.dsl_converter.extract_bm25_score_expressions(body["query"])
        if not expressions:
            return "0.0"
        return " + ".join(expressions) if len(expressions) > 1 else expressions[0]

    def _build_bm25_hint(self, index: str) -> str:
        """生成 BM25 查询的 hint，优先使用用户指定或类配置的索引名"""
        if self.bm25_index_name:
            return f" /*+ indexscan({index} {self.bm25_index_name}) */ "
        else:
            # 可考虑抛出警告或留空，避免硬编码
            return f" /*+ indexscan({index} bm25_idx_content) */ "

    def _build_hybrid_sql(
            self,
            index: str,
            primary_key: str,
            vector_field: str,
            vector_str: str,
            distance_func: str,
            shared_filter: str,
            bm25_score_expr: str,
            bm25_hint: str,
            knn_weight: float,
            bm25_weight: float,
            topk: int,
    ) -> str:
        """构建混合搜索的完整 SQL，纯字符串拼接，无业务逻辑"""
        return f"""
WITH knn_results AS (
    SELECT
        {primary_key},
        {vector_field} {distance_func} '{vector_str}' AS _distance,
        0.0 AS _score
    FROM {index}
    WHERE {shared_filter}
    ORDER BY _distance
    LIMIT {topk}
),
bm25_results AS (
    SELECT {bm25_hint}
        {primary_key},
        0.0 AS _distance,
        ({bm25_score_expr}) AS _score
    FROM {index}
    WHERE {shared_filter}
    ORDER BY _score DESC
    LIMIT {topk}
),
merged AS (
    SELECT
        {primary_key},
        MIN(_distance) AS _distance,
        MAX(_score) AS _score
    FROM (
        SELECT {primary_key}, _distance, _score FROM knn_results
        UNION ALL
        SELECT {primary_key}, _distance, _score FROM bm25_results
    ) AS combined
    GROUP BY {primary_key}
)
SELECT
    t.*,
    m._distance,
    m._score,
    (m._score * {bm25_weight} - m._distance * {knn_weight}) AS hybrid_score
FROM merged m
JOIN {index} t ON m.{primary_key} = t.{primary_key}
ORDER BY hybrid_score DESC
LIMIT {topk};
        """.strip()
