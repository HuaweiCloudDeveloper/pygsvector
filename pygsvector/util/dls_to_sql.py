# pygsvector/utils/dsl_to_sql.py
import re
from typing import Dict, List, Union, Optional


class DSLToSQLConverter:
    """
    将 Elasticsearch-style DSL 查询转换为兼容 GaussDB 的 SQL WHERE 子句。

    支持：
      - bool (must, should, must_not, filter)
      - term / terms
      - range
      - match / match_phrase
      - query_string
      - exists / missing

    不支持（可扩展）：
      - nested, geo, wildcard, fuzzy, script 等高级查询
    """

    def __init__(
            self,
            default_operator: str = "AND",
            enable_identifier_validation: bool = True,
    ):
        if default_operator not in ("AND", "OR"):
            raise ValueError("default_operator must be 'AND' or 'OR'")

        self.default_operator = default_operator
        self.enable_identifier_validation = enable_identifier_validation

    def convert(self, dsl_query: Union[Dict, List]) -> str:
        """
        入口方法：将顶层 DSL 转换为 SQL WHERE 表达式（不含 WHERE 关键字）
        """
        if not dsl_query:
            return "1=1"
        expr = self._convert_clause(dsl_query)
        return expr if expr != "1=1" else "1=1"

    def _convert_clause(self, clause: Union[Dict, List]) -> str:
        if isinstance(clause, list):
            if not clause:
                return "1=1"
            parts = [self._convert_clause(c) for c in clause]
            if len(parts) == 1:
                return parts[0]
            return f"({' AND '.join(parts)})"

        if not isinstance(clause, dict):
            raise ValueError(f"Invalid clause type: {type(clause)}")

        # 分发到具体处理器
        handlers = {
            "bool": self._handle_bool,
            "term": self._handle_term,
            "terms": self._handle_terms,
            "range": self._handle_range,
            "match": lambda x: self._handle_match(x, mode="natural"),
            "match_phrase": lambda x: self._handle_match(x, mode="phrase"),
            "query_string": self._handle_query_string,
            "exists": self._handle_exists,
            "missing": self._handle_missing,
        }

        for key, handler in handlers.items():
            if key in clause:
                return handler(clause[key])

        raise NotImplementedError(f"Unsupported query type: {list(clause.keys())}")

    def _quote_identifier(self, name: str) -> str:
        if self.enable_identifier_validation:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                raise ValueError(f"Invalid SQL identifier: '{name}'")
        return f"{name}"

    from typing import Any

    def _escape_literal(self, value: Any) -> str:
        if isinstance(value, str):
            if any(c in value for c in ('\0', '\n', '\r', '\x1a')):
                raise ValueError("String contains invalid control characters")
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return '1' if value else '0'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            raise TypeError(f"Unsupported literal type: {type(value)}")

    # --- 处理器方法 ---

    def _handle_bool(self, bool_clause: Dict) -> str:
        parts = []

        # must & filter → AND
        for key in ("must", "filter"):
            clauses = bool_clause.get(key, [])
            if isinstance(clauses, dict):
                clauses = [clauses]
            if clauses:
                sub_exprs = [self._convert_clause([c]) for c in clauses]
                combined = " AND ".join(sub_exprs)
                parts.append(f"({combined})")

        # should → OR
        should_clauses = bool_clause.get("should", [])
        if isinstance(should_clauses, dict):
            should_clauses = [should_clauses]
        if should_clauses:
            sub_exprs = [self._convert_clause([c]) for c in should_clauses]
            combined = " OR ".join(sub_exprs)
            min_should = bool_clause.get("minimum_should_match", 1)
            if min_should != 1:
                raise NotImplementedError("minimum_should_match != 1 is not supported")
            parts.append(f"({combined})")

        # must_not → NOT (...)
        must_not_clauses = bool_clause.get("must_not", [])
        if isinstance(must_not_clauses, dict):
            must_not_clauses = [must_not_clauses]
        if must_not_clauses:
            neg_parts = []
            for c in must_not_clauses:
                sub = self._convert_clause([c])
                neg_parts.append(f"NOT ({sub})")
            parts.append(f"({' AND '.join(neg_parts)})")

        if not parts:
            return "1=1"
        return " AND ".join(parts)

    def _handle_term(self, term_clause: Dict) -> str:
        field, value = next(iter(term_clause.items()))
        return f"{self._quote_identifier(field)} = {self._escape_literal(value)}"

    def _handle_terms(self, terms_clause: Dict) -> str:
        field, values = next(iter(terms_clause.items()))
        if not values:
            return "1=0"
        vals = ",".join(self._escape_literal(v) for v in values)
        return f"{self._quote_identifier(field)} IN ({vals})"

    def _handle_range(self, range_clause: Dict) -> str:
        field, cond = next(iter(range_clause.items()))
        ops = []
        for op_key, sql_op in [("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")]:
            if op_key in cond:
                ops.append(f"{self._quote_identifier(field)} {sql_op} {self._escape_literal(cond[op_key])}")
        return " AND ".join(ops) if ops else "1=1"

    def _handle_match(self, match_clause: Dict, mode: str) -> str:
        return "1=1"

    def _handle_query_string(self, qs_clause: Dict) -> str:
        return "1=1"

    def _handle_exists(self, exists_clause: Dict) -> str:
        field = exists_clause["field"]
        return f"{self._quote_identifier(field)} IS NOT NULL"

    def _handle_missing(self, missing_clause: Dict) -> str:
        field = missing_clause["field"]
        return f"{self._quote_identifier(field)} IS NULL"

    def extract_bm25_score_expressions(self, query: Union[Dict, List]) -> List[str]:
        """
        从 DSL 查询中提取所有 match / match_phrase / query_string 子句，
        并转换为 GaussDB 的 BM25 打分表达式（如 "title ### 'AI'"）。
        返回表达式字符串列表。
        """
        score_exprs = []

        def _walk(node):
            if isinstance(node, list):
                for item in node:
                    _walk(item)
            elif isinstance(node, dict):
                if "bool" in node:
                    b = node["bool"]
                    # 递归遍历所有子句（must, should, filter 等都可能包含 match）
                    for key in ["must", "should", "filter", "must_not"]:
                        clauses = b.get(key, [])
                        if isinstance(clauses, dict):
                            clauses = [clauses]
                        for c in clauses:
                            _walk(c)
                elif "match" in node:
                    expr = self._handle_match_bm25(node["match"])
                    if expr:
                        score_exprs.append(expr)
                elif "match_phrase" in node:
                    expr = self._handle_match_phrase_bm25(node["match_phrase"])
                    if expr:
                        score_exprs.append(expr)
                elif "query_string" in node:
                    expr = self._handle_query_string_bm25(node["query_string"])
                    if expr:
                        score_exprs.append(expr)
                # 其他类型（term/range等）不产生打分，跳过

        _walk(query)
        return score_exprs

    def _handle_match_bm25(self, match_clause: Dict) -> Optional[str]:
        field, query_text = next(iter(match_clause.items()))
        if isinstance(query_text, dict):
            query_text = query_text.get("query", "")
        query_text = str(query_text).strip()
        if not query_text:
            return None
        field_id = self._quote_identifier(field)
        return f"{field_id} ### {self._escape_literal(query_text)}"

    def _handle_match_phrase_bm25(self, phrase_clause: Dict) -> Optional[str]:
        # 在 BM25 中，phrase 通常也用相同操作符，GaussDB 可能内部处理短语匹配
        # 如果需显式加引号，可包装为 '"query"'，但根据你给的例子，直接传字符串即可
        field, query_text = next(iter(phrase_clause.items()))
        if isinstance(query_text, dict):
            query_text = query_text.get("query", "")
        query_text = str(query_text).strip()
        if not query_text:
            return None
        field_id = self._quote_identifier(field)
        # 可选：是否加引号？根据 GaussDB 行为决定。示例中未加，故不加。
        return f"{field_id} ### {self._escape_literal(query_text)}"

    def _handle_query_string_bm25(self, qs_clause: Dict) -> Optional[str]:
        query = qs_clause.get("query", "").strip()
        fields = qs_clause.get("fields", [])
        if not query or not fields:
            return None

        clean_fields = [f.split("^")[0] for f in fields]  # 忽略 boost (^2)
        if len(clean_fields) == 1:
            field_id = self._quote_identifier(clean_fields[0])
            return f"{field_id} ### {self._escape_literal(query)}"
        else:
            # 多字段：GaussDB 是否支持 (col1 || ' ' || col2) ### 'query'？
            # 若不支持，则取第一个字段，或报错/警告
            # 这里保守处理：只取第一个字段（或抛异常）
            raise NotImplementedError(
                "GaussDB BM25 with multiple fields in query_string is not supported yet. "
                "Please specify a single field."
            )
