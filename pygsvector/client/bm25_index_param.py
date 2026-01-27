"""A module to specify fts index parameters"""
from enum import Enum
from typing import List, Optional

class BM25IndexParam:
    def __init__(
        self,
        index_name: str,
        field_names: List[str],
        num_parallels: Optional[str] = None,
    ):
        self.index_name = index_name
        self.field_names = field_names
        self.num_parallels = num_parallels

    def param_str(self) -> str:
        if self.num_parallels is None:
            return "16"
        else:
            return self.num_parallels
    def __iter__(self):
        yield "index_name", self.index_name
        yield "field_names", self.field_names
        if self.num_parallels:
            yield "num_parallels", self.num_parallels

    def __str__(self):
        return str(dict(self))

    def __eq__(self, other: None):
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)

        if isinstance(other, dict):
            return dict(self) == other
        return False
