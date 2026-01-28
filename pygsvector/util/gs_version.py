"""GaussDB version module."""
import copy
import re
from typing import List


class GsDBVersion:
    """The class to describe GaussDB version.

    Attributes:
        version_nums (List[int]): version number of GaussDB. For example, '506.0.0' -> [506, 0, 0]
    """
    def __init__(self, version_nums: List[int]):
        self.version_nums = copy.deepcopy(version_nums)

    @classmethod
    def from_db_version_string(cls, version: str):
        """Construct GaussDB Version from the output of GaussDB's `SELECT version();`.

        Args:
            version: Full version string returned by GaussDB, e.g.,
                     "gaussdb (GaussDB Kernel 506.0.0 build b04b08b6) ..."
                     or "gaussdb (GaussVector 102.1.0 build 2976c2d7) ..."

        Returns:
            GaussDB Version instance with parsed version numbers.
            - For "GaussDB Kernel", uses the actual version.
            - For "GaussVector", returns fixed version 505.2.1.

        Raises:
            ValueError: If version string does not contain a recognized GaussDB variant.
        """
        # Try to match GaussDB Kernel first
        kernel_match = re.search(r"GaussDB\s+Kernel\s+(\d+\.\d+\.\d+)", version)
        if kernel_match:
            version_str = kernel_match.group(1)
            version_nums = [int(part) for part in version_str.split(".")]
            return cls(version_nums)

        # Then try GaussVector
        vector_match = re.search(r"GaussVector\s+(\d+\.\d+\.\d+)", version)
        if vector_match:
            # Ignore actual version; use fixed compatible version
            version_nums = [505, 2, 1]
            return cls(version_nums)

        # Neither matched
        raise ValueError(f"Unable to parse GaussDB version from string: {version}")

    @classmethod
    def from_db_version_nums(cls, main_ver: int, sub_ver1: int, sub_ver2: int):
        """Construct GsDBVersion with 3 version numbers (GaussDB uses 3-part versioning).

        Args:
            main_ver: main version (e.g., 506)
            sub_ver1: first subversion (e.g., 0)
            sub_ver2: second subversion (e.g., 0)
        """
        return cls([main_ver, sub_ver1, sub_ver2])

    def __lt__(self, other):
        if len(self.version_nums) != len(other.version_nums):
            raise ValueError("Version number list lengths are not equal")
        for a, b in zip(self.version_nums, other.version_nums):
            if a < b:
                return True
            if a > b:
                return False
        return False

    def __str__(self):
        return ".".join(map(str, self.version_nums))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.version_nums})"