"""vec_dist_func: An extended system function for SQLAlchemy.

The system function to calculate distance between vectors.
"""

import logging

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy import Float

logger = logging.getLogger(__name__)

def parse_vec_distance_func_args(element, compiler, **kwargs):
    args = []
    for arg in element.args:
        if isinstance(arg, str) or isinstance(arg, list):
            args.append(f"'{arg}'")
        else:
            args.append(compiler.process(arg))
    args = ", ".join(args)
    return args

class l2_distance(FunctionElement):
    """Vector distance function: l2_distance.
    
    Attributes:
    type : result type
    """
    type = Float()

    def __init__(self, *args):
        super().__init__()
        self.args = args

@compiles(l2_distance)
def compile_l2_distance(element, compiler, **kwargs): # pylint: disable=unused-argument
    """Compile l2_distance function.

    Args:
        element: l2_distance arguments
        compiler: SQL compiler
    """
    return f"l2_distance({parse_vec_distance_func_args(element, compiler, **kwargs)})"


class cosine_distance(FunctionElement):
    """Vector distance function: cosine_distance.
    
    Attributes:
    type : result type
    """
    type = Float()

    def __init__(self, *args):
        super().__init__()
        self.args = args

@compiles(cosine_distance)
def compile_cosine_distance(element, compiler, **kwargs): # pylint: disable=unused-argument
    """Compile cosine_distance function.

    Args:
        element: cosine_distance arguments
        compiler: SQL compiler
    """
    return f"cosine_distance({parse_vec_distance_func_args(element, compiler, **kwargs)})"


class hamming_bool_distance(FunctionElement):
    """Vector distance function: hamming_bool_distance.
    
    Attributes:
    type : result type
    """
    type = Float()

    def __init__(self, *args):
        super().__init__()
        self.args = args

@compiles(hamming_bool_distance)
def compile_hamming_distance(element, compiler, **kwargs): # pylint: disable=unused-argument
    """Compile hamming_bool_distance function.

    Args:
        element: hamming_bool_distance arguments
        compiler: SQL compiler
    """
    return f"hamming_bool_distance({parse_vec_distance_func_args(element, compiler, **kwargs)})"
