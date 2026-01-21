# Rewrites/__init__.py
"""
Rewrites module - central hub for all rewrite operations
"""

# Core modules
from .EGraph import egraph
from .MatrixSort import Matrix
#from .Rewrites import Rewrites
from .VectorSort import Vector

# Unary operations
from .Unary import *
# This imports everything defined in Unary/__init__.py

# Binary operations  
from .Binary import *
# This imports everything defined in Binary/__init__.py

# If you want to be explicit about what's exported:
__all__ = [
    'EGraph',
    'MatrixSort',
    'Rewrites',
    'VectorSort',
    # Add other top-level names you want to export
]