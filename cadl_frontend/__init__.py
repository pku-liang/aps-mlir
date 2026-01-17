"""
CADL Frontend - Python parser for Computer Architecture Description Language

This package provides a Python implementation of the CADL parser,
originally implemented in Rust with LALRPOP.
"""

from .parser import parse_proc
from .ast import *

__version__ = "0.1.0"
__all__ = ["parse_proc"]