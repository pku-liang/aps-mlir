"""
Internal rewrite rules for egglog-based optimization.

This module provides Python-based mathematical rewrite rules for the Megg compiler
optimization pipeline.
"""

try:
    from .external_rewrites import (
        ExternalRewriteEngine,
        ExternalRewriteRegistry,
        apply_external_rewrites
    )
except ImportError:
    # External rewrites may not be available if MLIR is not installed
    ExternalRewriteEngine = None
    ExternalRewriteRegistry = None
    apply_external_rewrites = None

# Import new internal rewrite functions
from .internal_rewrites import (
    basic_math_laws,
    constant_folding_laws,
    register_internal_rewrites,
)
from .match_rewrites import (
    build_ruleset_from_module,
)

# Re-export commonly used functions
__all__ = [
    # External rewrites
    'apply_external_rewrites',
    'ExternalRewriteEngine',
    'ExternalRewriteRegistry',
    # Mathematical rewrite rules (main interface)
    'basic_math_laws',
    'type_annotation_ruleset',
    'constant_folding_laws',
    'register_internal_rewrites',
    # Custom instruction rewrites
    'build_instruction_rewrite',
    'build_ruleset_from_module',
]
