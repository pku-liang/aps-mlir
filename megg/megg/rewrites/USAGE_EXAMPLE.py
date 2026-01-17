#!/usr/bin/env python3
"""
Usage example for Megg internal rewrites.

This demonstrates the recommended way to use the mathematical rewrite rules
in the Megg compiler optimization pipeline.
"""

import os
import egglog
from egglog import EGraph

# Import the main interface (recommended)
from megg.rewrites import register_internal_rewrites

# Import individual components (for advanced usage)
from megg.rewrites import basic_math_laws, constant_folding_laws

# Import Term definitions
from megg.egraph.term import Term, LitTerm


def main():
    """Demonstrate basic usage of internal rewrites."""

    # Setup environment
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir

    print("=== Megg Internal Rewrites Usage Example ===\n")

    # Example 1: Simple usage (recommended)
    print("1. Simple usage with register_internal_rewrites():")
    egraph1 = EGraph()
    register_internal_rewrites(egraph1)

    # Create and test expression: x + (-x) should equal 0
    x = egraph1.let("x", Term.arg(egglog.i64(0), egglog.String("i32")))
    neg_x = egraph1.let("neg_x", Term.neg(x, egglog.String("i32")))
    expr = egraph1.let("expr", Term.add(x, neg_x, egglog.String("i32")))
    zero = egraph1.let("zero", Term.lit(LitTerm.int(egglog.i64(0)), egglog.String("i32")))

    # Apply rules and test
    math_laws = basic_math_laws()
    egraph1.run(math_laws.saturate())

    result = egraph1.check_bool(egglog.eq(expr).to(zero))
    print(f"   x + (-x) = 0? {result}")
    print()

    # Example 2: Advanced usage with separate rulesets
    print("2. Advanced usage with separate rulesets:")
    egraph2 = EGraph()

    # Apply basic laws first
    math_laws = basic_math_laws()
    egraph2.run(math_laws.saturate())
    print("   Applied basic mathematical laws")

    # Optionally add constant folding
    # Note: This requires registering egglog functions first
    try:
        from megg.rewrites import (extract_int, add_i64, sub_i64, mul_i64,
                                  extract_float, add_f64, sub_f64, mul_f64)

        egraph2.register(extract_int)
        egraph2.register(extract_float)
        egraph2.register(add_i64)
        egraph2.register(sub_i64)
        egraph2.register(mul_i64)
        egraph2.register(add_f64)
        egraph2.register(sub_f64)
        egraph2.register(mul_f64)

        cf_laws = constant_folding_laws()
        egraph2.run(cf_laws.saturate())
        print("   Applied constant folding laws")
    except Exception as e:
        print(f"   Constant folding setup failed: {e}")
    print()

    # Example 3: Integration with existing code
    print("3. Integration example:")

    def optimize_expression_with_megg(expression_builder_func):
        """Example of how to integrate with existing optimization pipeline."""
        egraph = EGraph()

        # Register mathematical rewrite rules
        register_internal_rewrites(egraph, include_constant_folding=False)

        # Build expression using provided function
        expr = expression_builder_func(egraph)

        # Apply optimizations
        math_laws = basic_math_laws()
        egraph.run(math_laws.saturate())

        return expr, egraph

    def build_test_expression(egraph):
        """Build a test expression: (x * 1) + 0"""
        x = egraph.let("x", Term.arg(egglog.i64(0), egglog.String("i32")))
        one = egraph.let("one", Term.lit(LitTerm.int(egglog.i64(1)), egglog.String("i32")))
        zero = egraph.let("zero", Term.lit(LitTerm.int(egglog.i64(0)), egglog.String("i32")))

        x_times_1 = egraph.let("x_times_1", Term.mul(x, one, egglog.String("i32")))
        result = egraph.let("result", Term.add(x_times_1, zero, egglog.String("i32")))

        return result

    expr, egraph = optimize_expression_with_megg(build_test_expression)

    # Test that (x * 1) + 0 simplifies to x
    x_ref = egraph.let("x_ref", Term.arg(egglog.i64(0), egglog.String("i32")))
    simplified = egraph.check_bool(egglog.eq(expr).to(x_ref))
    print(f"   (x * 1) + 0 simplified to x? {simplified}")
    print()

    print("=== Examples completed successfully! ===")


if __name__ == "__main__":
    main()