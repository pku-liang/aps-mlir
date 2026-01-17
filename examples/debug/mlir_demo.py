#!/usr/bin/env python3
"""
CADL to MLIR Conversion Demo

This script demonstrates the CADL AST to MLIR converter framework
by converting a simple CADL processor to MLIR IR.
"""

import sys
import os

# Add cadl_frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cadl_frontend.parser import parse_proc
from cadl_frontend.mlir_converter import convert_cadl_to_mlir


def demo_simple_conversion():
    """Demonstrate conversion of simple CADL code to MLIR"""

    # Simple CADL source code
    cadl_source = """
    // Simple function for demonstration
    fn add_numbers(a: u32, b: u32) -> (u32) {
        return (a + b);
    }

    // Simple flow
    flow multiply(x: u32, y: u32) {
        let result: u32 = x * y;
        return (result);
    }

    // Static variable
    static counter: u32 = 42;
    """

    print("CADL Source:")
    print("=" * 50)
    print(cadl_source)
    print()

    try:
        # Parse CADL to AST
        print("Parsing CADL to AST...")
        ast = parse_proc(cadl_source, "demo.cadl")

        print("AST Structure:")
        print("-" * 30)
        print(ast.pretty_print())
        print()

        # Convert AST to MLIR
        print("Converting AST to MLIR...")
        mlir_module = convert_cadl_to_mlir(ast)

        print("MLIR Output:")
        print("-" * 30)
        print(mlir_module)
        print()

        print("✓ Conversion completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ MLIR import error: {e}")
        print("Note: MLIR Python bindings may not be available in current environment")
        print("Try running with: pixi run python examples/mlir_demo.py")
        return False

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_framework_features():
    """Demonstrate key framework features"""

    print("\nFramework Features Demonstration:")
    print("=" * 50)

    features = [
        "✓ CADL AST to MLIR Module conversion",
        "✓ Function definitions → func.func operations",
        "✓ Flow definitions → function-based representations",
        "✓ SSA form generation with symbol table management",
        "✓ Type mapping: CADL types → MLIR types (with fallbacks)",
        "✓ Expression conversion: literals, identifiers, binary/unary ops",
        "✓ Binary operations → arith/comb dialect operations",
        "✓ Control flow: do-while → scf.while, for → scf.while",
        "✓ Function calls → func.call operations",
        "✓ Scoped symbol table for variable bindings",
        "✓ Integration with CIRCT dialects for hardware operations"
    ]

    for feature in features:
        print(f"  {feature}")

    print("\nType Mapping Examples:")
    print("-" * 20)
    print("  CADL u32     → MLIR i32 (signless)")
    print("  CADL i32     → MLIR i32 (signed)")
    print("  CADL f32     → MLIR f32")
    print("  CADL [u8; 4] → MLIR memref<4xi8>")

    print("\nOperation Mapping Examples:")
    print("-" * 25)
    print("  CADL a + b   → arith.addi")
    print("  CADL a & b   → comb.and (bitwise)")
    print("  CADL a && b  → arith.andi (logical)")
    print("  CADL a << b  → comb.shl")
    print("  CADL -a      → arith.subi (0, a)")


if __name__ == "__main__":
    print("CADL to MLIR Converter Framework Demo")
    print("=" * 60)

    success = demo_simple_conversion()
    demo_framework_features()

    if success:
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps for development:")
        print("  1. Test with more complex CADL examples")
        print("  2. Enhance type system for ApFixed/ApUFixed")
        print("  3. Implement proper global variable handling")
        print("  4. Add support for register files")
        print("  5. Optimize control flow generation")
        print("  6. Add MLIR verification and optimization passes")
    else:
        print("\n⚠️  Demo encountered issues - see output above")
        sys.exit(1)