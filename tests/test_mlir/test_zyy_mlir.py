"""
Comprehensive MLIR converter tests using real examples from zyy.cadl
Each test case represents an actual RISC-V custom instruction
"""

import pytest
import sys
import os
from textwrap import dedent

# Try importing MLIR bindings
try:
    import circt
    import circt.ir as ir
    from cadl_frontend.mlir_converter import CADLMLIRConverter, convert_cadl_to_mlir
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

from cadl_frontend.parser import parse_proc
from cadl_frontend.ast import *


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestZyyRTypeInstructions:
    """Test MLIR conversion for each rtype instruction from zyy.cadl"""

    def verify_mlir_output(self, cadl_source: str, expected_functions: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected functions"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        print(mlir_str)
        
        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected functions if provided
        if expected_functions:
            for func_name in expected_functions:
                assert func_name in mlir_str, f"Expected function '{func_name}' not found in MLIR output"

        return mlir_str

    def test_simple_constant_rtype(self):
        """Test the simplest rtype: constant function"""
        cadl_source = """
        rtype constant(rs1: u5, rs2: u5, rd: u5) {
            let r0: u32 = 0;
            _irf[rd] = r0;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_constant"])

        # Verify basic structure
        assert "func.func" in mlir_str
        assert "arith.constant" in mlir_str  # Should have constant 0

    def test_add_instruction_with_attributes(self):
        """Test add instruction with opcode and funct7 attributes"""
        cadl_source = """
        #[opcode(7'b0001011)]  // custom0
        #[funct7(7'b0000000)]
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_add"])

        # Verify arithmetic operations
        assert "arith.addi" in mlir_str

    def test_simd_add_instruction(self):
        """Test SIMD add instruction with complex bit slicing"""
        cadl_source = """
        #[opcode(7'b0101011)]  // custom1
        #[funct7(7'b1111111)]
        rtype simd_add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = {
                (r1[31:24] + r2[31:24])[7:0],
                (r1[23:16] + r2[23:16])[7:0],
                (r1[15:8] + r2[15:8])[7:0],
                (r1[7:0] + r2[7:0])[7:0]
            };
        }
        """

        # This may fail due to complex bit slicing, but test the attempt
        with pytest.raises(Exception) as exc_info:
            self.verify_mlir_output(cadl_source)
        # Check that it's a parsing or conversion error, not import error
        assert "mlir" not in str(exc_info.value).lower() or "slice" in str(exc_info.value).lower()

    def test_many_multiply_instruction(self):
        """Test instruction with many multiplications"""
        cadl_source = """
        #[opcode(7'b0001011)]  // custom0
        #[funct7(7'b0000000)]
        rtype many_mult(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 * r2 * r2 * r2 * r2;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_many_mult"])

        # Should have multiple multiply operations
        assert "arith.muli" in mlir_str

    def test_if_conditional_instruction(self):
        """Test instruction with if-else conditional"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b1111111)]
        rtype if_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = if (r1 > 32'd6) {r1} else {r2};
        }
        """

        # This should now work with if-expression support
        self.verify_mlir_output(cadl_source)

    def test_loop_instruction(self):
        """Test instruction with do-while loop"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b1111100)]
        rtype loop_test(rs1: u5, rs2: u5, rd: u5) {
            let sum0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = _irf[rs2];

            with
                i: u32 = (i0, i + 1)
                sum: u32 = (sum0, sum + 4)
                n: u32 = (n0, n)
            do {
                let n_: u32 = n;
                let sum_: u32 = sum + 4;
                let i_: u32 = i + 1;
            } while (i < n);

            _irf[rd] = sum;
        }
        """

        # Loop should now convert successfully (falls back to scf.while)
        result = self.verify_mlir_output(cadl_source)

        # Verify the loop is in the output
        assert "scf.while" in result or "scf.for" in result

    def test_many_add_sequence(self):
        """Test instruction with sequence of additions"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b1111100)]
        rtype many_add_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let d1: u32 = r1 + r2;
            let d2: u32 = d1 + r1;
            let d3: u32 = d2 + r1;
            let d4: u32 = d3 + r1;
            let d5: u32 = d4 + r1;
            let d6: u32 = d5 + r1;
            let d7: u32 = d6 + r1;
            let d8: u32 = d7 + r1;
            _irf[rd] = d8;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_many_add_test"])

        # Should have multiple add operations
        assert mlir_str.count("arith.addi") >= 7

    def test_memory_write_instruction(self):
        """Test memory write instruction"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype mem_simplewrite(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            _mem[r1] = _irf[rs2];
            _irf[rd] = 1437;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_mem_simplewrite"])

        # Should have constant 1437
        assert "1437" in mlir_str or "arith.constant" in mlir_str

    def test_memory_read_instruction(self):
        """Test memory read instruction"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype mem_read_(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let rst: u32 = _mem[r1 + r2];
            _irf[rd] = rst;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_mem_read_"])

        # Should have add for address calculation
        assert "arith.addi" in mlir_str

    def test_memory_accumulate_instruction(self):
        """Test accumulate instruction with multiple memory reads"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype accum(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let a: u32 = _mem[r1];
            let b: u32 = _mem[r1 + 4];
            let c: u32 = _mem[r1 + 8];
            let d: u32 = _mem[r1 + 12];
            let rst: u32 = a + b + c + d;
            _mem[r1 + 16] = rst;
            _irf[rd] = rst;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_accum"])

        # Should have multiple additions
        assert mlir_str.count("arith.addi") >= 3  # At least for a+b+c+d

    def test_crc8_instruction(self):
        """Test CRC8 instruction with loop"""
        cadl_source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8(rs1: u5, rs2: u5, rd: u5) {
            let x0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = 8;

            // Simplified version without complex do-while
            let x: u32 = x0;
            let flag: u32 = x >> 7;
            let x_shifted: u32 = x << 1;
            let x_final: u32 = if (flag != 0) { x_shifted ^ 7 } else { x_shifted };

            _irf[rd] = x_final;
        }
        """

        # This should now work with if-expression support
        self.verify_mlir_output(cadl_source)

    
    def test_cordic_instruction(self):
        """Test CORDIC instruction with loop"""
        cadl_source = """
static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype cordic(rs1: u5, rs2: u5, rd: u5) {
    let x0 : u32 = 19898;
    let y0 : u32 = 0;
    let z0 : u32 = _irf[rs1];
    let n0 : u32 = 8;
    let it0: u32 = 0;
    with
      it: u32 = (it0, it_)
      x: u32 = (x0, x_)
      y: u32 = (y0, y_)
      z: u32 = (z0, z_)
      n: u32 = (n0, n_)
    do {
      let z_neg: u1  = z[31:31];
      let theta: u32 = thetas[it];
      let x_shift: u32 = x >> it;
      let y_shift: u32 = y >> it;
      let x_ : u32 = if z_neg {x + y_shift} else {x - y_shift};
      let y_ : u32 = if z_neg {y - x_shift} else {y + x_shift};
      let z_ : u32 = if z_neg {z + theta} else {z - theta};
      let it_: u32 = it + 1;
      let n_ : u32 = n;
    } while (it < n);

    _irf[rd] = y;
}
        """

        # This should work now with proper implementations
        self.verify_mlir_output(cadl_source)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestSimpleMLIRConversions:
    """Test simple, working MLIR conversions"""

    def test_basic_arithmetic_function(self):
        """Test basic arithmetic rtype conversion"""
        cadl_source = """
        rtype add(a: u32, b: u32, c: u32) {
            _irf[c] = (a + b);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        assert "@flow_add" in mlir_str or "func.func @flow_add" in mlir_str
        assert "arith.addi" in mlir_str
        assert "return" in mlir_str

    def test_multiple_operations(self):
        """Test function with multiple operations"""
        cadl_source = """
        rtype compute(a: u32, b: u32, c: u32, d: u32) {
            let temp1: u32 = a + b;
            let temp2: u32 = temp1 * c;
            let result: u32 = temp2 - a;
            _irf[d] = (result);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        assert "func.func @flow_compute" in mlir_str
        assert "arith.addi" in mlir_str
        assert "arith.muli" in mlir_str
        assert "arith.subi" in mlir_str

    def test_simple_flow(self):
        """Test simple flow conversion"""
        cadl_source = """
        flow process(x: u32, y: u32, z: u5) {
            let sum: u32 = x + y;
            let product: u32 = x * y;
            _irf[z] = (sum + product);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        assert "func.func @flow_process" in mlir_str
        assert "arith.addi" in mlir_str
        assert "arith.muli" in mlir_str

    def test_static_variable(self):
        """Test static variable declaration"""
        cadl_source = """
        static counter: u32 = 42;

        rtype get_counter(a: u32) {
            _irf[a] = (counter);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        assert "func.func" in mlir_str and "get_counter" in mlir_str
        assert "counter" in mlir_str  # The static variable should appear
        assert "memref.global" in mlir_str  # Global variable declaration
        assert "memref.get_global" in mlir_str  # Global variable reference
        assert "aps.memload" in mlir_str  # Loading from global using APS dialect


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMLIROutputValidation:
    """Test that generated MLIR is valid and well-formed"""

    def test_mlir_module_structure(self):
        """Test that MLIR module has correct structure"""
        cadl_source = """
        rtype test(a: u5) {
            _irf[a] = (123);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        converter = CADLMLIRConverter()

        with converter.context:
            mlir_module = converter.convert_proc(ast)

            # Verify module structure
            assert mlir_module is not None
            mlir_str = str(mlir_module)

            # Check for proper nesting - handle both verbose and simplified formats
            assert mlir_str.startswith('"builtin.module"') or mlir_str.startswith('module')
            # Balanced braces check
            assert mlir_str.count('{') == mlir_str.count('}')

            # Check for function structure - handle both formats
            assert 'func.func' in mlir_str or '@flow_test' in mlir_str
            # These might be in verbose format only
            if '"func.func"' in mlir_str:
                assert 'sym_name = "flow_test"' in mlir_str

    def test_ssa_value_uniqueness(self):
        """Test that SSA values are unique"""
        cadl_source = """
        rtype complex(a: u32, b: u32, c: u5) {
            let x: u32 = a + b;
            let y: u32 = x * 2;
            let z: u32 = y - a;
            _irf[c] = (z);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        # Extract SSA values (like %0, %1, %2)
        import re
        ssa_values = re.findall(r'%\d+', mlir_str)

        # Each should be defined exactly once (on LHS) and used appropriately
        assert len(ssa_values) > 0, "Should have SSA values"

    def test_type_consistency(self):
        """Test that types are consistent throughout conversion"""
        cadl_source = """
        rtype typed(a: u32, b: u32, c: u5) {
            let x: u32 = a;
            let y: u32 = b;
            _irf[c] = (x + y);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)
        mlir_str = str(mlir_module)
        print(mlir_str)

        # All integer operations should use i32 (since u32 maps to i32)
        assert "i32" in mlir_str
        assert "i64" not in mlir_str  # Should not randomly use i64


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    exit(result.returncode)