"""
Comprehensive MLIR conversion tests

Tests that CADL constructs correctly parse AND convert to valid MLIR.
Consolidates parser tests with MLIR generation verification.
"""

import pytest
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


def verify_mlir(cadl_source: str, expected_ops: list = None):
    """Helper to parse CADL and verify MLIR generation"""
    ast = parse_proc(cadl_source, "test.cadl")
    mlir_module = convert_cadl_to_mlir(ast)
    assert mlir_module is not None

    mlir_str = str(mlir_module)
    assert "module" in mlir_str or "builtin.module" in mlir_str

    if expected_ops:
        for op in expected_ops:
            assert op in mlir_str, f"Expected '{op}' in MLIR output"

    return mlir_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestBasicMLIRConversion:
    """Test basic CADL constructs convert to MLIR"""

    def test_simple_arithmetic(self):
        """Test basic arithmetic operations"""
        source = """
        rtype add(a: u32, b: u32, c: u32) {
            let result: u32 = a + b;
            _irf[c] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.addi", "aps.writerf"])

    def test_multiple_operations(self):
        """Test multiple arithmetic operations"""
        source = """
        rtype compute(a: u32, b: u32, c: u32, d: u32) {
            let temp1: u32 = a + b;
            let temp2: u32 = temp1 * c;
            let result: u32 = temp2 - a;
            _irf[d] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.addi", "arith.muli", "arith.subi"])

    def test_bitwise_operations(self):
        """Test bitwise operations"""
        source = """
        rtype bitwise(a: u32, b: u32, c: u32) {
            let and_result: u32 = a & b;
            let or_result: u32 = a | b;
            let xor_result: u32 = a ^ b;
            _irf[c] = and_result + or_result + xor_result;
        }
        """
        mlir = verify_mlir(source, ["arith.andi", "arith.ori", "arith.xori"])

    def test_shift_operations(self):
        """Test shift operations"""
        source = """
        rtype shifter(value: u32, amount: u32, rd: u32) {
            let left: u32 = value << amount;
            let right: u32 = value >> amount;
            _irf[rd] = left + right;
        }
        """
        mlir = verify_mlir(source, ["arith.shli", "arith.shrui"])

    def test_comparison_operations(self):
        """Test comparison operations"""
        source = """
        rtype compare(a: u32, b: u32, rd: u32) {
            let eq: u1 = a == b;
            let lt: u1 = a < b;
            let result: u32 = if eq {1} else {0};
            _irf[rd] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.cmpi"])


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestRegisterFileMLIR:
    """Test _irf register file operations"""

    def test_irf_read(self):
        """Test _irf read operations"""
        source = """
        rtype test(rs1: u5, rd: u5) {
            let value: u32 = _irf[rs1];
            _irf[rd] = value;
        }
        """
        verify_mlir(source, ["aps.readrf", "aps.writerf"])

    def test_irf_arithmetic(self):
        """Test _irf with arithmetic"""
        source = """
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """
        verify_mlir(source, ["aps.readrf", "aps.writerf", "arith.addi"])

    def test_irf_complex(self):
        """Test _irf with complex expressions"""
        source = """
        rtype complex(rs1: u5, rs2: u5, rs3: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let r3: u32 = _irf[rs3];
            let result: u32 = (r1 + r2) * r3;
            _irf[rd] = result;
        }
        """
        verify_mlir(source, ["aps.readrf", "aps.writerf", "arith.addi", "arith.muli"])


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMemoryMLIR:
    """Test _mem memory operations"""
    def test_mem_read(self):
        """Test _mem read operations"""
        source = """
        rtype test(addr: u32, rd: u5) {
            let value: u32 = _mem[addr];
            _irf[rd] = value;
        }
        """
        mlir = verify_mlir(source, ["aps.memload"])
        assert "_cpu_memory" in mlir

    def test_mem_write(self):
        """Test _mem write operations"""
        source = """
        rtype test(addr: u32, value: u32) {
            _mem[addr] = value;
        }
        """
        mlir = verify_mlir(source, ["aps.memstore"])
        assert "_cpu_memory" in mlir

    def test_mem_and_irf(self):
        """Test _mem and _irf together"""
        source = """
        rtype test(rs1: u5, addr: u32, rd: u5) {
            let reg_val: u32 = _irf[rs1];
            let mem_val: u32 = _mem[addr];
            let result: u32 = reg_val + mem_val;
            _irf[rd] = result;
        }
        """
        verify_mlir(source, ["aps.readrf", "aps.writerf", "aps.memload", "arith.addi"])

    def test_no_memory_when_unused(self):
        """Test that flows without _mem don't get memory globals"""
        source = """
        rtype test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """
        mlir = verify_mlir(source, ["aps.readrf"])
        # Should NOT have memory globals
        assert "memref.global" not in mlir or "_cpu_memory" not in mlir


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestStaticMLIR:
    """Test static variable declarations"""
    def test_static_simple(self):
        """Test simple static declaration"""
        source = """
        static counter: u32 = 42;

        rtype test(rd: u5) {
            _irf[rd] = counter;
        }
        """
        mlir = verify_mlir(source, ["memref.global"])
        assert "counter" in mlir

    def test_static_array(self):
        """Test static array declaration"""
        source = """
        static buffer: [i32; 1024];

        rtype test() {
            buffer[0] = 42;
        }
        """
        mlir = verify_mlir(source, ["memref.global"])
        assert "buffer" in mlir
        assert "1024" in mlir

    def test_static_with_impl_attribute(self):
        """Test static with impl attribute"""
        source = """
        #[impl("1rw")]
        static buffer: [i32; 1024];

        rtype test() {
            buffer[0] = 42;
        }
        """
        mlir = verify_mlir(source, ["memref.global"])
        assert 'impl = "1rw"' in mlir

    def test_static_with_multiple_attributes(self):
        """Test static with multiple attributes"""
        source = """
        #[impl("2rw")]
        #[partition("cyclic")]
        static scratch: [i32; 512];

        rtype test() {
            scratch[0] = 1;
        }
        """
        mlir = verify_mlir(source, ["memref.global"])
        assert 'impl = "2rw"' in mlir
        assert 'partition = "cyclic"' in mlir


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestControlFlowMLIR:
    """Test control flow constructs"""
    def test_if_expression(self):
        """Test if expression"""
        source = """
        rtype test(a: u32, b: u32, rd: u5) {
            let result: u32 = if (a > b) {a} else {b};
            _irf[rd] = result;
        }
        """
        # If expressions generate arith.select operations
        mlir = verify_mlir(source, ["arith.select", "arith.cmpi"])

    def test_nested_if(self):
        """Test nested if expressions"""
        source = """
        rtype test(a: u32, b: u32, c: u32, rd: u5) {
            let result: u32 = if (a > b) {
                if (a > c) {a} else {c}
            } else {
                if (b > c) {b} else {c}
            };
            _irf[rd] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.select"])
        # Should have multiple select operations for nested ifs
        assert mlir.count("arith.select") >= 2

    def test_select_expression_simple(self):
        """Test simple select expression"""
        source = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 10,
                x > 0: 20,
            };
            _irf[rd] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.select", "arith.cmpi"])
        # Should have select operations for select arms
        assert mlir.count("arith.select") >= 1

    def test_select_expression_multiple_arms(self):
        """Test select with multiple arms"""
        source = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 100,
                x < 10: 200,
                x < 20: 300,
                x >= 20: 400,
            };
            _irf[rd] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.select", "arith.cmpi"])
        # Should have multiple select operations (one per condition)
        assert mlir.count("arith.select") >= 3

    def test_select_with_complex_values(self):
        """Test select with complex expressions as values"""
        source = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: x + 10,
                x == 1: x * 20,
                x > 10: x << 2,
                1: 0,
            };
            _irf[rd] = result;
        }
        """
        mlir = verify_mlir(source, ["arith.select", "arith.addi", "arith.muli", "arith.shli"])
        # Should have arithmetic operations in the select values
        assert "arith.addi" in mlir
        assert "arith.muli" in mlir


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestComplexExamples:
    """Test real-world complex examples"""
    def test_risc_v_add(self):
        """Test RISC-V add instruction"""
        source = """
        #[opcode(7'b0001011)]
        #[funct7(7'b0000000)]
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """
        verify_mlir(source, ["aps.readrf", "aps.writerf", "arith.addi"])

    def test_memory_accumulate(self):
        """Test accumulate with multiple memory reads"""
        source = """
        #[opcode(7'b1011011)]
        #[funct7(7'b0000000)]
        rtype accum(rs1: u5, rd: u5) {
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
        mlir = verify_mlir(source, ["aps.memload", "aps.memstore", "arith.addi"])
        # Should have multiple loads
        assert mlir.count("aps.memload") >= 4

    def test_crc8_simplified(self):
        """Test simplified CRC8 computation"""
        source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8(rs1: u5, rd: u5) {
            let x0: u32 = _irf[rs1];
            let flag: u32 = x0 >> 7;
            let x_shifted: u32 = x0 << 1;
            let x_final: u32 = if (flag != 0) { x_shifted ^ 7 } else { x_shifted };
            _irf[rd] = x_final;
        }
        """
        verify_mlir(source, ["arith.shrui", "arith.shli", "arith.xori", "arith.select"])


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMLIRStructure:
    """Test MLIR output structure and validity"""

    def test_module_structure(self):
        """Test that MLIR module has correct structure"""
        source = """
        rtype test(a: u32, b: u32, rd: u5) {
            _irf[rd] = a + b;
        }
        """

        mlir_str = verify_mlir(source)

        # Check structure
        assert mlir_str.startswith('"builtin.module"') or mlir_str.startswith('module')
        assert mlir_str.count('{') == mlir_str.count('}')
        assert 'func.func' in mlir_str

    def test_ssa_values(self):
        """Test that SSA values are generated"""
        source = """
        rtype complex(a: u32, b: u32, c: u32, rd: u5) {
            let x: u32 = a + b;
            let y: u32 = x * 2;
            let z: u32 = y - a;
            _irf[rd] = z;
        }
        """

        mlir_str = verify_mlir(source)

        # Should have SSA values
        import re
        ssa_values = re.findall(r'%\d+', mlir_str)
        assert len(ssa_values) > 0

    def test_type_consistency(self):
        """Test that types are consistent"""
        source = """
        rtype typed(a: u32, b: u32, c: u32, rd: u5) {
            let x: u32 = a;
            let y: u32 = b;
            _irf[rd] = x + y;
        }
        """

        mlir_str = verify_mlir(source)

        # All integer ops should use i32
        assert "i32" in mlir_str
        assert "i64" not in mlir_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
