"""
Test cases for memory operations

Tests _irf, _mem access, and burst operations with range slice syntax.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestMemoryAccess:
    """Test _mem memory access operations"""

    def test_memory_read_access(self):
        """Test reading from _mem"""
        source = """
        rtype test(addr: u32) {
            let data: u32 = _mem[addr];
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        assert isinstance(assign.rhs, IndexExpr)
        assert isinstance(assign.rhs.expr, IdentExpr)
        assert assign.rhs.expr.name == "_mem"

    def test_memory_write_access(self):
        """Test writing to _mem"""
        source = """
        rtype test(addr: u32, value: u32) {
            _mem[addr] = value;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        assert isinstance(assign.lhs, IndexExpr)
        assert isinstance(assign.lhs.expr, IdentExpr)
        assert assign.lhs.expr.name == "_mem"

    def test_memory_with_address_calculation(self):
        """Test _mem with address calculation"""
        source = """
        rtype test(base: u32, offset: u32) {
            let addr: u32 = base + offset;
            let data: u32 = _mem[addr];
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 2


class TestIRFAccess:
    """Test _irf register file access operations"""

    def test_irf_read_access(self):
        """Test reading from _irf"""
        source = """
        rtype test(rs1: u5) {
            let value: u32 = _irf[rs1];
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        assert isinstance(assign.rhs, IndexExpr)
        assert isinstance(assign.rhs.expr, IdentExpr)
        assert assign.rhs.expr.name == "_irf"

    def test_irf_write_access(self):
        """Test writing to _irf"""
        source = """
        rtype test(rd: u5, value: u32) {
            _irf[rd] = value;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        assert isinstance(assign.lhs, IndexExpr)
        assert isinstance(assign.lhs.expr, IdentExpr)
        assert assign.lhs.expr.name == "_irf"

    def test_irf_with_multiple_registers(self):
        """Test multiple _irf accesses"""
        source = """
        rtype test(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = _irf[rs1];
            let b: u32 = _irf[rs2];
            let c: u32 = a + b;
            _irf[rd] = c;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 4


class TestMixedMemoryOps:
    """Test mixed memory and IRF operations"""

    def test_mixed_mem_irf_operations(self):
        """Test mixed _mem and _irf operations"""
        source = """
        rtype test(rs1: u5, rd: u5) {
            let addr: u32 = _irf[rs1];
            let data: u32 = _mem[addr];
            _irf[rd] = data;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 3

    def test_complex_expression_with_mem_irf(self):
        """Test complex expressions involving _mem and _irf"""
        source = """
        rtype test(rs1: u5, rs2: u5, rd: u5) {
            let base: u32 = _irf[rs1];
            let offset: u32 = _irf[rs2];
            let addr: u32 = base + offset * 4;
            let data: u32 = _mem[addr];
            _irf[rd] = data + 1;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 5


class TestRangeSliceSyntax:
    """Test range slice [start +: length] syntax"""

    def test_range_slice_explicit_length(self):
        """Test range slice with explicit length"""
        source = """
        static buffer: [i32; 1024];
        rtype test() {
            let x: i32 = buffer[0 +: 10];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert isinstance(assign.rhs, RangeSliceExpr)
        assert isinstance(assign.rhs.expr, IdentExpr)
        assert assign.rhs.expr.name == "buffer"
        assert assign.rhs.length is not None

    def test_range_slice_omitted_length(self):
        """Test range slice with omitted length"""
        source = """
        static buffer: [i32; 1024];
        rtype test() {
            let x: i32 = buffer[0 +: ];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert isinstance(assign.rhs, RangeSliceExpr)
        assert assign.rhs.length is None

    def test_range_slice_with_variables(self):
        """Test range slice with variable start and length"""
        source = """
        static buffer: [i32; 1024];
        rtype test(offset: u32, count: u32) {
            let data: i32 = buffer[offset +: count];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert isinstance(assign.rhs, RangeSliceExpr)
        assert isinstance(assign.rhs.start, IdentExpr)
        assert isinstance(assign.rhs.length, IdentExpr)

    def test_range_slice_as_lvalue(self):
        """Test range slice on left side of assignment"""
        source = """
        static buffer: [i32; 1024];
        rtype test(offset: u32) {
            buffer[offset +: 10] = 0;
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert isinstance(assign.lhs, RangeSliceExpr)


class TestBurstOperations:
    """Test burst read/write operations"""

    def test_burst_read_basic(self):
        """Test basic burst read operation"""
        source = """
        static buffer: [i32; 512];
        rtype test(cpu_addr: u64, offset: u32, count: u32) {
            buffer[offset +: ] = __burst_read[cpu_addr +: count];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        # LHS: buffer range slice
        assert isinstance(assign.lhs, RangeSliceExpr)
        assert assign.lhs.expr.name == "buffer"

        # RHS: __burst_read range slice
        assert isinstance(assign.rhs, RangeSliceExpr)
        assert assign.rhs.expr.name == "__burst_read"

    def test_burst_write_basic(self):
        """Test basic burst write operation"""
        source = """
        static buffer: [i32; 512];
        rtype test(cpu_addr: u64, offset: u32, count: u32) {
            __burst_write[cpu_addr +: count] = buffer[offset +: ];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        # LHS: __burst_write range slice
        assert isinstance(assign.lhs, RangeSliceExpr)
        assert assign.lhs.expr.name == "__burst_write"

        # RHS: buffer range slice
        assert isinstance(assign.rhs, RangeSliceExpr)
        assert assign.rhs.expr.name == "buffer"

    def test_burst_bidirectional(self):
        """Test both burst read and write in same flow"""
        source = """
        static buffer: [i32; 512];
        rtype test(cpu_addr: u64, offset: u32, count: u32) {
            buffer[offset +: ] = __burst_read[cpu_addr +: count];
            __burst_write[cpu_addr +: count] = buffer[offset +: ];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assert len(flow.body) == 2

    def test_burst_explicit_length_both_sides(self):
        """Test burst with explicit length on both sides"""
        source = """
        static buffer: [i32; 512];
        rtype test(cpu_addr: u64, offset: u32) {
            buffer[offset +: 64] = __burst_read[cpu_addr +: 64];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert assign.lhs.length is not None
        assert assign.rhs.length is not None

    def test_burst_with_arithmetic_expressions(self):
        """Test burst operations with arithmetic in indices"""
        source = """
        static buffer: [i32; 512];
        rtype test(cpu_addr: u64, base: u32, count: u32) {
            buffer[base * 2 +: count] = __burst_read[cpu_addr + 0x1000 +: count];
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        # Check LHS has arithmetic expression
        assert isinstance(assign.lhs.start, BinaryExpr)

        # Check RHS has arithmetic expression
        assert isinstance(assign.rhs.start, BinaryExpr)

    def test_burst_in_rtype_flow(self):
        """Test that burst operations work in rtype flows"""
        source = """
        static scratch: [i32; 512];
        rtype dma_transfer(cpu_addr: u64, offset: u32, count: u32) {
            scratch[offset +: ] = __burst_read[cpu_addr +: count];
            __burst_write[cpu_addr +: count] = scratch[offset +: ];
        }
        """
        ast = parse_proc(source)

        # Verify it's an rtype
        flow = ast.flows["dma_transfer"]
        assert flow.kind == FlowKind.RTYPE

        # Verify both burst operations present
        assert len(flow.body) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
