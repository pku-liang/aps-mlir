#!/usr/bin/env python3
"""
Integration tests using real-world CADL examples

These tests validate the parser against complex, realistic CADL code patterns
combining multiple language features.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestIntegration:
    """Integration tests with realistic CADL examples"""

    def test_simple_add_rtype(self):
        """Test basic add operation with attributes and _irf access"""
        source = """
        #[opcode(7'b0001011)]
        #[funct7(7'b0000000)]
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "add"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 3
        
        # Check attributes in detail
        assert flow.attrs.get("opcode") is not None
        assert flow.attrs.get("funct7") is not None

        # Validate opcode attribute value: 7'b0001011 (binary 11)
        opcode_lit = flow.attrs.get("opcode")
        assert isinstance(opcode_lit, LitExpr)
        assert opcode_lit.literal.ty.width == 7  # 7-bit binary
        assert opcode_lit.literal.lit.value == int('0001011', 2)  # Binary to decimal: 11

        # Validate funct7 attribute value: 7'b0000000 (binary 0)
        funct7_lit = flow.attrs.get("funct7")
        assert isinstance(funct7_lit, LitExpr)
        assert funct7_lit.literal.ty.width == 7
        assert funct7_lit.literal.lit.value == 0  # All zeros
        
        # Verify input parameter types and names
        assert flow.inputs[0] == ("rs1", DataType_Single(BasicType_ApUFixed(5)))
        assert flow.inputs[1] == ("rs2", DataType_Single(BasicType_ApUFixed(5)))  
        assert flow.inputs[2] == ("rd", DataType_Single(BasicType_ApUFixed(5)))
        
        # Check body structure: 2 let assignments + 1 irf write
        assert len(flow.body) == 3
        
        # Validate _irf reads and write in detail
        read1 = flow.body[0]  # let r1: u32 = _irf[rs1];
        read2 = flow.body[1]  # let r2: u32 = _irf[rs2];
        write = flow.body[2]  # _irf[rd] = r1 + r2;
        
        # Validate first read statement structure
        assert isinstance(read1, AssignStmt) and read1.is_let
        assert read1.lhs.name == "r1"
        assert read1.type_annotation == DataType_Single(BasicType_ApUFixed(32))
        assert isinstance(read1.rhs, IndexExpr)
        assert read1.rhs.expr.name == "_irf"
        assert len(read1.rhs.indices) == 1
        assert isinstance(read1.rhs.indices[0], IdentExpr)
        assert read1.rhs.indices[0].name == "rs1"
        
        # Validate second read statement structure  
        assert isinstance(read2, AssignStmt) and read2.is_let
        assert read2.lhs.name == "r2"
        assert read2.type_annotation == DataType_Single(BasicType_ApUFixed(32))
        assert isinstance(read2.rhs, IndexExpr)
        assert read2.rhs.expr.name == "_irf"
        assert read2.rhs.indices[0].name == "rs2"
        
        # Validate write statement structure
        assert isinstance(write, AssignStmt) and not write.is_let
        assert isinstance(write.lhs, IndexExpr)
        assert write.lhs.expr.name == "_irf"
        assert write.lhs.indices[0].name == "rd"
        
        # Validate addition operation: r1 + r2
        assert isinstance(write.rhs, BinaryExpr)
        assert write.rhs.op == BinaryOp.ADD
        assert isinstance(write.rhs.left, IdentExpr)
        assert write.rhs.left.name == "r1"
        assert isinstance(write.rhs.right, IdentExpr) 
        assert write.rhs.right.name == "r2"

    def test_if_expression_with_literals(self):
        """Test if expressions with width-specified decimal literals"""
        source = """
        rtype if_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = if r1 > 32'd6 {r1} else {r2};
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "if_test"
        assert flow.kind == FlowKind.RTYPE
        
        # Validate input parameters
        assert len(flow.inputs) == 3
        for i, (name, ty) in enumerate(flow.inputs):
            expected_names = ["rs1", "rs2", "rd"]
            assert name == expected_names[i]
            assert isinstance(ty, DataType_Single)
            assert isinstance(ty.basic_type, BasicType_ApUFixed)
            assert ty.basic_type.width == 5
        
        # Should have 2 reads + 1 write with if expression
        assert len(flow.body) == 3
        
        # Validate first two statements are _irf reads
        read1, read2 = flow.body[0], flow.body[1]
        assert isinstance(read1, AssignStmt) and read1.is_let
        assert isinstance(read2, AssignStmt) and read2.is_let
        assert read1.lhs.name == "r1" and read2.lhs.name == "r2"
        
        # Validate the if expression assignment
        write_stmt = flow.body[2]
        assert isinstance(write_stmt, AssignStmt) and not write_stmt.is_let
        assert isinstance(write_stmt.lhs, IndexExpr)
        assert write_stmt.lhs.expr.name == "_irf"
        assert write_stmt.lhs.indices[0].name == "rd"
        
        # Deep validation of if expression structure
        if_expr = write_stmt.rhs
        assert isinstance(if_expr, IfExpr)
        
        # Check condition: r1 > 32'd6
        condition = if_expr.condition
        assert isinstance(condition, BinaryExpr)
        assert condition.op == BinaryOp.GT
        
        # Left side should be identifier "r1"
        assert isinstance(condition.left, IdentExpr)
        assert condition.left.name == "r1"
        
        # Right operand should be width-specified literal 32'd6
        assert isinstance(condition.right, LitExpr)
        literal = condition.right.literal
        assert literal.lit.value == 6
        assert literal.ty.width == 32  # 32'd6 should have 32-bit width
        assert isinstance(literal.ty, BasicType_ApUFixed)
        
        # Check then branch: should be "r1"
        assert isinstance(if_expr.then_branch, IdentExpr)
        assert if_expr.then_branch.name == "r1"
        
        # Check else branch: should be "r2"
        assert isinstance(if_expr.else_branch, IdentExpr)
        assert if_expr.else_branch.name == "r2"

    def test_memory_operations(self):
        """Test memory read and write operations"""
        source = """
        rtype mem_write(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let a: u32 = r1 + r2;
            _mem[r1] = a;
            _irf[rd] = a + r2;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have: 2 irf reads + 1 calculation + 1 mem write + 1 irf write
        assert len(flow.body) == 5
        
        mem_write = flow.body[3]
        irf_write = flow.body[4]
        
        # Check memory write: _mem[r1] = a
        assert isinstance(mem_write, AssignStmt)
        assert not mem_write.is_let
        assert isinstance(mem_write.lhs, IndexExpr)
        assert mem_write.lhs.expr.name == "_mem"
        
        # Check final irf write has addition: a + r2
        assert isinstance(irf_write.rhs, BinaryExpr)
        assert irf_write.rhs.op == BinaryOp.ADD
        assert isinstance(irf_write.rhs.left, IdentExpr)
        assert irf_write.rhs.left.name == "a"
        assert isinstance(irf_write.rhs.right, IdentExpr)
        assert irf_write.rhs.right.name == "r2"
        
        # Validate all variable types are consistent
        for stmt in flow.body[:3]:  # First 3 are let statements
            assert stmt.is_let
            assert stmt.type_annotation == DataType_Single(BasicType_ApUFixed(32))

    def test_accumulator_multiple_memory_reads(self):
        """Test multiple sequential memory accesses with address calculation"""
        source = """
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
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have: 1 irf read + 4 mem reads + 1 sum calc + 1 mem write + 1 irf write
        assert len(flow.body) == 8
        
        # Check memory reads with address calculation
        mem_reads = [flow.body[1], flow.body[2], flow.body[3], flow.body[4]]
        offsets = [0, 4, 8, 12]  # Expected offsets
        
        for i, mem_read in enumerate(mem_reads):
            assert isinstance(mem_read.rhs, IndexExpr)
            assert mem_read.rhs.expr.name == "_mem"
            
            if i == 0:  # First read is just r1
                assert isinstance(mem_read.rhs.indices[0], IdentExpr)
            else:  # Others are r1 + offset
                index_expr = mem_read.rhs.indices[0]
                assert isinstance(index_expr, BinaryExpr)
                assert index_expr.op == BinaryOp.ADD
                assert isinstance(index_expr.right, LitExpr)
                assert index_expr.right.literal.lit.value == offsets[i]
        
        # Check final sum: a + b + c + d
        sum_calc = flow.body[5]
        assert isinstance(sum_calc.rhs, BinaryExpr)
        # Should be nested additions: ((a + b) + c) + d
        assert sum_calc.rhs.op == BinaryOp.ADD

    def test_static_variable_declarations(self):
        """Test static variable declarations"""
        source = """
        static addr: u32 = 0;
        static st: u32 = 0;
        
        rtype state_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            addr = addr + 4;
            st = st + 1;
            _irf[rd] = r1;
        }
        """
        
        ast = parse_proc(source)
        assert len(ast.statics) == 2
        assert len(ast.flows) == 1
        
        # Check static declarations
        addr_static = ast.statics["addr"]
        st_static = ast.statics["st"]
        
        assert addr_static.id == "addr"
        assert isinstance(addr_static.ty, DataType_Single)
        assert isinstance(addr_static.expr, LitExpr)
        assert addr_static.expr.literal.lit.value == 0
        
        # Check static assignments in flow
        flow = list(ast.flows.values())[0]
        addr_assign = flow.body[1]  # addr = addr + 4
        st_assign = flow.body[2]    # st = st + 1
        
        assert isinstance(addr_assign, AssignStmt)
        assert not addr_assign.is_let  # Not a let statement
        assert addr_assign.lhs.name == "addr"
        assert isinstance(addr_assign.rhs, BinaryExpr)

    def test_array_static_with_aggregate(self):
        """Test static array with aggregate initializer"""
        source = """
        static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
        
        rtype array_test(rs1: u5, rs2: u5, rd: u5) {
            let idx: u32 = _irf[rs1];
            let theta: u32 = thetas[idx];
            _irf[rd] = theta;
        }
        """
        
        ast = parse_proc(source)
        assert len(ast.statics) == 1
        
        # Check array static
        thetas_static = ast.statics["thetas"]
        assert thetas_static.id == "thetas"
        assert isinstance(thetas_static.ty, DataType_Array)
        assert thetas_static.ty.dimensions == [8]
        
        # Check aggregate initializer
        assert isinstance(thetas_static.expr, AggregateExpr)
        assert len(thetas_static.expr.elements) == 8
        
        # Verify first and last values
        first_val = thetas_static.expr.elements[0]
        last_val = thetas_static.expr.elements[7]
        assert isinstance(first_val, LitExpr)
        assert first_val.literal.lit.value == 1474560
        assert isinstance(last_val, LitExpr)
        assert last_val.literal.lit.value == 14667
        
        # Check array access in flow
        flow = list(ast.flows.values())[0]
        array_access = flow.body[1]
        assert isinstance(array_access.rhs, IndexExpr)
        assert array_access.rhs.expr.name == "thetas"

    def test_complex_bitwise_operations(self):
        """Test complex bitwise operations and bit slicing"""
        source = """
        rtype cplx_mult(rs1: u5, rs2: u5, rd: u5) {
            let r1: i32 = _irf[rs1];
            let r2: i32 = _irf[rs2];
            
            let ar: i16 = r1[31:16];
            let ai: i16 = r1[15:0];
            let br: i16 = r2[31:16];
            let bi: i16 = r2[15:0];
            let zr: i32 = ar * br - ai * bi;
            let zi: i32 = ai * br + ar * bi;
            _irf[rd] = {zr[15:0], zi[15:0]};
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check signed types in the flow  
        assert len(flow.body) == 9
        
        # Check bit slicing: r1[31:16]
        ar_assign = flow.body[2]
        assert isinstance(ar_assign.rhs, SliceExpr)
        slice_expr = ar_assign.rhs
        assert slice_expr.expr.name == "r1"
        assert isinstance(slice_expr.start, LitExpr)
        assert isinstance(slice_expr.end, LitExpr)
        assert slice_expr.start.literal.lit.value == 31
        assert slice_expr.end.literal.lit.value == 16
        
        # Check complex arithmetic: ar * br - ai * bi
        zr_assign = flow.body[6]
        assert isinstance(zr_assign.rhs, BinaryExpr)
        assert zr_assign.rhs.op == BinaryOp.SUB
        
        # Left side should be multiplication
        left_mult = zr_assign.rhs.left
        assert isinstance(left_mult, BinaryExpr)
        assert left_mult.op == BinaryOp.MUL
        
        # Check aggregate assignment with bit slicing
        final_assign = flow.body[8]  # Last statement is now index 8
        assert isinstance(final_assign.rhs, AggregateExpr)
        assert len(final_assign.rhs.elements) == 2
        
        # Both elements should be bit slices
        elem1 = final_assign.rhs.elements[0]
        elem2 = final_assign.rhs.elements[1]
        assert isinstance(elem1, SliceExpr)
        assert isinstance(elem2, SliceExpr)

    def test_shift_operations(self):
        """Test left and right shift operations"""
        source = """
        rtype shift_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let left_shifted: u32 = r1 << 1;
            let right_shifted: u32 = r1 >> 2;
            _irf[rd] = left_shifted + right_shifted;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check left shift
        left_shift = flow.body[1]
        assert isinstance(left_shift.rhs, BinaryExpr)
        assert left_shift.rhs.op == BinaryOp.LSHIFT
        assert isinstance(left_shift.rhs.right, LitExpr)
        assert left_shift.rhs.right.literal.lit.value == 1
        
        # Check right shift  
        right_shift = flow.body[2]
        assert isinstance(right_shift.rhs, BinaryExpr)
        assert right_shift.rhs.op == BinaryOp.RSHIFT
        assert isinstance(right_shift.rhs.right, LitExpr)
        assert right_shift.rhs.right.literal.lit.value == 2

    def test_hex_literals(self):
        """Test hexadecimal literal parsing"""
        source = """
        rtype hex_test(rs1: u5, rs2: u5, rd: u5) {
            let mask: u32 = 0xEDB88320;
            let val: u32 = 32'hDEADBEEF;
            _irf[rd] = mask ^ val;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check 0x format hex literal  
        mask_assign = flow.body[0]  # First assignment: let mask: u32 = 0xEDB88320;
        assert isinstance(mask_assign.rhs, LitExpr)
        # 0xEDB88320 = 3988292384 in decimal
        assert mask_assign.rhs.literal.lit.value == 3988292384
        
        # Check 32'h format hex literal
        val_assign = flow.body[1]   # Second assignment: let val: u32 = 32'hDEADBEEF;
        assert isinstance(val_assign.rhs, LitExpr)
        # 32'hDEADBEEF = 3735928559 in decimal
        assert val_assign.rhs.literal.lit.value == 3735928559
        assert val_assign.rhs.literal.ty.width == 32
        
        # Check XOR operation
        xor_assign = flow.body[2]  # Third assignment: _irf[rd] = mask ^ val;
        assert isinstance(xor_assign.rhs, BinaryExpr)
        assert xor_assign.rhs.op == BinaryOp.BIT_XOR

    def test_multiple_register_reads(self):
        """Test multiple reads from same register"""
        source = """
        rtype multiple_rs_read(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs1];  // Same register read twice
            let r3: u32 = _irf[rs2];
            let r4: u32 = _irf[rs2];  // Same register read twice
            _irf[rd] = r1 + r2 + r3 + r4;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        assert len(flow.body) == 5
        
        # Check that we can read same register multiple times
        read1 = flow.body[0]
        read2 = flow.body[1]
        read3 = flow.body[2]
        read4 = flow.body[3]
        
        # First two reads from rs1
        assert read1.rhs.indices[0].name == "rs1"
        assert read2.rhs.indices[0].name == "rs1"
        
        # Next two reads from rs2
        assert read3.rhs.indices[0].name == "rs2"
        assert read4.rhs.indices[0].name == "rs2"
        
        # Final sum should be nested additions
        sum_expr = flow.body[4].rhs
        assert isinstance(sum_expr, BinaryExpr)
        assert sum_expr.op == BinaryOp.ADD

    def test_comprehensive_expression_precedence(self):
        """Test complex expression with multiple operator precedence levels"""
        source = """
        rtype precedence_test(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = _irf[rs1];
            let b: u32 = _irf[rs2];
            
            // Test: a + b * 2 << 1 & 0xFF == 0x10
            let complex: u32 = a + b * 2 << 1 & 32'hFF;
            let comparison: u1 = complex == 32'd16;
            
            _irf[rd] = if comparison {1} else {0};
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Validate complex expression precedence: a + b * 2 << 1 & 32'hFF
        complex_assign = flow.body[2]
        complex_expr = complex_assign.rhs
        
        # Should parse as: ((a + (b * 2)) << 1) & 32'hFF
        assert isinstance(complex_expr, BinaryExpr)
        assert complex_expr.op == BinaryOp.BIT_AND  # Lowest precedence
        
        # Left side: (a + (b * 2)) << 1
        left_shift = complex_expr.left
        assert isinstance(left_shift, BinaryExpr)
        assert left_shift.op == BinaryOp.LSHIFT
        
        # Left of shift: a + (b * 2) 
        addition = left_shift.left
        assert isinstance(addition, BinaryExpr)
        assert addition.op == BinaryOp.ADD
        
        # Right of addition: b * 2 (multiplication has higher precedence)
        multiplication = addition.right
        assert isinstance(multiplication, BinaryExpr)
        assert multiplication.op == BinaryOp.MUL
        assert isinstance(multiplication.left, IdentExpr)
        assert multiplication.left.name == "b"
        assert isinstance(multiplication.right, LitExpr)
        assert multiplication.right.literal.lit.value == 2
        
        # Right side of AND: 32'hFF
        mask_literal = complex_expr.right
        assert isinstance(mask_literal, LitExpr)
        assert mask_literal.literal.lit.value == 0xFF
        assert mask_literal.literal.ty.width == 32

    def test_error_handling_and_edge_cases(self):
        """Test various edge cases and boundary conditions"""
        
        # Test empty body
        source1 = """rtype empty_test() {}"""
        ast1 = parse_proc(source1)
        flow1 = list(ast1.flows.values())[0]
        assert flow1.body is None or len(flow1.body) == 0
        
        # Test maximum width literals
        source2 = """
        rtype wide_literal_test(rd: u5) {
            let max_val: u64 = 64'hFFFFFFFFFFFFFFFF;
            _irf[rd] = $unsigned(max_val);
        }
        """
        ast2 = parse_proc(source2)
        flow2 = list(ast2.flows.values())[0]
        max_literal = flow2.body[0].rhs
        assert isinstance(max_literal, LitExpr)
        assert max_literal.literal.ty.width == 64
        assert max_literal.literal.lit.value == 0xFFFFFFFFFFFFFFFF
        
        # Validate cast operation
        cast_stmt = flow2.body[1]
        assert isinstance(cast_stmt.rhs, UnaryExpr)
        assert cast_stmt.rhs.op == UnaryOp.UNSIGNED_CAST

    def test_nested_function_calls_and_complex_indexing(self):
        """Test complex indexing and function calls (without fn definitions)"""
        source = """
        rtype complex_indexing(rs1: u5, rs2: u5, rd: u5) {
            let base: u32 = _irf[rs1];
            let offset: u32 = _irf[rs2];

            // Complex memory access with function call
            let addr: u32 = base + helper(offset);
            let data: u32 = _mem[addr];

            // Chained operations
            let result: u32 = helper(data + base);
            _irf[rd] = result;
        }
        """

        ast = parse_proc(source)
        assert len(ast.flows) == 1

        flow = list(ast.flows.values())[0]

        # Validate function call in address calculation
        addr_calc = flow.body[2].rhs  # base + helper(offset)
        assert isinstance(addr_calc, BinaryExpr)
        assert addr_calc.op == BinaryOp.ADD
        assert isinstance(addr_calc.right, CallExpr)
        assert addr_calc.right.name == "helper"
        assert len(addr_calc.right.args) == 1
        assert isinstance(addr_calc.right.args[0], IdentExpr)
        assert addr_calc.right.args[0].name == "offset"
        
        # Validate nested function call: helper(data + base)
        result_calc = flow.body[4].rhs
        assert isinstance(result_calc, CallExpr)
        assert result_calc.name == "helper"
        nested_expr = result_calc.args[0]
        assert isinstance(nested_expr, BinaryExpr)
        assert nested_expr.op == BinaryOp.ADD
        assert nested_expr.left.name == "data"
        assert nested_expr.right.name == "base"

    def test_all_binary_operators(self):
        """Comprehensive test of all binary operators"""
        source = """
        rtype operator_test(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = _irf[rs1];
            let b: u32 = _irf[rs2];
            
            // Arithmetic operators
            let add_result: u32 = a + b;
            let sub_result: u32 = a - b;
            let mul_result: u32 = a * b;
            let div_result: u32 = a / b;
            let rem_result: u32 = a % b;
            
            // Comparison operators  
            let eq_result: u1 = a == b;
            let ne_result: u1 = a != b;
            let lt_result: u1 = a < b;
            let le_result: u1 = a <= b;
            let gt_result: u1 = a > b;
            let ge_result: u1 = a >= b;
            
            // Logical operators
            let and_result: u1 = eq_result && ne_result;
            let or_result: u1 = eq_result || ne_result;
            
            // Bitwise operators
            let bit_and: u32 = a & b;
            let bit_or: u32 = a | b;
            let bit_xor: u32 = a ^ b;
            let lshift: u32 = a << 2;
            let rshift: u32 = a >> 1;
            
            _irf[rd] = bit_xor;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Test all arithmetic operators
        operators = [
            (BinaryOp.ADD, 2), (BinaryOp.SUB, 3), (BinaryOp.MUL, 4), 
            (BinaryOp.DIV, 5), (BinaryOp.REM, 6),
            (BinaryOp.EQ, 7), (BinaryOp.NE, 8), (BinaryOp.LT, 9),
            (BinaryOp.LE, 10), (BinaryOp.GT, 11), (BinaryOp.GE, 12),
            (BinaryOp.AND, 13), (BinaryOp.OR, 14),
            (BinaryOp.BIT_AND, 15), (BinaryOp.BIT_OR, 16), (BinaryOp.BIT_XOR, 17),
            (BinaryOp.LSHIFT, 18), (BinaryOp.RSHIFT, 19)
        ]
        
        for expected_op, stmt_idx in operators:
            stmt = flow.body[stmt_idx]
            assert isinstance(stmt.rhs, BinaryExpr)
            assert stmt.rhs.op == expected_op, f"Statement {stmt_idx} should have operator {expected_op}, got {stmt.rhs.op}"

    def test_all_unary_operators(self):
        """Test all unary operators and type casts"""
        source = """
        rtype unary_test(rs1: u5, rd: u5) {
            let val: i32 = $signed(_irf[rs1]);
            
            // Unary arithmetic
            let negated: i32 = -val;
            let bit_not: u32 = ~$unsigned(val);
            let logical_not: u1 = !(val == 0);
            
            // Type casts
            let as_unsigned: u32 = $unsigned(val);
            let as_signed: i32 = $signed(as_unsigned);
            let as_float: f32 = $f32(val);
            let as_double: f64 = $f64(val);
            let as_int: u32 = $int(as_float);
            let as_uint: u32 = $uint(as_double);
            
            _irf[rd] = as_uint;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Test unary operations
        unary_ops = [
            (UnaryOp.NEG, 1),
            (UnaryOp.BIT_NOT, 2),
            (UnaryOp.NOT, 3)
        ]
        
        for expected_op, stmt_idx in unary_ops:
            stmt = flow.body[stmt_idx]
            assert isinstance(stmt.rhs, UnaryExpr)
            assert stmt.rhs.op == expected_op
        
        # Test type cast operations
        cast_ops = [
            (UnaryOp.SIGNED_CAST, 0),    # Initial cast in first statement
            (UnaryOp.UNSIGNED_CAST, 4),  # as_unsigned
            (UnaryOp.SIGNED_CAST, 5),    # as_signed  
            (UnaryOp.F32_CAST, 6),       # as_float
            (UnaryOp.F64_CAST, 7),       # as_double
            (UnaryOp.INT_CAST, 8),       # as_int
            (UnaryOp.UINT_CAST, 9)       # as_uint
        ]
        
        for expected_op, stmt_idx in cast_ops:
            stmt = flow.body[stmt_idx]
            if stmt_idx == 0:
                # First statement has cast in RHS of assignment
                cast_expr = stmt.rhs
            else:
                cast_expr = stmt.rhs
            assert isinstance(cast_expr, UnaryExpr)
            assert cast_expr.op == expected_op


if __name__ == "__main__":
    pytest.main([__file__])