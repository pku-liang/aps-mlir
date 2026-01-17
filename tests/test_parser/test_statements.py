"""
Test cases for statement parsing

Tests assignment, return, loops, guards, directives, spawn statements.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestAssignmentStatements:
    """Test assignment statement parsing"""

    def test_simple_let_assignment(self):
        """Test simple let assignment"""
        source = """
        rtype test() {
            let x: u32 = 42;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 1
        assign = flow.body[0]
        assert isinstance(assign, AssignStmt)
        assert assign.is_let
        assert isinstance(assign.lhs, IdentExpr)
        assert assign.lhs.name == "x"

    def test_let_assignment_with_type(self):
        """Test let assignment with explicit type annotation"""
        source = """
        rtype test() {
            let result: u32 = 100;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert assign.is_let
        assert assign.type_annotation is not None

    def test_assignment_without_let(self):
        """Test assignment without let keyword"""
        source = """
        rtype test() {
            let x: u32 = 0;
            x = 42;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 2

        assign1 = flow.body[0]
        assert assign1.is_let

        assign2 = flow.body[1]
        assert not assign2.is_let

    def test_multiple_assignments(self):
        """Test multiple sequential assignments"""
        source = """
        rtype test() {
            let a: u32 = 1;
            let b: u32 = 2;
            let c: u32 = a + b;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 3
        for stmt in flow.body:
            assert isinstance(stmt, AssignStmt)

    def test_assignment_with_literals(self):
        """Test assignments with various literal types"""
        source = """
        rtype test() {
            let a: u32 = 0xFF;
            let b: u32 = 5'b101010;
            let c: u32 = 3'o123;
            let d: u32 = 15'd123;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assignments = flow.body

        assert len(assignments) == 4
        for assign in assignments:
            assert isinstance(assign, AssignStmt)
            assert isinstance(assign.rhs, LitExpr)

    def test_assignment_with_complex_expressions(self):
        """Test assignments with complex arithmetic expressions"""
        source = """
        rtype test(a: u32, b: u32, c: u32) {
            let expr1: u32 = a + b * c;
            let expr2: u32 = (a + b) * c;
            let expr3: u32 = a << 2 + b;
            let expr4: u32 = a & b | c;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 4

    def test_assignment_with_type_casts(self):
        """Test assignments with type casts"""
        source = """
        rtype test(a: u32, b: i32) {
            let signed_a: i32 = $signed(a);
            let unsigned_b: u32 = $unsigned(b);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 2

        for assign in flow.body:
            assert isinstance(assign.rhs, UnaryExpr)


class TestReturnStatements:
    """Test return statement parsing"""

    def test_simple_return(self):
        """Test simple return statement"""
        source = """
        rtype test() {
            return (42);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 1
        ret = flow.body[0]
        assert isinstance(ret, ReturnStmt)

    def test_return_tuple(self):
        """Test return with tuple"""
        source = """
        rtype test() {
            return (1, 2, 3);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        ret = flow.body[0]
        assert isinstance(ret, ReturnStmt)
        # The tuple is wrapped as a single expression
        assert len(ret.exprs) == 1
        assert isinstance(ret.exprs[0], TupleExpr)


class TestLoopStatements:
    """Test loop statement parsing"""

    def test_do_while_loop(self):
        """Test comprehensive do-while loop parsing"""
        source = """
        rtype test_loop() {
            with i: u8 = (0, i + 1) do {
                let x: u32 = i * 2;
            } while i_ < 10;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 1

        do_while = flow.body[0]
        assert isinstance(do_while, DoWhileStmt)
        assert len(do_while.bindings) == 1
        assert len(do_while.body) == 1

    def test_crc8_loop(self):
        """Test CRC8 loop with bitwise operations"""
        source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8_type1(data: u8) {
            with
                i: u8 = (0, i_) 
                crc: u8 = (0, crc_)
            do {
                let bit: u1 = (data >> i) & 1;
                let xor_val: u8 = if bit == 1 {0x8C} else {0};
                let crc_: u8 = crc ^ xor_val;
                crc_ = crc_ << 1;
                let i_: u8 = i + 1; // increment i
            } while i_ < 8;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]

        # Should have: do-while only (directives are attributes, not statements)
        assert len(flow.body) == 1

        # Check do-while loop
        do_while_stmt = flow.body[0]
        assert isinstance(do_while_stmt, DoWhileStmt)

        # Check that we have shift, XOR, AND operations in the loop body
        # 5 statements: let bit, let xor_val, let crc_, crc_ reassignment, let i_
        loop_body = do_while_stmt.body
        assert len(loop_body) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
