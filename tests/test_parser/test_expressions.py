"""
Test cases for expression parsing

Tests all expression types: binary, unary, if, call, index, slice, etc.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestBasicExpressions:
    """Test basic expression parsing"""

    def test_identifier_expression(self):
        """Test identifier expressions"""
        source = "rtype test() { mdfasdf; }"
        ast = parse_proc(source)
        assert ast is not None

    def test_tuple_expression(self):
        """Test tuple expressions"""
        test_cases = [
            "(a,b,c)",
            "(a)",
        ]

        for case in test_cases:
            source = f"rtype test() {{ {case}; }}"
            ast = parse_proc(source)
            assert ast is not None

    def test_function_call(self):
        """Test function call expressions"""
        source = "rtype test() { xyz(xxx); }"
        ast = parse_proc(source)
        assert ast is not None

    def test_indexing_expressions(self):
        """Test array indexing"""
        test_cases = [
            "a[asdaadasd]",
            "a[1]",
        ]

        for case in test_cases:
            source = f"rtype test() {{ {case}; }}"
            ast = parse_proc(source)
            assert ast is not None

    def test_slicing_expression(self):
        """Test array slicing"""
        source = "rtype test() { vec[1:3]; }"
        ast = parse_proc(source)
        assert ast is not None


class TestBinaryOperators:
    """Test binary operator expressions"""

    def test_arithmetic_operators(self):
        """Test arithmetic binary operators"""
        operators = ["+", "-", "*", "/", "%"]

        for op in operators:
            source = f"rtype test(a: u32, b: u32) {{ let x: u32 = a {op} b; }}"
            ast = parse_proc(source)
            assert ast is not None
            flow = list(ast.flows.values())[0]
            assign = flow.body[0]
            assert isinstance(assign.rhs, BinaryExpr)

    def test_comparison_operators(self):
        """Test comparison operators"""
        operators = ["==", "!=", "<", "<=", ">", ">="]

        for op in operators:
            source = f"rtype test(a: u32, b: u32) {{ let x: u1 = a {op} b; }}"
            ast = parse_proc(source)
            assert ast is not None

    def test_logical_operators(self):
        """Test logical operators"""
        operators = ["&&", "||"]

        for op in operators:
            source = f"rtype test(a: u1, b: u1) {{ let x: u1 = a {op} b; }}"
            ast = parse_proc(source)
            assert ast is not None

    def test_bitwise_operators(self):
        """Test bitwise operators"""
        operators = ["&", "|", "^", "<<", ">>"]

        for op in operators:
            source = f"rtype test(a: u32, b: u32) {{ let x: u32 = a {op} b; }}"
            ast = parse_proc(source)
            assert ast is not None


class TestUnaryOperators:
    """Test unary operator expressions"""

    def test_negation(self):
        """Test negation operator"""
        source = "rtype test(a: i32) { let x: i32 = -a; }"
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, UnaryExpr)
        assert assign.rhs.op == UnaryOp.NEG

    def test_logical_not(self):
        """Test logical NOT operator"""
        source = "rtype test(a: u1) { let x: u1 = !a; }"
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, UnaryExpr)
        assert assign.rhs.op == UnaryOp.NOT

    def test_bitwise_not(self):
        """Test bitwise NOT operator"""
        source = "rtype test(a: u32) { let x: u32 = ~a; }"
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, UnaryExpr)
        assert assign.rhs.op == UnaryOp.BIT_NOT

    def test_cast_operators(self):
        """Test type cast operators"""
        casts = [
            ("$signed", UnaryOp.SIGNED_CAST),
            ("$unsigned", UnaryOp.UNSIGNED_CAST),
            ("$f32", UnaryOp.F32_CAST),
            ("$f64", UnaryOp.F64_CAST),
        ]

        for cast_op, expected_op in casts:
            source = f"rtype test(a: u32) {{ let x: i32 = {cast_op}(a); }}"
            ast = parse_proc(source)
            flow = list(ast.flows.values())[0]
            assign = flow.body[0]
            assert isinstance(assign.rhs, UnaryExpr)
            assert assign.rhs.op == expected_op


class TestIfExpressions:
    """Test if expression parsing"""

    def test_simple_if_expression(self):
        """Test basic if-else expression"""
        source = """
        rtype test(x: u32) {
            let result: u32 = if x > 5 {10} else {20};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, IfExpr)
        assert isinstance(assign.rhs.condition, BinaryExpr)

    def test_if_expression_in_assignment(self):
        """Test if expression as assignment RHS"""
        source = """
        rtype test(a: u32, b: u32) {
            let max: u32 = if a > b {a} else {b};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, IfExpr)

    def test_nested_if_expressions(self):
        """Test nested if expressions"""
        source = """
        rtype test(x: u32) {
            let result: u32 = if x > 10 {1} else {if x > 5 {2} else {3}};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, IfExpr)
        assert isinstance(assign.rhs.else_branch, IfExpr)

    def test_if_with_complex_conditions(self):
        """Test if with complex boolean conditions"""
        source = """
        rtype test(a: u32, b: u32, c: u32) {
            let result: u32 = if (a > b) && (b > c) {1} else {0};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, IfExpr)
        assert isinstance(assign.rhs.condition, BinaryExpr)

    def test_if_with_arithmetic_expressions(self):
        """Test if with arithmetic in branches"""
        source = """
        rtype test(x: u32, y: u32) {
            let result: u32 = if x > y {x + y} else {x - y};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, IfExpr)
        assert isinstance(assign.rhs.then_branch, BinaryExpr)

    def test_chained_if_expressions(self):
        """Test chained if-else-if expressions"""
        source = """
        rtype test(x: u32) {
            let result: u32 = if x > 10 {3} else {if x > 5 {2} else {if x > 0 {1} else {0}}};
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        # First if
        assert isinstance(assign.rhs, IfExpr)
        # Second if (in else branch)
        middle_if = assign.rhs.else_branch
        assert isinstance(middle_if, IfExpr)


class TestOperatorPrecedence:
    """Test operator precedence"""

    def test_arithmetic_precedence(self):
        """Test that * has higher precedence than +"""
        source = """
        rtype test(a: u32, b: u32, c: u32) {
            let result: u32 = a + b * c;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        # Should parse as: a + (b * c)
        assert isinstance(assign.rhs, BinaryExpr)
        assert assign.rhs.op == BinaryOp.ADD
        assert isinstance(assign.rhs.right, BinaryExpr)
        assert assign.rhs.right.op == BinaryOp.MUL

    def test_bitwise_precedence(self):
        """Test bitwise operator precedence"""
        source = """
        rtype test(a: u32, b: u32, c: u32) {
            let result: u32 = a | b & c;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        # Should parse as: a | (b & c)
        assert isinstance(assign.rhs, BinaryExpr)
        assert assign.rhs.op == BinaryOp.BIT_OR

    def test_shift_vs_arithmetic_precedence(self):
        """Test that arithmetic has higher precedence than shift"""
        source = """
        rtype test(a: u32, b: u32) {
            let result: u32 = a << 2 + b;
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]

        # Should parse as: a << (2 + b)
        assert isinstance(assign.rhs, BinaryExpr)
        assert assign.rhs.op == BinaryOp.LSHIFT

    def test_comparison_vs_bitwise_precedence(self):
        """Test comparison vs bitwise precedence"""
        source = """
        rtype test(a: u32, b: u32, c: u32) {
            let result: u1 = a & b == c;
        }
        """
        ast = parse_proc(source)
        assert ast is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
