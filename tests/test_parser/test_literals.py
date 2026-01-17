"""
Test cases for literal parsing

Tests number literals, string literals, and literal width specifications.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestNumberLiterals:
    """Test number literal parsing"""

    def test_decimal_literals(self):
        """Test decimal number parsing"""
        test_cases = [
            "1231231",
            "12345678901234566789",
            "42",
        ]

        for case in test_cases:
            source = f"static x: u32 = {case};"
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.statics) == 1

    def test_hexadecimal_literals(self):
        """Test hexadecimal number parsing"""
        source = "static x: u32 = 0x1234;"
        ast = parse_proc(source)
        assert ast is not None
        static = list(ast.statics.values())[0]
        assert isinstance(static.expr, LitExpr)

    def test_binary_literals(self):
        """Test binary number parsing"""
        source = "static x: u32 = 5'b101010;"
        ast = parse_proc(source)
        assert ast is not None
        static = list(ast.statics.values())[0]
        assert isinstance(static.expr, LitExpr)

    def test_octal_literals(self):
        """Test octal number parsing"""
        source = "static x: u32 = 3'o123;"
        ast = parse_proc(source)
        assert ast is not None
        static = list(ast.statics.values())[0]
        assert isinstance(static.expr, LitExpr)


class TestLiteralWidths:
    """Test literal width specifications"""

    def test_width_specified_literals(self):
        """Test literals with explicit width specifications"""
        test_cases = [
            ("5'b101010", BasicType_ApUFixed, 5, 42),
            ("8'hFF", BasicType_ApUFixed, 8, 255),
            ("15'd123", BasicType_ApUFixed, 15, 123),
            ("3'o123", BasicType_ApUFixed, 3, 83),
        ]

        for literal_str, expected_type_class, expected_width, expected_value in test_cases:
            source = f"static x: u32 = {literal_str};"
            ast = parse_proc(source)

            static = list(ast.statics.values())[0]
            assert isinstance(static.expr, LitExpr)

            literal = static.expr.literal
            assert isinstance(literal.ty, expected_type_class)
            assert literal.ty.width == expected_width

            lit_inner = literal.lit
            assert isinstance(lit_inner, LiteralInner_Fixed)
            assert lit_inner.value == expected_value

    def test_default_width_literals(self):
        """Test that literals without width get default 32-bit unsigned type"""
        test_cases = [
            ("0x1234", BasicType_ApUFixed, 32),
            ("42", BasicType_ApUFixed, 32),
            ("0b1010", BasicType_ApUFixed, 32),
        ]

        for literal_str, expected_type_class, expected_width in test_cases:
            source = f"static x: u32 = {literal_str};"
            ast = parse_proc(source)

            static = list(ast.statics.values())[0]
            assert isinstance(static.expr, LitExpr)

            literal = static.expr.literal
            assert isinstance(literal.ty, expected_type_class)
            assert literal.ty.width == expected_width

    def test_literal_type_consistency(self):
        """Test that literal types are consistent across same format"""
        source = """
        static a: u32 = 8'b11111111;
        static b: u32 = 8'hFF;
        static c: u32 = 8'd255;
        """
        ast = parse_proc(source)

        assert len(ast.statics) == 3

        for static in ast.statics.values():
            assert isinstance(static.expr, LitExpr)
            assert isinstance(static.expr.literal.ty, BasicType_ApUFixed)
            assert static.expr.literal.ty.width == 8

    def test_number_format_parsing(self):
        """Test different number format parsing"""
        formats = {
            "binary": "5'b101010",
            "octal": "8'o377",
            "decimal": "15'd123",
            "hex": "8'hFF",
        }

        for format_name, literal in formats.items():
            source = f"static x: u32 = {literal};"
            ast = parse_proc(source)
            assert ast is not None, f"Failed to parse {format_name} literal: {literal}"


class TestStringLiterals:
    """Test string literal parsing"""

    def test_simple_string(self):
        """Test simple string literal"""
        source = '''
        rtype test() {
            let s: [u8; 10] = "hello";
        }
        '''
        ast = parse_proc(source)
        assert ast is not None
        flow = list(ast.flows.values())[0]
        assign = flow.body[0]
        assert isinstance(assign.rhs, StringLitExpr)
        assert assign.rhs.value == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
