"""
Tests for select expressions (sel)
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import (
    SelectExpr, BinaryExpr, LitExpr, IdentExpr, BinaryOp,
    AssignStmt, Proc
)


class TestSelectExpression:
    """Test select expression parsing and semantics"""

    def test_simple_select(self):
        """Test basic select with two arms"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 10,
                x > 0: 20,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        assert len(ast.flows) == 1

        flow = ast.flows['test']
        assert flow is not None
        assert len(flow.body) == 3

        # Check the assignment with select
        assign_stmt = flow.body[1]
        assert isinstance(assign_stmt, AssignStmt)
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # Should have 1 conditional arm + 1 default
        assert len(sel_expr.arms) == 1
        assert sel_expr.default is not None

        # Check first arm
        cond1, val1 = sel_expr.arms[0]
        assert isinstance(cond1, BinaryExpr)
        assert cond1.op == BinaryOp.EQ
        assert isinstance(val1, LitExpr)

    def test_select_multiple_arms(self):
        """Test select with multiple conditional arms"""
        code = """
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
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # 3 conditional arms + 1 default (last one)
        assert len(sel_expr.arms) == 3
        assert sel_expr.default is not None

        # Verify conditions
        assert sel_expr.arms[0][0].op == BinaryOp.EQ
        assert sel_expr.arms[1][0].op == BinaryOp.LT
        assert sel_expr.arms[2][0].op == BinaryOp.LT

    def test_select_single_arm(self):
        """Test select with only one arm (becomes default)"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let result: u32 = sel {
                1: 42,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[0]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # Single arm becomes default, no conditional arms
        assert len(sel_expr.arms) == 0
        assert sel_expr.default is not None

    def test_select_with_complex_conditions(self):
        """Test select with complex boolean conditions"""
        code = """
        rtype test(rs1: u5, rs2: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let y: u32 = _irf[rs2];
            let result: u32 = sel {
                (x == 0) && (y == 0): 0,
                (x > 0) && (y > 0): 1,
                (x < 0) || (y < 0): 2,
                1: 3,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[2]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # 3 conditional arms + default
        assert len(sel_expr.arms) == 3

        # Check that conditions are logical operations
        # (Need parentheses due to current grammar precedence)
        assert sel_expr.arms[0][0].op == BinaryOp.AND
        assert sel_expr.arms[1][0].op == BinaryOp.AND
        assert sel_expr.arms[2][0].op == BinaryOp.OR

    def test_select_with_complex_values(self):
        """Test select with complex value expressions"""
        code = """
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
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # Check that values are expressions
        assert isinstance(sel_expr.arms[0][1], BinaryExpr)
        assert sel_expr.arms[0][1].op == BinaryOp.ADD
        assert sel_expr.arms[1][1].op == BinaryOp.MUL
        assert sel_expr.arms[2][1].op == BinaryOp.LSHIFT

    def test_select_nested(self):
        """Test nested select expressions"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: sel {
                    x < 5: 1,
                    1: 2,
                },
                x > 0: 10,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)

        # First arm's value should be another SelectExpr
        inner_sel = sel_expr.arms[0][1]
        assert isinstance(inner_sel, SelectExpr)
        assert len(inner_sel.arms) == 1

    def test_select_in_assignment(self):
        """Test select used in various assignment contexts"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let a: u32 = sel { x == 0: 1, 1: 0, };
            let b: u32 = a + sel { x > 0: 10, 1: 20, };
            _irf[rd] = b;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        # First select in direct assignment
        assert isinstance(flow.body[1].rhs, SelectExpr)

        # Second select in binary expression
        binary_expr = flow.body[2].rhs
        assert isinstance(binary_expr, BinaryExpr)
        assert isinstance(binary_expr.right, SelectExpr)

    def test_select_with_bit_slicing(self):
        """Test select with bit slicing in conditions and values"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x[7:0] == 0: 100,
                x[15:8] == 0: 200,
                1: x[31:16],
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs
        assert isinstance(sel_expr, SelectExpr)
        assert len(sel_expr.arms) == 2

    def test_select_string_representation(self):
        """Test that select expressions have proper string representation"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 10,
                x > 10: 20,
                1: 30,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs

        # Check string representation
        str_repr = str(sel_expr)
        assert "sel" in str_repr
        assert "default" in str_repr
        assert ":" in str_repr

    def test_select_preserves_order(self):
        """Test that select arms preserve their order"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x < 10: 1,
                x < 20: 2,
                x < 30: 3,
                1: 4,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[1]
        sel_expr = assign_stmt.rhs

        # All arms should have LT comparison
        for i in range(3):
            assert sel_expr.arms[i][0].op == BinaryOp.LT
            # Values should be 1, 2, 3
            assert sel_expr.arms[i][1].literal.lit.value == i + 1


class TestSelectEdgeCases:
    """Test edge cases and error conditions for select expressions"""

    def test_select_with_identifiers(self):
        """Test select with identifier values"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let a: u32 = 100;
            let b: u32 = 200;
            let result: u32 = sel {
                x == 0: a,
                x > 0: b,
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        assign_stmt = flow.body[3]
        sel_expr = assign_stmt.rhs

        # Check that values are identifiers
        assert isinstance(sel_expr.arms[0][1], IdentExpr)
        assert sel_expr.arms[0][1].name == "a"

    def test_select_with_memory_access(self):
        """Test select with memory access in values"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: _mem[0],
                x == 1: _mem[4],
                1: _mem[8],
            };
            _irf[rd] = result;
        }
        """
        ast = parse_proc(code, "test.cadl")
        assert len(ast.flows) == 1

    def test_select_in_memory_write(self):
        """Test select expression used in memory write"""
        code = """
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            _mem[0] = sel {
                x == 0: 100,
                1: 200,
            };
            _irf[rd] = 1;
        }
        """
        ast = parse_proc(code, "test.cadl")
        flow = ast.flows['test']

        # Should have expression statement with index expression containing select
        assert len(flow.body) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
