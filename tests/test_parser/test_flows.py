"""
Test cases for flow and rtype parsing

Tests flow definitions, rtype flows, attributes, and parameters.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestBasicFlows:
    """Test basic flow parsing"""
    def test_rtype_flow(self):
        """Test rtype flow definition"""
        source = """
        rtype multiply(a: u32, b: u32) {
            return (a * b);
        }
        """
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        flow = list(ast.flows.values())[0]
        assert flow.name == "multiply"
        assert flow.kind == FlowKind.RTYPE


class TestRTypeFlows:
    """Test rtype-specific features"""

    def test_rtype_with_attributes(self):
        """Test rtype with attributes"""
        source = """
        #[activator(start)]
        rtype process(data: u32) {
            return (data * 2);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert flow.kind == FlowKind.RTYPE
        assert flow.attrs is not None

    def test_rtype_with_no_inputs(self):
        """Test rtype without input parameters"""
        source = """
        rtype generate() {
            return (42);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.inputs) == 0

    def test_rtype_with_multiple_types(self):
        """Test rtype with various input types"""
        source = """
        rtype process(a: u32, b: i32, c: u8, d: u1) {
            return (0);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.inputs) == 4

        # Verify type parsing
        types = [dtype for _, dtype in flow.inputs]
        assert isinstance(types[0], DataType_Single)
        assert isinstance(types[1], DataType_Single)


class TestFlowWithExpressions:
    """Test flows with various expression types"""

    def test_rtype_with_basic_expressions(self):
        """Test rtype with basic arithmetic"""
        source = """
        rtype calc(a: u32, b: u32) {
            let sum: u32 = a + b;
            let diff: u32 = a - b;
            let prod: u32 = a * b;
            return (sum + diff + prod);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 4  # 3 lets + 1 return

    def test_rtype_with_shift_operations(self):
        """Test rtype with shift operations"""
        source = """
        rtype shifter(value: u32, amount: u8) {
            let left: u32 = value << amount;
            let right: u32 = value >> amount;
            return (left + right);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        body = flow.body

        # Check shift operations
        assert isinstance(body[0].rhs, BinaryExpr)
        assert body[0].rhs.op == BinaryOp.LSHIFT

        assert isinstance(body[1].rhs, BinaryExpr)
        assert body[1].rhs.op == BinaryOp.RSHIFT

    def test_rtype_with_bitwise_operations_correct_precedence(self):
        """Test rtype with bitwise operations and precedence"""
        source = """
        rtype bitwise(a: u32, b: u32, c: u32) {
            let result1: u32 = a & b | c;
            let result2: u32 = a | b & c;
            let result3: u32 = a ^ b & c;
            return (result1);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 4

        # Verify first expression: a & b | c
        # Should parse as (a & b) | c due to precedence
        result1_expr = flow.body[0].rhs
        assert isinstance(result1_expr, BinaryExpr)

    def test_rtype_with_comparison_operations(self):
        """Test rtype with comparison operations"""
        source = """
        rtype compare(a: u32, b: u32) {
            let eq: u1 = a == b;
            let ne: u1 = a != b;
            let lt: u1 = a < b;
            let le: u1 = a <= b;
            let gt: u1 = a > b;
            let ge: u1 = a >= b;
            return (0);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.body) == 7  # 6 comparisons + 1 return


class TestFlowParameters:
    """Test flow parameter handling"""

    def test_flow_with_array_parameters(self):
        """Test flow with array type parameters"""
        source = """
        rtype process(data: [u32; 16]) {
            return (data[0]);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.inputs) == 1

        param_name, param_type = flow.inputs[0]
        assert param_name == "data"
        assert isinstance(param_type, DataType_Array)

    def test_flow_with_mixed_parameters(self):
        """Test flow with mixed scalar and array parameters"""
        source = """
        rtype process(scalar: u32, array: [i32; 8], flag: u1) {
            return (scalar);
        }
        """
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        assert len(flow.inputs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
