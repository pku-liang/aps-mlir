"""
Test cases for declarations

Tests static variables, regfiles, and their attributes.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestRegfileDeclarations:
    """Test regfile declaration parsing"""

    def test_regfile_definition(self):
        """Test regfile definition parsing"""
        source = "regfile rf(32, 16);"
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.regfiles) == 1

        regfile = list(ast.regfiles.values())[0]
        assert regfile.name == "rf"
        assert regfile.width == 32
        assert regfile.depth == 16

    def test_multiple_regfiles(self):
        """Test multiple regfile declarations"""
        source = """
        regfile rf1(32, 16);
        regfile rf2(64, 32);
        """
        ast = parse_proc(source)
        assert len(ast.regfiles) == 2


class TestStaticDeclarations:
    """Test static variable declarations"""

    def test_simple_static(self):
        """Test simple static declaration"""
        source = "static counter: u32;"
        ast = parse_proc(source)
        assert len(ast.statics) == 1

        static = list(ast.statics.values())[0]
        assert static.id == "counter"
        assert isinstance(static.ty, DataType_Single)

    def test_static_with_initialization(self):
        """Test static with initialization"""
        source = "static counter: u32 = 42;"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert static.expr is not None
        assert isinstance(static.expr, LitExpr)

    def test_static_array(self):
        """Test static array declaration"""
        source = "static buffer: [i32; 1024];"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert isinstance(static.ty, DataType_Array)

    def test_static_array_with_aggregate(self):
        """Test static array with aggregate initialization"""
        source = "static arr: [u32; 3] = {1, 2, 3};"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert isinstance(static.expr, AggregateExpr)
        assert len(static.expr.elements) == 3


class TestStaticAttributes:
    """Test attribute parsing on static declarations"""

    def test_static_with_single_attribute(self):
        """Test static with single attribute"""
        source = """
        #[impl("1rw")]
        static buffer: [i32; 1024];
        """
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.statics) == 1

        buffer = ast.statics["buffer"]
        assert buffer.attrs is not None
        assert "impl" in buffer.attrs
        assert isinstance(buffer.attrs["impl"], StringLitExpr)
        assert buffer.attrs["impl"].value == "1rw"

    def test_static_with_multiple_attributes(self):
        """Test static with multiple attributes"""
        source = """
        #[impl("2rw")]
        #[partition("cyclic")]
        static scratch: [i32; 512];
        """
        ast = parse_proc(source)
        assert ast is not None

        scratch = ast.statics["scratch"]
        assert len(scratch.attrs) == 2
        assert "impl" in scratch.attrs
        assert "partition" in scratch.attrs

        assert isinstance(scratch.attrs["impl"], StringLitExpr)
        assert scratch.attrs["impl"].value == "2rw"

        assert isinstance(scratch.attrs["partition"], StringLitExpr)
        assert scratch.attrs["partition"].value == "cyclic"

    def test_static_with_attribute_and_init(self):
        """Test static with attribute and initialization"""
        source = """
        #[impl("1rw")]
        static counter: i32 = 42;
        """
        ast = parse_proc(source)
        assert ast is not None

        counter = ast.statics["counter"]
        assert "impl" in counter.attrs
        assert counter.expr is not None
        assert isinstance(counter.expr, LitExpr)

    def test_static_without_attributes(self):
        """Test static without attributes (regression test)"""
        source = """
        static normal: [i32; 256];
        """
        ast = parse_proc(source)
        assert ast is not None

        normal = ast.statics["normal"]
        assert len(normal.attrs) == 0

    def test_static_attribute_with_integer_value(self):
        """Test static with integer attribute value"""
        source = """
        #[factor(4)]
        static data: [i32; 1024];
        """
        ast = parse_proc(source)
        assert ast is not None

        data = ast.statics["data"]
        assert "factor" in data.attrs
        assert isinstance(data.attrs["factor"], LitExpr)

    def test_mixed_statics_with_and_without_attributes(self):
        """Test multiple statics, some with attributes, some without"""
        source = """
        #[impl("1rw")]
        static buffer1: [i32; 512];

        static buffer2: [i32; 256];

        #[impl("2rw")]
        #[partition("block")]
        static buffer3: [i32; 128];
        """
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.statics) == 3

        buffer1 = ast.statics["buffer1"]
        assert len(buffer1.attrs) == 1
        assert "impl" in buffer1.attrs

        buffer2 = ast.statics["buffer2"]
        assert len(buffer2.attrs) == 0

        buffer3 = ast.statics["buffer3"]
        assert len(buffer3.attrs) == 2
        assert "impl" in buffer3.attrs
        assert "partition" in buffer3.attrs


class TestDeclarationCombinations:
    """Test combinations of declarations"""

    def test_mixed_declarations(self):
        """Test mixing regfile and static declarations"""
        source = """
        regfile rf(32, 16);
        static counter: u32 = 0;
        static buffer: [i32; 256];
        """
        ast = parse_proc(source)
        assert len(ast.regfiles) == 1
        assert len(ast.statics) == 2

    def test_declarations_with_flows(self):
        """Test declarations alongside flow definitions"""
        source = """
        static data: [u32; 100];

        rtype process() {
            data[0] = 42;
        }
        """
        ast = parse_proc(source)
        assert len(ast.statics) == 1
        assert len(ast.flows) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
