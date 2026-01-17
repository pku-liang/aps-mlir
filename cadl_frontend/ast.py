"""
AST (Abstract Syntax Tree) classes for CADL

These classes mirror the Rust structures in the original cadl_rust implementation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# Type aliases
Ident = str
Map = Dict


# Expression types
@dataclass
class Expr:
    """Base class for all expressions"""
    pass

    def __str__(self) -> str:
        """String representation for expressions"""
        return self.__class__.__name__


@dataclass
class LitExpr(Expr):
    """Literal expression with proper type information"""
    literal: Literal
    
    def __str__(self) -> str:
        lit_inner = self.literal.lit
        if isinstance(lit_inner, (LiteralInner_Fixed, LiteralInner_Float)):
            return f"{lit_inner.value}_{self.literal.ty}"
        return f"literal_{self.literal.ty}"


@dataclass
class StringLitExpr(Expr):
    """String literal expression"""
    value: str

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass
class ComplexExpr(Expr):
    """Complex number expression"""
    real: Expr
    imag: Expr

    def __str__(self) -> str:
        return f"({self.real} + {self.imag}i)"


@dataclass
class IdentExpr(Expr):
    """Identifier expression"""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass
class TupleExpr(Expr):
    """Tuple expression"""
    elements: List[Expr]
    
    def __str__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"({elements_str})"


@dataclass
class BinaryExpr(Expr):
    """Binary operation expression"""
    op: BinaryOp
    left: Expr
    right: Expr
    
    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"


@dataclass
class UnaryExpr(Expr):
    """Unary operation expression"""
    op: UnaryOp
    operand: Expr
    
    def __str__(self) -> str:
        if self.op.value.startswith("$"):
            return f"{self.op.value}({self.operand})"
        else:
            return f"{self.op.value}{self.operand}"


@dataclass
class CallExpr(Expr):
    """Function call expression"""
    name: str
    args: List[Expr]
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"


@dataclass
class IndexExpr(Expr):
    """Array/vector indexing expression"""
    expr: Expr
    indices: List[Expr]
    
    def __str__(self) -> str:
        indices_str = ", ".join(str(idx) for idx in self.indices)
        return f"{self.expr}[{indices_str}]"


@dataclass
class SliceExpr(Expr):
    """Array/vector slicing expression"""
    expr: Expr
    start: Expr
    end: Expr

    def __str__(self) -> str:
        return f"{self.expr}[{self.start}:{self.end}]"


@dataclass
class RangeSliceExpr(Expr):
    """Range slicing expression with +: operator (e.g., arr[start +: length])"""
    expr: Expr
    start: Expr
    length: Optional[Expr] = None

    def __str__(self) -> str:
        length_str = str(self.length) if self.length else ""
        return f"{self.expr}[{self.start} +: {length_str}]"


@dataclass
class SelectExpr(Expr):
    """Select expression"""
    arms: List[tuple[Expr, Expr]]
    default: Expr

    def __str__(self) -> str:
        arms_str = ", ".join(f"{cond}:{val}" for cond, val in self.arms)
        return f"sel {{{arms_str}, default:{self.default}}}"


@dataclass
class IfExpr(Expr):
    """If expression"""
    condition: Expr
    then_branch: Expr
    else_branch: Expr

    def __str__(self) -> str:
        return f"if {self.condition} {{{self.then_branch}}} else {{{self.else_branch}}}"


@dataclass
class AggregateExpr(Expr):
    """Aggregate expression (like struct literals)"""
    elements: List[Expr]

    def __str__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"{{{elements_str}}}"


@dataclass
class ArrayLiteralExpr(Expr):
    """Array literal expression (used in annotations like #[attr([0, 1])])"""
    elements: List[Expr]

    def __str__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"[{elements_str}]"


# Binary operators
class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    REM = "%"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"


# Unary operators
class UnaryOp(Enum):
    NEG = "-"
    NOT = "!"
    BIT_NOT = "~"
    SIGNED_CAST = "$signed"
    UNSIGNED_CAST = "$unsigned"
    F32_CAST = "$f32"
    F64_CAST = "$f64"
    INT_CAST = "$int"
    UINT_CAST = "$uint"


# New IR Type System - matches type_sys_ir.rs
from enum import Enum

class BasicType:
    """Base class for BasicType enum variants"""
    pass

@dataclass  
class BasicType_ApFixed(BasicType):
    """Signed fixed-point type - BasicType::ApFixed(u32)"""
    width: int
    
    def __str__(self) -> str:
        return f"i{self.width}"

@dataclass
class BasicType_ApUFixed(BasicType): 
    """Unsigned fixed-point type - BasicType::ApUFixed(u32)"""
    width: int
    
    def __str__(self) -> str:
        return f"u{self.width}"

@dataclass
class BasicType_Float32(BasicType):
    """32-bit float type - BasicType::Float32"""
    pass
    
    def __str__(self) -> str:
        return "f32"

@dataclass
class BasicType_Float64(BasicType):
    """64-bit float type - BasicType::Float64"""  
    pass
    
    def __str__(self) -> str:
        return "f64"

@dataclass
class BasicType_String(BasicType):
    """String type - BasicType::String"""
    pass
    
    def __str__(self) -> str:
        return "string"

@dataclass
class BasicType_USize(BasicType):
    """USize type - BasicType::USize"""
    pass
    
    def __str__(self) -> str:
        return "usize"


class DataType:
    """Base class for DataType enum variants"""
    pass

@dataclass
class DataType_Single(DataType):
    """Single data type - DataType::Single(BasicType)"""
    basic_type: BasicType
    
    def __str__(self) -> str:
        return str(self.basic_type)

@dataclass
class DataType_Array(DataType):
    """Array data type - DataType::Array(BasicType, Vec<usize>)"""
    element_type: BasicType
    dimensions: List[int]
    
    def __str__(self) -> str:
        dims_str = "; ".join(str(d) for d in self.dimensions)
        return f"[{self.element_type}; {dims_str}]"

@dataclass
class DataType_Instance(DataType):
    """Instance data type - DataType::Instance"""
    pass
    
    def __str__(self) -> str:
        return "Instance"


class CompoundType:
    """Base class for CompoundType enum variants"""
    
    def to_basic(self) -> DataType:
        """Convert to basic type - matches Rust as_basic() method"""
        if isinstance(self, CompoundType_Basic):
            return self.data_type
        else:
            raise RuntimeError(f"Unknown CompoundType variant: {type(self)}")

@dataclass
class CompoundType_Basic(CompoundType):
    """Basic compound type - CompoundType::Basic(DataType)"""
    data_type: DataType

    def __str__(self) -> str:
        return str(self.data_type)


# Literal System - matches literal.rs
class LiteralInner:
    """Base class for LiteralInner enum variants"""
    pass

@dataclass
class LiteralInner_Fixed(LiteralInner):
    """Fixed-point literal - LiteralInner::Fixed(BigInt)"""
    value: int  # Using int instead of BigInt for simplicity

@dataclass
class LiteralInner_Float(LiteralInner):
    """Float literal - LiteralInner::Float(f64)"""
    value: float

@dataclass
class Literal:
    """Literal with type information - matches Rust Literal struct"""
    lit: LiteralInner
    ty: BasicType


# Statements
@dataclass
class Stmt:
    """Base class for all statements"""
    pass
    
    def __str__(self) -> str:
        return self.__class__.__name__


@dataclass
class ExprStmt(Stmt):
    """Expression statement"""
    expr: Expr
    
    def __str__(self) -> str:
        return f"{self.expr};"


@dataclass
class AssignStmt(Stmt):
    """Assignment statement"""
    is_let: bool
    lhs: Expr
    rhs: Expr
    type_annotation: Optional[DataType] = None
    
    def __str__(self) -> str:
        let_str = "let " if self.is_let else ""
        type_str = f": {self.type_annotation}" if self.type_annotation else ""
        return f"{let_str}{self.lhs}{type_str} = {self.rhs};"


@dataclass
class ReturnStmt(Stmt):
    """Return statement"""
    exprs: List[Expr]
    
    def __str__(self) -> str:
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"return ({exprs_str});"


@dataclass
class ForStmt(Stmt):
    """For loop statement"""
    init: Stmt
    condition: Expr
    update: Stmt
    body: List[Stmt]

    def __str__(self) -> str:
        lines = [f"for ({self.init}; {self.condition}; {self.update}) {{"]
        for stmt in self.body:
            lines.append(f"  {stmt}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class StaticStmt(Stmt):
    """Static variable declaration statement"""
    static: 'Static'

    def __str__(self) -> str:
        return f"{self.static}"


@dataclass
class GuardStmt(Stmt):
    """Guard statement (conditional execution)"""
    condition: Expr
    stmt: Stmt

    def __str__(self) -> str:
        return f"[{self.condition}]: {self.stmt}"


@dataclass
class DoWhileStmt(Stmt):
    """Do-while loop statement"""
    bindings: List['WithBinding']
    body: List[Stmt]
    condition: Expr

    def __str__(self) -> str:
        lines = []
        if self.bindings:
            bindings_str = ", ".join(str(b) for b in self.bindings)
            lines.append(f"with {bindings_str}")
        lines.append("do {")
        for stmt in self.body:
            lines.append(f"  {stmt}")
        lines.append(f"}} while {self.condition};")
        return "\n".join(lines)


@dataclass
class DirectiveStmt(Stmt):
    """Directive statement (compiler hints)"""
    name: str
    expr: Optional[Expr] = None

    def __str__(self) -> str:
        expr_str = f"({self.expr})" if self.expr else ""
        return f"[[{self.name}{expr_str}]]"


@dataclass
class SpawnStmt(Stmt):
    """Spawn statement (parallel execution)"""
    stmts: List[Stmt]

    def __str__(self) -> str:
        lines = ["spawn {"]
        for stmt in self.stmts:
            lines.append(f"  {stmt}")
        lines.append("}")
        return "\n".join(lines)

# Function-related structures
@dataclass
class WithBinding:
    """With binding for loop constructs"""
    id: str
    ty: BasicType
    init: Optional[Expr] = None
    next: Optional[Expr] = None

    def __str__(self) -> str:
        init_str = str(self.init) if self.init else ""
        next_str = str(self.next) if self.next else ""
        return f"{self.id}: {self.ty} = ({init_str}, {next_str})"


@dataclass
class FnArg:
    """Function argument (used for flows)"""
    id: str
    ty: CompoundType

    def __str__(self) -> str:
        return f"{self.id}: {self.ty}"


@dataclass
class Static:
    """Static variable declaration"""
    id: str
    ty: DataType
    expr: Optional[Expr] = None
    attrs: Dict[str, Optional[Expr]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Pretty print static variable"""
        attrs_str = ""
        if self.attrs:
            attr_list = []
            for name, value in self.attrs.items():
                if value is not None:
                    attr_list.append(f"#[{name}({value})]")
                else:
                    attr_list.append(f"#[{name}]")
            attrs_str = " ".join(attr_list) + " "

        if self.expr:
            return f"{attrs_str}{self.id}: {self.ty} = {self.expr}"
        else:
            return f"{attrs_str}{self.id}: {self.ty}"

# Flow-related structures
class FlowKind(Enum):
    DEFAULT = "default"
    RTYPE = "rtype"


@dataclass
class FlowAttributes:
    """Flow attributes (decorators)"""
    attrs: Dict[str, Optional[Expr]] = field(default_factory=dict)

    @classmethod
    def from_tuples(cls, tuples: List[tuple[str, Optional[Expr]]]) -> FlowAttributes:
        """Create from list of attribute tuples"""
        return cls(attrs=dict(tuples))

    def with_activator(self, activator: Expr) -> FlowAttributes:
        """Add activator attribute"""
        self.attrs["activator"] = activator
        return self

    def get(self, name: str) -> Optional[Expr]:
        """Get attribute by name"""
        return self.attrs.get(name)

    def set(self, name: str, value: Optional[Expr]) -> None:
        """Set attribute by name"""
        self.attrs[name] = value

    def __str__(self) -> str:
        """Pretty print attributes"""
        if not self.attrs:
            return ""
        attr_strs = []
        for name, value in self.attrs.items():
            if value is not None:
                attr_strs.append(f"{name}={value}")
            else:
                attr_strs.append(name)
        return f"[{', '.join(attr_strs)}]"


@dataclass
class Flow:
    """Flow definition"""
    name: str
    kind: FlowKind
    inputs: List[tuple[str, DataType]] = field(default_factory=list)
    attrs: FlowAttributes = field(default_factory=FlowAttributes)
    body: Optional[List[Stmt]] = None

    def fields(self) -> List[tuple[str, DataType]]:
        """Get flow input fields"""
        return self.inputs

    def get_body(self) -> Optional[List[Stmt]]:
        """Get flow body"""
        return self.body
    
    def __str__(self) -> str:
        kind_str = "rtype" if self.kind == FlowKind.RTYPE else "flow"
        args = ", ".join(f"{name}: {dtype}" for name, dtype in self.inputs)
        attrs_str = str(self.attrs) if self.attrs.attrs else ""
        attrs_part = f" {attrs_str}" if attrs_str else ""

        if self.body:
            lines = [f"{kind_str} {self.name}({args}){attrs_part} {{"]
            for stmt in self.body:
                lines.append(f"  {stmt}")
            lines.append("}")
            return "\n".join(lines)
        else:
            return f"{kind_str} {self.name}({args}){attrs_part} {{ (empty body) }}"


@dataclass
class Regfile:
    """Register file definition"""
    name: str
    width: int
    depth: int

    def __str__(self) -> str:
        """Pretty print regfile"""
        return f"{self.name}: {self.width}x{self.depth}"


# Processor structure
@dataclass
class ProcPart:
    """Base class for processor parts"""
    pass


@dataclass
class RegfilePart(ProcPart):
    """Regfile processor part"""
    regfile: Regfile


@dataclass
class FlowPart(ProcPart):
    """Flow processor part"""
    flow: Flow


@dataclass
class StaticPart(ProcPart):
    """Static processor part"""
    static: Static


@dataclass
class Proc:
    """Main processor structure"""
    regfiles: Map[str, Regfile] = field(default_factory=dict)
    flows: Map[str, Flow] = field(default_factory=dict)
    statics: Map[str, Static] = field(default_factory=dict)

    def get_flows(self) -> List[Flow]:
        """Get all flows"""
        return list(self.flows.values())

    def add_part(self, part: ProcPart) -> None:
        """Add a processor part"""
        if isinstance(part, RegfilePart):
            self.regfiles[part.regfile.name] = part.regfile
        elif isinstance(part, FlowPart):
            self.flows[part.flow.name] = part.flow
        elif isinstance(part, StaticPart):
            self.statics[part.static.id] = part.static
    
    def __str__(self) -> str:
        parts = []
        if self.regfiles:
            parts.append(f"{len(self.regfiles)} regfiles")
        if self.flows:
            parts.append(f"{len(self.flows)} flows")
        if self.statics:
            parts.append(f"{len(self.statics)} statics")
        return f"Proc({', '.join(parts)})"
    
    def pretty_print(self) -> str:
        """Detailed pretty printing of the processor"""
        lines = ["Processor AST:"]
        
        if self.regfiles:
            lines.append("  Regfiles:")
            for regfile in self.regfiles.values():
                lines.append(f"    {regfile}")

        if self.statics:
            lines.append("  Static Variables:")
            for static in self.statics.values():
                lines.append(f"    {static}")

        if self.flows:
            lines.append("  Flows:")
            for flow in self.flows.values():
                flow_lines = str(flow).split('\n')
                for line in flow_lines:
                    lines.append(f"    {line}")
                
        return "\n".join(lines)

    @classmethod
    def from_parts(cls, parts: List[ProcPart]) -> Proc:
        """Create processor from parts"""
        proc = cls()
        for part in parts:
            proc.add_part(part)
        return proc


# Helper methods for expressions
def expr_is_lval(expr: Expr) -> bool:
    """Check if expression is an lvalue"""
    return isinstance(expr, (IdentExpr, IndexExpr, SliceExpr, RangeSliceExpr))


def expr_as_literal(expr: Expr) -> Optional[str]:
    """Get expression as literal if possible"""
    if isinstance(expr, LitExpr):
        lit_inner = expr.literal.lit
        if isinstance(lit_inner, (LiteralInner_Fixed, LiteralInner_Float)):
            return str(lit_inner.value)
    return None


def expr_flatten(expr: Expr) -> List[Expr]:
    """Flatten tuple expressions"""
    if isinstance(expr, TupleExpr):
        result = []
        for element in expr.elements:
            result.extend(expr_flatten(element))
        return result
    return [expr]


# Convenience factory methods for type system  
def parse_basic_type_from_string(type_str: str) -> BasicType:
    """Parse a basic type from string (like 'u32', 'i8', 'f32')"""
    if type_str == "usize":
        return BasicType_USize()
    elif type_str.startswith('u'):
        width = int(type_str[1:])
        return BasicType_ApUFixed(width)
    elif type_str.startswith('i'):
        width = int(type_str[1:])  
        return BasicType_ApFixed(width)
    elif type_str == "f32":
        return BasicType_Float32()
    elif type_str == "f64":
        return BasicType_Float64()
    elif type_str == "string":
        return BasicType_String()
    else:
        raise ValueError(f"Unknown basic type: {type_str}")


def parse_literal_from_string(literal_str: str) -> Literal:
    """Parse a number literal string into a Literal with proper type"""

    if "'" in literal_str:
        # Handle width-specified literals like "5'b101010"
        width_str, format_and_value = literal_str.split("'", 1)
        width = int(width_str)

        # Parse based on format specifier
        if format_and_value.startswith(('b', 'B')):
            value = int(format_and_value[1:], 2)
        elif format_and_value.startswith(('h', 'H')):
            value = int(format_and_value[1:], 16)
        elif format_and_value.startswith(('o', 'O')):
            value = int(format_and_value[1:], 8)
        elif format_and_value.startswith(('d', 'D')):
            value = int(format_and_value[1:])
        else:
            # No format specifier, treat as decimal
            value = int(format_and_value)

        return Literal(
            LiteralInner_Fixed(value),
            BasicType_ApUFixed(width)
        )
    else:
        # Handle literals without width specification
        if literal_str.startswith(('0x', '0X')):
            value = int(literal_str, 16)
        elif literal_str.startswith(('0b', '0B')):
            value = int(literal_str, 2)
        elif literal_str.startswith(('0o', '0O')):
            value = int(literal_str, 8)
        else:
            value = int(literal_str)

        # Default to 32-bit unsigned
        return Literal(
            LiteralInner_Fixed(value),
            BasicType_ApUFixed(32)
        )