#!/usr/bin/env python3
"""
CADL to C Transpiler

Converts CADL AST to C code for Polygeist compatibility.

Key transformations:
1. Precise bit-width types → Byte-aligned C types (u3 → uint8_t)
2. Bit slicing → Bit masking (idx[2:0] → idx & 0x7)
3. CADL control flow → C control flow

Usage:
    python -m cadl_frontend.transpile_to_c <input.cadl> -o <output.c>
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import CADL frontend
from .parser import parse_proc
from .ast import (
    Proc, Flow, Stmt, Expr,
    AssignStmt, ExprStmt, DoWhileStmt, ReturnStmt, DirectiveStmt,
    LitExpr, IdentExpr, BinaryExpr, BinaryOp, UnaryExpr, UnaryOp, SliceExpr, RangeSliceExpr,
    IndexExpr, IfExpr, CallExpr, SelectExpr, TupleExpr,
    DataType, DataType_Single, DataType_Array,
    BasicType, BasicType_ApFixed, BasicType_ApUFixed, BasicType_Float32, BasicType_Float64,
    Literal, LiteralInner_Fixed, LiteralInner_Float,
)


def _sanitize_identifier(name: str) -> str:
    """Return a C-safe identifier fragment."""
    sanitized = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)
    if not sanitized:
        sanitized = 'value'
    if sanitized[0].isdigit():
        sanitized = f'v_{sanitized}'
    return sanitized


@dataclass
class RegisterReadInfo:
    key: str
    index_expr: Optional[Expr]
    name_hint: str
    type_hints: Set[str] = field(default_factory=set)
    pointer_element_types: Set[str] = field(default_factory=set)
    used_outside_index: bool = False
    parameter_name: Optional[str] = None
    is_pointer_param: bool = False


@dataclass
class RegisterWriteInfo:
    key: str
    index_expr: Expr
    type_hints: Set[str] = field(default_factory=set)


@dataclass
class ScratchpadAlias:
    static_name: str
    register_key: str
    offset_elements: Optional[int]
    element_type: str
    element_size: int
    # For _mem-based aliases: maps static array index -> element offset from register base
    index_to_offset: Optional[Dict[int, int]] = None


@dataclass
class ReturnSpec:
    kind: str  # 'void', 'scalar', 'struct'
    type_name: str
    fields: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class FlowAnalysisData:
    flow_inputs: "OrderedDict[str, DataType]"
    irf_read_infos: Dict[str, RegisterReadInfo]
    irf_write_infos: Dict[str, RegisterWriteInfo]
    used_statics: Set[str]
    scratchpad_aliases: Dict[str, "ScratchpadAlias"]
    needs_mem_pointer: bool
    explicit_return_types: List[List[str]]
    return_spec: ReturnSpec
    math_headers: Set[str]
    var_types: Dict[str, str]


class CTranspiler:
    """Transpile CADL AST to C code (High-Level Mode Only)

    Generates clean C code with pure computational semantics:
    - Drops cpu_mem and _irf base pointers; keeps _mem pointer only when reads remain
    - Converts _irf[rsX] reads to direct value parameters when used as scalars
    - Promotes _irf-backed memory traffic to pointer parameters and rewrites burst traffic
    - Compatible with Polygeist and instruction matching
    """

    def __init__(self):
        self.indent_level = 0
        self.output_lines: List[str] = []
        self.declared_vars: Set[str] = set()
        self.declared_var_stack: List[Set[str]] = []
        self.proc: Optional[Proc] = None
        self.current_flow: Optional[Flow] = None
        self.static_scalar_names: Set[str] = set()
        self.static_array_names: Set[str] = set()
        self.flow_inputs: "OrderedDict[str, DataType]" = OrderedDict()
        self.irf_read_infos: Dict[str, RegisterReadInfo] = {}
        self.irf_write_infos: Dict[str, RegisterWriteInfo] = {}
        self.irf_read_params: Dict[str, str] = {}
        self.used_statics: Set[str] = set()
        self.scratchpad_aliases: Dict[str, ScratchpadAlias] = {}
        self.needs_mem_pointer: bool = False
        self.input_usage: Dict[str, int] = {}
        self.explicit_return_types: List[List[str]] = []
        self.return_spec: ReturnSpec = ReturnSpec("void", "void", [])
        self.result_locals: Dict[str, str] = {}
        self.math_functions: Set[str] = set()
        self.var_types: Dict[str, str] = {}
        self.irf_index_usage: Dict[str, int] = {}
        self.required_headers: Set[str] = {"stdint.h"}
        self.irf_aliases: Dict[str, str] = {}

    def push_decl_scope(self):
        self.declared_var_stack.append(set())

    def pop_decl_scope(self):
        if not self.declared_var_stack:
            return
        scope = self.declared_var_stack.pop()
        for name in scope:
            self.declared_vars.discard(name)

    def register_declaration(self, name: str):
        self.declared_vars.add(name)
        if self.declared_var_stack:
            self.declared_var_stack[-1].add(name)

    def register_alias_key(self, name: str) -> Optional[str]:
        """Return alias key for implicit register value references like r1, r2."""
        if name.startswith("r") and name[1:].isdigit():
            return f"alias:{name}"
        return None

    def _note_type(self, type_name: Optional[str]):
        if type_name == "bool":
            self.required_headers.add("stdbool.h")

    def indent(self) -> str:
        """Current indentation string"""
        return "    " * self.indent_level

    def emit(self, line: str = ""):
        """Emit a line of C code"""
        if line:
            self.output_lines.append(self.indent() + line)
        else:
            self.output_lines.append("")

    def transpile(self, cadl_file: Path) -> str:
        """Main entry point: CADL file → C code"""
        # Read CADL source
        with open(cadl_file, 'r') as f:
            source = f.read()

        # Parse CADL
        proc = parse_proc(source, str(cadl_file))
        self.proc = proc
        self.static_scalar_names = set()
        self.static_array_names = set()
        for name, static in proc.statics.items():
            ty = getattr(static, "ty", None)
            if isinstance(ty, DataType_Single):
                self.static_scalar_names.add(name)
            else:
                self.static_array_names.add(name)

        analyses: Dict[str, FlowAnalysisData] = {}
        self.required_headers = {"stdint.h"}

        for flow in proc.flows.values():
            analysis = self.analyze_flow(flow, proc)
            analyses[flow.name] = analysis
            for header in analysis.math_headers:
                self.required_headers.add(header)
            if analysis.needs_mem_pointer:
                self.required_headers.add("string.h")

        # Generate C code
        self.generate_header()
        for flow in proc.flows.values():
            self.generate_flow(flow, proc, analyses[flow.name])
            self.emit()

        return "\n".join(self.output_lines)

    def generate_header(self):
        """Generate C header"""
        self.emit("// Auto-generated from CADL by cadl-to-c")
        self.emit("// DO NOT EDIT - Regenerate from CADL source")
        self.emit()
        for header in sorted(self.required_headers):
            self.emit(f"#include <{header}>")
        self.emit()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze_flow(self, flow: Flow, proc: Proc):
        """Collect register accesses, types, and auxiliary metadata."""
        self.current_flow = flow
        self.irf_read_infos = {}
        self.irf_write_infos = {}
        self.irf_read_params = {}
        self.used_statics = set()
        self.scratchpad_aliases = {}
        self.needs_mem_pointer = False
        self.input_usage = {}
        self.irf_index_usage = {}
        self.explicit_return_types = []
        self.result_locals = {}
        self.math_functions = set()
        self.var_types = {}
        self.irf_aliases = {}

        self.flow_inputs = OrderedDict(flow.inputs)
        for name, dtype in self.flow_inputs.items():
            self.var_types[name] = self.map_type(dtype)
            self.input_usage[name] = 0
            self.irf_index_usage[name] = 0

        if flow.body:
            for stmt in flow.body:
                self.collect_stmt(stmt)

        # Prune inputs used only as _irf indices
        for name in list(self.flow_inputs.keys()):
            if self.input_usage.get(name, 0) == 0:
                key = f"ident:{name}"
                info = self.irf_read_infos.get(key)
                if info and not info.used_outside_index:
                    self.flow_inputs.pop(name)
                    continue
                if key in self.irf_write_infos:
                    self.flow_inputs.pop(name)

        self.return_spec = self.compute_return_spec()

        for static_name in self.scratchpad_aliases.keys():
            self.used_statics.discard(static_name)

        analysis = FlowAnalysisData(
            flow_inputs=OrderedDict(self.flow_inputs),
            irf_read_infos={k: RegisterReadInfo(
                key=info.key,
                index_expr=info.index_expr,
                name_hint=info.name_hint,
                type_hints=set(info.type_hints),
                pointer_element_types=set(info.pointer_element_types),
                used_outside_index=info.used_outside_index,
                parameter_name=None,
                is_pointer_param=info.is_pointer_param,
            ) for k, info in self.irf_read_infos.items()},
            irf_write_infos={k: RegisterWriteInfo(
                key=info.key,
                index_expr=info.index_expr,
                type_hints=set(info.type_hints),
            ) for k, info in self.irf_write_infos.items()},
            used_statics=set(self.used_statics),
            scratchpad_aliases={name: ScratchpadAlias(
                static_name=alias.static_name,
                register_key=alias.register_key,
                offset_elements=alias.offset_elements,
                element_type=alias.element_type,
                element_size=alias.element_size,
                index_to_offset=dict(alias.index_to_offset) if alias.index_to_offset else None,
            ) for name, alias in self.scratchpad_aliases.items()},
            needs_mem_pointer=self.needs_mem_pointer,
            explicit_return_types=[list(types) for types in self.explicit_return_types],
            return_spec=ReturnSpec(
                kind=self.return_spec.kind,
                type_name=self.return_spec.type_name,
                fields=list(self.return_spec.fields),
            ),
            math_headers=set(self.math_functions),
            var_types=dict(self.var_types),
        )

        return analysis

    def collect_stmt(self, stmt: Stmt):
        if isinstance(stmt, AssignStmt):
            expected_type = None
            lhs_name = None
            pending_mem_assignment: Optional[Tuple[Expr, Optional[str]]] = None

            if isinstance(stmt.lhs, IdentExpr):
                lhs_name = stmt.lhs.name
                if not stmt.is_let and self.proc and lhs_name in self.static_scalar_names:
                    static_obj = self.proc.statics.get(lhs_name)
                    if static_obj:
                        inferred = self.map_static_array_type(static_obj)
                        self.var_types[lhs_name] = inferred
                        expected_type = inferred
                if stmt.is_let or lhs_name not in self.var_types:
                    if stmt.type_annotation:
                        inferred = self.map_type(stmt.type_annotation)
                    else:
                        inferred = self.var_types.get(lhs_name)
                    if inferred:
                        self.var_types[lhs_name] = inferred
                    expected_type = inferred
                else:
                    expected_type = self.var_types.get(lhs_name)
            elif isinstance(stmt.lhs, IndexExpr):
                base = stmt.lhs.expr
                if isinstance(base, IdentExpr):
                    if base.name == "_mem" and stmt.lhs.indices:
                        pending_mem_assignment = (stmt.lhs.indices[0], expected_type)
                    elif self.proc and base.name in self.proc.statics:
                        self.used_statics.add(base.name)
            elif isinstance(stmt.lhs, (SliceExpr, RangeSliceExpr)):
                base = stmt.lhs.expr
                if isinstance(base, IdentExpr):
                    self.used_statics.add(base.name)

            self.collect_expr(stmt.rhs, expected_type=expected_type)
            rhs_type = self.infer_c_type_from_expr(stmt.rhs)

            if lhs_name:
                self._register_irf_alias(lhs_name, stmt.rhs)

            if pending_mem_assignment:
                index_expr, explicit_type = pending_mem_assignment
                pointer_info = self._extract_register_base(index_expr)
                element_type_raw = explicit_type or rhs_type or "uint32_t"
                element_type = self._normalize_pointer_element_type(element_type_raw)
                if pointer_info and element_type:
                    register_key, offset_bytes = pointer_info
                    element_size = self.c_type_size(element_type)
                    if element_size > 0 and offset_bytes % element_size == 0:
                        info = self._ensure_register_info(register_key, index_expr)
                        info.pointer_element_types.add(element_type)
                        self._mark_register_pointer_aliases(register_key, element_type)
                    else:
                        self.needs_mem_pointer = True
                else:
                    self.needs_mem_pointer = True

            self._maybe_record_burst_alias(stmt)
            self._maybe_record_burst_write_alias(stmt)
            self._maybe_record_mem_read_alias(stmt)
            self._maybe_record_mem_write_alias(stmt)

            if lhs_name:
                if lhs_name not in self.var_types:
                    self.var_types[lhs_name] = rhs_type
            elif isinstance(stmt.lhs, IndexExpr):
                base = stmt.lhs.expr
                if isinstance(base, IdentExpr) and base.name == "_irf" and stmt.lhs.indices:
                    key, _ = self.register_key(stmt.lhs.indices[0])
                    info = self.irf_write_infos.setdefault(
                        key,
                        RegisterWriteInfo(key=key, index_expr=stmt.lhs.indices[0]),
                    )
                    if expected_type:
                        info.type_hints.add(expected_type)
                    if rhs_type:
                        info.type_hints.add(rhs_type)
                elif isinstance(base, IdentExpr) and base.name == "_mem" and not pending_mem_assignment:
                    self.needs_mem_pointer = True
            return

        elif isinstance(stmt, ExprStmt):
            self.collect_expr(stmt.expr)

        elif isinstance(stmt, DoWhileStmt):
            for binding in stmt.bindings:
                if binding.init is not None:
                    self.collect_expr(binding.init)
                if binding.next is not None:
                    self.collect_expr(binding.next)
            for inner in stmt.body:
                self.collect_stmt(inner)
            self.collect_expr(stmt.condition)

        elif isinstance(stmt, ReturnStmt):
            types = []
            for expr in stmt.exprs:
                inferred = self.infer_c_type_from_expr(expr)
                types.append(inferred)
                self.collect_expr(expr, expected_type=inferred)
            self.explicit_return_types.append(types)

        elif isinstance(stmt, DirectiveStmt):
            return

        else:
            for field in getattr(stmt, '__dict__', {}).values():
                if isinstance(field, Stmt):
                    self.collect_stmt(field)
                elif isinstance(field, Expr):
                    self.collect_expr(field)
                elif isinstance(field, list):
                    for element in field:
                        if isinstance(element, Stmt):
                            self.collect_stmt(element)
                        elif isinstance(element, Expr):
                            self.collect_expr(element)

    def collect_expr(self, expr: Expr, *, expected_type: Optional[str] = None, context: str = "value"):
        if isinstance(expr, LitExpr):
            return

        if isinstance(expr, IdentExpr):
            name = expr.name
            if name in self.proc.statics:
                self.used_statics.add(name)
            if name in self.flow_inputs and context != "irf_index":
                self.input_usage[name] = self.input_usage.get(name, 0) + 1
                info = self.irf_read_infos.get(f"ident:{name}")
                if info:
                    info.used_outside_index = True
                return
            if name in self.var_types:
                return
            alias_key = self.register_alias_key(name)
            if alias_key:
                info = self.irf_read_infos.setdefault(
                    alias_key,
                    RegisterReadInfo(
                        key=alias_key,
                        index_expr=None,
                        name_hint=name,
                    ),
                )
                if expected_type:
                    info.type_hints.add(expected_type)
                info.used_outside_index = True
                return
            return

        if isinstance(expr, UnaryExpr):
            self.collect_expr(expr.operand, expected_type=expected_type)
            return

        if isinstance(expr, BinaryExpr):
            self.collect_expr(expr.left)
            self.collect_expr(expr.right)
            return

        if isinstance(expr, SliceExpr):
            self.collect_expr(expr.expr)
            if expr.start:
                self.collect_expr(expr.start)
            if expr.end:
                self.collect_expr(expr.end)
            return

        if isinstance(expr, RangeSliceExpr):
            self.collect_expr(expr.expr)
            if expr.start:
                self.collect_expr(expr.start)
            if expr.length:
                self.collect_expr(expr.length)
            return

        if isinstance(expr, IndexExpr):
            base = expr.expr
            if isinstance(base, IdentExpr):
                if base.name == "_irf" and expr.indices:
                    key, hint = self.register_key(expr.indices[0])
                    info = self.irf_read_infos.setdefault(
                        key,
                        RegisterReadInfo(key=key, index_expr=expr.indices[0], name_hint=hint),
                    )
                    if expected_type:
                        info.type_hints.add(expected_type)
                    self.collect_expr(expr.indices[0], context="irf_index")
                    return
                if base.name == "_mem":
                    pointer_info = self._extract_register_base(expr.indices[0])
                    inferred_type_raw = expected_type or self.infer_c_type_from_expr(expr)
                    inferred_type = self._normalize_pointer_element_type(inferred_type_raw)
                    if pointer_info and inferred_type:
                        register_key, offset_bytes = pointer_info
                        element_size = self.c_type_size(inferred_type)
                        if element_size > 0 and offset_bytes % element_size == 0:
                            info = self._ensure_register_info(register_key, expr.indices[0])
                            info.pointer_element_types.add(inferred_type)
                            return
                    self.needs_mem_pointer = True
                if base.name in self.proc.statics:
                    self.used_statics.add(base.name)
            for index_expr in expr.indices:
                self.collect_expr(index_expr)
            return

        if isinstance(expr, CallExpr):
            builtin_headers = {
                "abs": "stdlib.h",
                "labs": "stdlib.h",
                "llabs": "stdlib.h",
                "sqrt": "math.h",
                "exp": "math.h",
                "log": "math.h",
                "pow": "math.h",
                "floor": "math.h",
                "ceil": "math.h",
            }
            header = builtin_headers.get(expr.name)
            if header:
                self.math_functions.add(header)
            for arg in expr.args:
                self.collect_expr(arg)
            return

        if isinstance(expr, SelectExpr):
            for condition, value in expr.arms:
                self.collect_expr(condition)
                self.collect_expr(value, expected_type=expected_type)
            self.collect_expr(expr.default, expected_type=expected_type)
            return

        if isinstance(expr, IfExpr):
            self.collect_expr(expr.condition)
            self.collect_expr(expr.then_branch, expected_type=expected_type)
            self.collect_expr(expr.else_branch, expected_type=expected_type)
            return

        if isinstance(expr, TupleExpr):
            for element in expr.elements:
                self.collect_expr(element)
            return

        for field in getattr(expr, '__dict__', {}).values():
            if isinstance(field, Expr):
                self.collect_expr(field)
            elif isinstance(field, list):
                for element in field:
                    if isinstance(element, Expr):
                        self.collect_expr(element)

    def _register_irf_alias(self, lhs_name: Optional[str], rhs: Expr):
        if not lhs_name:
            return
        key = self._extract_irf_read_key(rhs)
        if key:
            self.irf_aliases[lhs_name] = key
        else:
            self.irf_aliases.pop(lhs_name, None)

    def _ensure_register_info(self, key: str, index_expr: Optional[Expr] = None) -> RegisterReadInfo:
        info = self.irf_read_infos.get(key)
        if info:
            return info
        if key.startswith("ident:"):
            name_hint = key.split(":", 1)[1]
        elif key.startswith("literal:"):
            name_hint = f"irf_{key.split(':', 1)[1]}"
        else:
            name_hint = _sanitize_identifier(key)
        info = RegisterReadInfo(key=key, index_expr=index_expr, name_hint=name_hint)
        self.irf_read_infos[key] = info
        return info

    def _extract_irf_read_key(self, expr: Expr) -> Optional[str]:
        if isinstance(expr, IndexExpr) and isinstance(expr.expr, IdentExpr):
            if expr.expr.name == "_irf" and expr.indices:
                key, _ = self.register_key(expr.indices[0])
                return key
        return None

    def _maybe_record_burst_alias(self, stmt: AssignStmt):
        if not isinstance(stmt.lhs, RangeSliceExpr):
            return
        lhs_expr = stmt.lhs.expr
        if not isinstance(lhs_expr, IdentExpr):
            return
        static_name = lhs_expr.name
        if not self.proc or static_name not in self.proc.statics:
            return
        rhs = stmt.rhs
        if not isinstance(rhs, RangeSliceExpr):
            return
        if not isinstance(rhs.expr, IdentExpr) or rhs.expr.name != "_burst_read":
            return
        pointer_info = self._extract_register_base(rhs.start)
        if not pointer_info:
            return
        register_key, offset_bytes = pointer_info
        static_obj = self.proc.statics.get(static_name)
        if not static_obj:
            return
        element_type = self.map_static_array_type(static_obj)
        element_size = self.c_type_size(element_type)
        if element_size <= 0:
            return
        if offset_bytes % element_size != 0:
            return
        offset_elements = offset_bytes // element_size
        alias = ScratchpadAlias(
            static_name=static_name,
            register_key=register_key,
            offset_elements=offset_elements,
            element_type=element_type,
            element_size=element_size,
        )
        self.scratchpad_aliases[static_name] = alias
        info = self.irf_read_infos.get(register_key)
        if info:
            info.pointer_element_types.add(element_type)
            self._mark_register_pointer_aliases(register_key, element_type)
            self._mark_register_pointer_aliases(register_key, element_type)

    def _maybe_record_burst_write_alias(self, stmt: AssignStmt):
        if not isinstance(stmt.lhs, RangeSliceExpr):
            return
        lhs_expr = stmt.lhs.expr
        if not isinstance(lhs_expr, IdentExpr) or lhs_expr.name != "_burst_write":
            return
        rhs = stmt.rhs
        if not isinstance(rhs, RangeSliceExpr):
            return
        rhs_expr = rhs.expr
        if not isinstance(rhs_expr, IdentExpr):
            return
        static_name = rhs_expr.name
        if not self.proc or static_name not in self.proc.statics:
            return
        pointer_info = self._extract_register_base(stmt.lhs.start)
        if not pointer_info:
            return
        register_key, offset_bytes = pointer_info
        static_obj = self.proc.statics.get(static_name)
        if not static_obj:
            return
        element_type = self.map_static_array_type(static_obj)
        element_size = self.c_type_size(element_type)
        if element_size <= 0:
            return
        if offset_bytes % element_size != 0:
            return
        offset_elements = offset_bytes // element_size
        alias = ScratchpadAlias(
            static_name=static_name,
            register_key=register_key,
            offset_elements=offset_elements,
            element_type=element_type,
            element_size=element_size,
        )
        self.scratchpad_aliases[static_name] = alias
        info = self.irf_read_infos.get(register_key)
        if info:
            info.pointer_element_types.add(element_type)

    def _maybe_record_mem_read_alias(self, stmt: AssignStmt):
        """Detect pattern: static[i] = _mem[addr + offset] and build alias mapping.

        Example:
            vec[0] = _mem[addr + 64];
            vec[1] = _mem[addr + 68];

        This builds a mapping from vec to the register backing addr.
        """
        # LHS must be IndexExpr into a static array
        if not isinstance(stmt.lhs, IndexExpr):
            return
        lhs_base = stmt.lhs.expr
        if not isinstance(lhs_base, IdentExpr):
            return
        static_name = lhs_base.name
        if not self.proc or static_name not in self.proc.statics:
            return
        if not stmt.lhs.indices:
            return
        static_index = self.try_eval_constant(stmt.lhs.indices[0])
        if static_index is None:
            return

        # RHS must be _mem[addr + offset]
        if not isinstance(stmt.rhs, IndexExpr):
            return
        rhs_base = stmt.rhs.expr
        if not isinstance(rhs_base, IdentExpr) or rhs_base.name != "_mem":
            return
        if not stmt.rhs.indices:
            return

        pointer_info = self._extract_register_base(stmt.rhs.indices[0])
        if not pointer_info:
            return
        register_key, offset_bytes = pointer_info

        static_obj = self.proc.statics.get(static_name)
        if not static_obj:
            return
        element_type = self.map_static_array_type(static_obj)
        element_size = self.c_type_size(element_type)
        if element_size <= 0:
            return
        if offset_bytes % element_size != 0:
            return
        offset_elements = offset_bytes // element_size

        # Get or create alias for this static array
        if static_name in self.scratchpad_aliases:
            alias = self.scratchpad_aliases[static_name]
            # Verify it's the same register
            if alias.register_key != register_key:
                return  # Mixed register sources, can't alias
            if alias.index_to_offset is None:
                alias.index_to_offset = {}
            alias.index_to_offset[static_index] = offset_elements
        else:
            alias = ScratchpadAlias(
                static_name=static_name,
                register_key=register_key,
                offset_elements=None,  # Will use index_to_offset instead
                element_type=element_type,
                element_size=element_size,
                index_to_offset={static_index: offset_elements},
            )
            self.scratchpad_aliases[static_name] = alias

        # Mark the register as needing pointer type
        info = self._ensure_register_info(register_key, stmt.rhs.indices[0])
        info.pointer_element_types.add(element_type)
        self._mark_register_pointer_aliases(register_key, element_type)

    def _maybe_record_mem_write_alias(self, stmt: AssignStmt):
        """Detect pattern: _mem[addr + offset] = static[i] and build alias mapping.

        Example:
            _mem[out_addr] = result[0];
            _mem[out_addr + 4] = result[1];

        This builds a mapping from result to the register backing out_addr.
        """
        # LHS must be _mem[addr + offset]
        if not isinstance(stmt.lhs, IndexExpr):
            return
        lhs_base = stmt.lhs.expr
        if not isinstance(lhs_base, IdentExpr) or lhs_base.name != "_mem":
            return
        if not stmt.lhs.indices:
            return

        pointer_info = self._extract_register_base(stmt.lhs.indices[0])
        if not pointer_info:
            return
        register_key, offset_bytes = pointer_info

        # RHS must be IndexExpr into a static array
        if not isinstance(stmt.rhs, IndexExpr):
            return
        rhs_base = stmt.rhs.expr
        if not isinstance(rhs_base, IdentExpr):
            return
        static_name = rhs_base.name
        if not self.proc or static_name not in self.proc.statics:
            return
        if not stmt.rhs.indices:
            return
        static_index = self.try_eval_constant(stmt.rhs.indices[0])
        if static_index is None:
            return

        static_obj = self.proc.statics.get(static_name)
        if not static_obj:
            return
        element_type = self.map_static_array_type(static_obj)
        element_size = self.c_type_size(element_type)
        if element_size <= 0:
            return
        if offset_bytes % element_size != 0:
            return
        offset_elements = offset_bytes // element_size

        # Get or create alias for this static array
        if static_name in self.scratchpad_aliases:
            alias = self.scratchpad_aliases[static_name]
            # Verify it's the same register
            if alias.register_key != register_key:
                return  # Mixed register sources, can't alias
            if alias.index_to_offset is None:
                alias.index_to_offset = {}
            alias.index_to_offset[static_index] = offset_elements
        else:
            alias = ScratchpadAlias(
                static_name=static_name,
                register_key=register_key,
                offset_elements=None,  # Will use index_to_offset instead
                element_type=element_type,
                element_size=element_size,
                index_to_offset={static_index: offset_elements},
            )
            self.scratchpad_aliases[static_name] = alias

        # Mark the register as needing pointer type
        info = self._ensure_register_info(register_key, stmt.lhs.indices[0])
        info.pointer_element_types.add(element_type)
        self._mark_register_pointer_aliases(register_key, element_type)

    def _extract_register_base(self, expr: Expr) -> Optional[Tuple[str, int]]:
        if isinstance(expr, IdentExpr):
            key = self.irf_aliases.get(expr.name)
            if key:
                return key, 0
        if isinstance(expr, IndexExpr) and isinstance(expr.expr, IdentExpr):
            if expr.expr.name == "_irf" and expr.indices:
                key, _ = self.register_key(expr.indices[0])
                return key, 0
        if isinstance(expr, UnaryExpr):
            value = self._extract_register_base(expr.operand)
            if value and expr.op == UnaryOp.NEG:
                key, offset = value
                return key, -offset
        if isinstance(expr, BinaryExpr):
            if expr.op == BinaryOp.ADD:
                left = self._extract_register_base(expr.left)
                right_const = self.try_eval_constant(expr.right)
                if left and right_const is not None:
                    key, offset = left
                    return key, offset + right_const
                right = self._extract_register_base(expr.right)
                left_const = self.try_eval_constant(expr.left)
                if right and left_const is not None:
                    key, offset = right
                    return key, offset + left_const
            elif expr.op == BinaryOp.SUB:
                left = self._extract_register_base(expr.left)
                right_const = self.try_eval_constant(expr.right)
                if left and right_const is not None:
                    key, offset = left
                    return key, offset - right_const
        return None

    def _format_alias_index(self, alias: ScratchpadAlias, index_code: str, static_index: Optional[int] = None) -> str:
        # If we have a per-index mapping, use it for constant indices
        if alias.index_to_offset is not None and static_index is not None:
            if static_index in alias.index_to_offset:
                return str(alias.index_to_offset[static_index])
        # For variable indices with index_to_offset, try to compute base offset
        # If all entries are offset = base + static_index, we can use base + index_code
        if alias.index_to_offset is not None and len(alias.index_to_offset) > 0:
            # Check if mapping is linear (offset = base + static_index)
            base_offset = None
            is_linear = True
            for idx, offset in alias.index_to_offset.items():
                computed_base = offset - idx
                if base_offset is None:
                    base_offset = computed_base
                elif base_offset != computed_base:
                    is_linear = False
                    break
            if is_linear and base_offset is not None:
                if base_offset == 0:
                    return index_code
                if index_code in {"0", "0u", "0U", "0UL", "0uL"}:
                    return str(base_offset)
                return f"{base_offset} + {index_code}"
        # Fall back to uniform offset
        offset = alias.offset_elements or 0
        if offset == 0:
            return index_code
        if index_code in {"0", "0u", "0U", "0UL", "0uL"}:
            return str(offset)
        return f"{offset} + {index_code}"

    def _generate_mem_index(self, index_expr: Expr, full_expr: IndexExpr) -> Optional[str]:
        pointer_info = self._extract_register_base(index_expr)
        if not pointer_info:
            return None
        register_key, offset_bytes = pointer_info
        info = self.irf_read_infos.get(register_key)
        if not info:
            return None
        param_name = self.irf_read_params.get(register_key)
        if not param_name:
            return None
        element_type_candidates = set(info.pointer_element_types)
        if not element_type_candidates:
            inferred = self.infer_c_type_from_expr(full_expr)
            if inferred:
                element_type_candidates.add(inferred)
        elem_type = self.choose_type_hint(element_type_candidates) if element_type_candidates else "uint32_t"
        element_size = self.c_type_size(elem_type)
        if element_size <= 0 or offset_bytes % element_size != 0:
            return None
        offset_elements = offset_bytes // element_size
        index_code = str(offset_elements)
        info.pointer_element_types.add(elem_type)
        self._mark_register_pointer_aliases(register_key, elem_type)
        return f"{param_name}[{index_code}]"

    def register_key(self, index_expr: Expr) -> Tuple[str, str]:
        if isinstance(index_expr, IdentExpr):
            name = index_expr.name
            return f"ident:{name}", name
        if isinstance(index_expr, LitExpr):
            literal = index_expr.literal.lit
            if isinstance(literal, LiteralInner_Fixed):
                return f"literal:{literal.value}", f"irf_{literal.value}"
            if isinstance(literal, LiteralInner_Float):
                return f"literal:{literal.value}", f"irf_{literal.value}"
        return f"expr:{_sanitize_identifier(str(index_expr))}", _sanitize_identifier(str(index_expr))

    def _merge_numeric_types(self, left: Optional[str], right: Optional[str]) -> str:
        if not left:
            return right or "uint32_t"
        if not right:
            return left
        if left == right:
            return left
        if left.startswith("uint") and right.startswith("uint"):
            return left if self._type_width(left) >= self._type_width(right) else right
        if left.startswith("int") and right.startswith("int"):
            return left if self._type_width(left) >= self._type_width(right) else right
        if "double" in (left, right):
            return "double"
        if "float" in (left, right):
            return "float"
        if left.startswith("int") or right.startswith("int"):
            return "int32_t"
        return "uint32_t"

    def compute_return_spec(self) -> ReturnSpec:
        if self.explicit_return_types:
            types = self.explicit_return_types[0]
            if not types:
                return ReturnSpec("void", "void", [])
            if len(types) == 1:
                return ReturnSpec("scalar", types[0], [])
            fields = [(f"value{idx}", ty) for idx, ty in enumerate(types)]
            struct_name = f"{self.current_flow.name}_result_t"
            return ReturnSpec("struct", struct_name, fields)

        if not self.irf_write_infos:
            return ReturnSpec("void", "void", [])

        if len(self.irf_write_infos) == 1:
            info = next(iter(self.irf_write_infos.values()))
            ty = self.choose_type_hint(info.type_hints)
            return ReturnSpec("scalar", ty, [])

        fields = []
        for idx, info in enumerate(self.irf_write_infos.values()):
            fields.append((f"value{idx}", self.choose_type_hint(info.type_hints)))
        struct_name = f"{self.current_flow.name}_result_t"
        return ReturnSpec("struct", struct_name, fields)

    def choose_type_hint(self, hints: Set[str], default: str = "uint32_t") -> str:
        for hint in sorted(hints):
            if hint:
                self._note_type(hint)
                return hint
        self._note_type(default)
        return default

    def unique_param_name(self, candidate: str, existing: Set[str]) -> str:
        name = candidate
        suffix = 1
        while name in existing:
            name = f"{candidate}_{suffix}"
            suffix += 1
        return name

    def result_local_name(self, key: str) -> str:
        if key.startswith("ident:"):
            return f"{key.split(':', 1)[1]}_result"
        if key.startswith("literal:"):
            return f"reg_{key.split(':', 1)[1]}_result"
        return f"result_{_sanitize_identifier(key)}"

    def _type_width(self, type_name: str) -> int:
        digits = ''.join(ch for ch in type_name if ch.isdigit())
        return int(digits) if digits else 32

    def _normalize_pointer_element_type(self, type_name: Optional[str]) -> str:
        base = (type_name or "uint32_t").strip()
        if base.startswith("const "):
            base = base[len("const "):].strip()
        if base.endswith("*"):
            base = base.rstrip("*").strip()
        if base == "bool":
            return "uint32_t"
        if base.startswith("uint"):
            width = self._type_width(base)
            return base if width >= 32 else "uint32_t"
        if base.startswith("int"):
            width = self._type_width(base)
            return base if width >= 32 else "int32_t"
        return base or "uint32_t"

    def _mark_register_pointer_aliases(self, register_key: str, element_type: str):
        pointer_type = f"{element_type} *"
        for alias_name, key in list(self.irf_aliases.items()):
            if key == register_key:
                current = self.var_types.get(alias_name)
                if current != pointer_type:
                    self.var_types[alias_name] = pointer_type

    def c_type_size(self, type_name: str) -> int:
        base = type_name.strip()
        if base.startswith("const "):
            base = base[len("const "):]
        if base.endswith("*"):
            base = base[:-1].strip()
        mapping = {
            "uint8_t": 1,
            "int8_t": 1,
            "bool": 1,
            "uint16_t": 2,
            "int16_t": 2,
            "uint32_t": 4,
            "int32_t": 4,
            "float": 4,
            "uint64_t": 8,
            "int64_t": 8,
            "double": 8,
        }
        return mapping.get(base, 4)

    def try_emit_for_loop(self, stmt: DoWhileStmt) -> bool:
        if not stmt.bindings or stmt.condition is None:
            return False

        init_parts: List[str] = []
        update_parts: List[str] = []
        declarations: List[Tuple[str, str]] = []
        step_hints: Dict[str, Tuple[BinaryOp, int]] = {}

        for binding in stmt.bindings:
            if binding.init is None or binding.next is None:
                return False
            var_name = binding.id
            if var_name not in self.declared_vars:
                decl_type = self.map_type(DataType_Single(binding.ty))
                declarations.append((decl_type, var_name))
            init_expr = self.generate_expr(binding.init)
            init_parts.append(f"{var_name} = {init_expr}")

            update_expr = self.format_loop_update(binding, stmt.body, step_hints)
            if update_expr is None:
                return False
            update_parts.append(update_expr)

        condition = self.rewrite_loop_condition(stmt.condition, stmt.bindings, stmt.body, step_hints)

        for decl_type, var_name in declarations:
            self.emit(f"{decl_type} {var_name};")
            self.register_declaration(var_name)

        init_clause = ", ".join(init_parts)
        update_clause = ", ".join(update_parts)
        self.emit(f"for ({init_clause}; {condition}; {update_clause}) {{")
        self.indent_level += 1
        self.push_decl_scope()
        for body_stmt in stmt.body:
            self.generate_stmt(body_stmt)
        self.pop_decl_scope()
        self.indent_level -= 1
        self.emit("}")
        return True

    def format_loop_update(self, binding, body_stmts: List[Stmt], step_hints: Dict[str, Tuple[BinaryOp, int]]) -> Optional[str]:
        var_name = binding.id
        next_expr = binding.next
        if isinstance(next_expr, BinaryExpr):
            if isinstance(next_expr.left, IdentExpr) and next_expr.left.name == var_name:
                step_str = self.generate_expr(next_expr.right)
                if next_expr.op == BinaryOp.ADD:
                    step_val = self.try_eval_constant(next_expr.right)
                    if step_val is not None:
                        step_hints[var_name] = (BinaryOp.ADD, step_val)
                    if self._expr_is_literal_one(next_expr.right):
                        return f"++{var_name}"
                    return f"{var_name} += {step_str}"
                if next_expr.op == BinaryOp.SUB:
                    step_val = self.try_eval_constant(next_expr.right)
                    if step_val is not None:
                        step_hints[var_name] = (BinaryOp.SUB, step_val)
                    if self._expr_is_literal_one(next_expr.right):
                        return f"--{var_name}"
                    return f"{var_name} -= {step_str}"
        if isinstance(next_expr, IdentExpr):
            step_info = self._find_step_in_body(body_stmts, var_name, next_expr.name)
            if step_info is not None:
                op, step_val = step_info
                step_hints[var_name] = step_info
                if op == BinaryOp.ADD:
                    if step_val == 1:
                        return f"++{var_name}"
                    return f"{var_name} += {step_val}"
                if op == BinaryOp.SUB:
                    if step_val == 1:
                        return f"--{var_name}"
                    return f"{var_name} -= {step_val}"
        return None

    def _expr_is_literal_one(self, expr: Expr) -> bool:
        if isinstance(expr, LitExpr):
            inner = expr.literal.lit
            if isinstance(inner, LiteralInner_Fixed):
                return inner.value == 1
        return False

    def _find_step_in_body(self, body_stmts: List[Stmt], var_name: str, temp_name: str) -> Optional[Tuple[BinaryOp, int]]:
        for stmt in body_stmts:
            if isinstance(stmt, AssignStmt):
                if isinstance(stmt.lhs, IdentExpr) and stmt.lhs.name == temp_name:
                    rhs = stmt.rhs
                    if isinstance(rhs, BinaryExpr):
                        if isinstance(rhs.left, IdentExpr) and rhs.left.name == var_name:
                            step_val = self.try_eval_constant(rhs.right)
                            if step_val is not None:
                                return rhs.op, step_val
        return None

    def rewrite_loop_condition(self, condition: Expr, bindings: List, body_stmts: List[Stmt], step_hints: Dict[str, Tuple[BinaryOp, int]]) -> str:
        if not bindings:
            return self.generate_expr(condition)
        primary = bindings[0]
        next_name = None
        if isinstance(primary.next, IdentExpr):
            next_name = primary.next.name
        if isinstance(condition, BinaryExpr) and next_name:
            if isinstance(condition.left, IdentExpr) and condition.left.name == next_name:
                if condition.op == BinaryOp.LT:
                    right = self.generate_expr(condition.right)
                    return f"{primary.id} < {right}"
                if condition.op == BinaryOp.LE:
                    right_val = self.try_eval_constant(condition.right)
                    if right_val is not None:
                        step_info = step_hints.get(primary.id) or self._find_step_in_body(body_stmts, primary.id, next_name)
                        if step_info:
                            _, step_val = step_info
                            adjusted = right_val - step_val
                            return f"{primary.id} <= {adjusted}"

        # Handle case where next is a BinaryExpr like (i + 1) and condition uses same pattern
        # e.g., with i = (0, i + 1) do {...} while (i + 1 < 16)
        # In CADL do-while semantics, body runs first then condition is checked
        # So "while (i + 1 < 16)" means loop while next value < 16
        # For C for-loop (checks condition before body), we need "i < 16"
        if isinstance(condition, BinaryExpr) and isinstance(primary.next, BinaryExpr):
            next_expr = primary.next
            cond_left = condition.left
            # Check if condition.left matches the next expression pattern
            if isinstance(cond_left, BinaryExpr):
                # Check if both are "var op const" patterns with same var and op
                if (isinstance(next_expr.left, IdentExpr) and
                    isinstance(cond_left.left, IdentExpr) and
                    next_expr.left.name == primary.id and
                    cond_left.left.name == primary.id and
                    next_expr.op == cond_left.op):
                    # Check if the step values match
                    next_step = self.try_eval_constant(next_expr.right)
                    cond_step = self.try_eval_constant(cond_left.right)
                    if next_step is not None and cond_step is not None and next_step == cond_step:
                        # Rewrite (i + step) < bound to i < bound for LT
                        # Rewrite (i + step) <= bound to i <= (bound - step) for LE
                        if condition.op == BinaryOp.LT:
                            right = self.generate_expr(condition.right)
                            return f"{primary.id} < {right}"
                        if condition.op == BinaryOp.LE:
                            right_val = self.try_eval_constant(condition.right)
                            if right_val is not None:
                                adjusted = right_val - next_step
                                return f"{primary.id} <= {adjusted}"

        return self.generate_expr(condition)

    def generate_flow(self, flow: Flow, proc: Proc, analysis: FlowAnalysisData):
        """Generate C function from CADL flow

        High-level mode only:
        - Static arrays become parameters when truly external
        - _irf[rsX] reads stay as value params unless memory traffic requires pointers
        - No cpu_mem or _irf base pointers (retain _mem pointer when reads remain)
        - Burst operations fold into pointer arithmetic tied to register bases
        """
        self.declared_vars = set()
        self.declared_var_stack = []
        self.flow_inputs = analysis.flow_inputs
        self.irf_read_infos = analysis.irf_read_infos
        self.irf_write_infos = analysis.irf_write_infos
        self.used_statics = analysis.used_statics
        self.scratchpad_aliases = analysis.scratchpad_aliases
        self.needs_mem_pointer = analysis.needs_mem_pointer
        self.explicit_return_types = analysis.explicit_return_types
        self.return_spec = analysis.return_spec
        self.math_functions = analysis.math_headers
        self.var_types = analysis.var_types
        self.irf_read_params = {}

        func_name = flow.name
        params: List[str] = []
        used_param_names: Set[str] = set()
        local_static_decls: List[str] = []

        self.push_decl_scope()

        for static_name in sorted(self.used_statics):
            used_param_names.add(static_name)
            if static_name in self.scratchpad_aliases:
                continue
            static_obj = proc.statics.get(static_name)
            if not static_obj:
                continue
            decl_line, type_hint = self.make_static_local_declaration(static_name, static_obj)
            if decl_line:
                local_static_decls.append(decl_line)
                self.register_declaration(static_name)
                if type_hint:
                    self.var_types[static_name] = type_hint

        register_param_order = [
            ("ident:rs1", "rs1"),
            ("ident:rs2", "rs2"),
        ]

        for key, base_name in register_param_order:
            info = self.irf_read_infos.get(key)
            if not info:
                info = RegisterReadInfo(key=key, index_expr=None, name_hint=base_name)
                self.irf_read_infos[key] = info
            sanitized = _sanitize_identifier(base_name) or "reg"
            if info.pointer_element_types:
                elem_type = self.choose_type_hint(info.pointer_element_types)
                candidate = sanitized
                info.is_pointer_param = True
            else:
                c_type = self.choose_type_hint(info.type_hints)
                candidate = f"{sanitized}_value"
                info.is_pointer_param = False
            param_name = self.unique_param_name(candidate, used_param_names)
            info.parameter_name = param_name
            self.irf_read_params[key] = param_name
            if info.is_pointer_param:
                decl = f"{elem_type} *{param_name}"
                self.var_types[param_name] = f"{elem_type} *"
            else:
                decl = f"{c_type} {param_name}"
                self.var_types[param_name] = c_type
            params.append(decl)
            used_param_names.add(param_name)
            self.register_declaration(param_name)

        params_str = ", ".join(params)
        return_type = self.return_spec.type_name

        if self.return_spec.kind == "struct":
            self.emit(f"typedef struct {{")
            self.indent_level += 1
            for field_name, field_type in self.return_spec.fields:
                self.emit(f"{field_type} {field_name};")
            self.indent_level -= 1
            self.emit(f"}} {self.return_spec.type_name};")
            self.emit()

        self.emit(f"{return_type} {func_name}({params_str}) {{")
        self.indent_level += 1

        if local_static_decls:
            for decl in local_static_decls:
                self.emit(decl)
            self.emit()

        self.result_locals = {}
        if not self.explicit_return_types and self.return_spec.kind != "void":
            if self.return_spec.kind == "scalar":
                key = next(iter(self.irf_write_infos.keys()))
                local_name = self.result_local_name(key)
                self.result_locals[key] = local_name
                self.emit(f"{self.return_spec.type_name} {local_name} = 0;")
                self.register_declaration(local_name)
            else:
                for idx, key in enumerate(self.irf_write_infos.keys()):
                    field_name, field_type = self.return_spec.fields[idx]
                    self.result_locals[key] = field_name
                    self.emit(f"{field_type} {field_name} = 0;")
                    self.register_declaration(field_name)

        if flow.body:
            for stmt in flow.body:
                self.generate_stmt(stmt)

        if not self.explicit_return_types:
            if self.return_spec.kind == "void":
                self.emit("return;")
            elif self.return_spec.kind == "scalar":
                key = next(iter(self.irf_write_infos.keys()))
                local_name = self.result_locals.get(key, self.result_local_name(key))
                self.emit(f"return {local_name};")
            elif self.return_spec.kind == "struct":
                init_fields = []
                for field_name, _ in self.return_spec.fields:
                    init_fields.append(f".{field_name} = {field_name}")
                init = ", ".join(init_fields)
                self.emit(f"return ({self.return_spec.type_name}){{{init}}};")

        self.indent_level -= 1
        self.emit("}")
        self.pop_decl_scope()

    def map_static_array_type(self, static_obj) -> str:
        """Map CADL static array to C element type"""
        # Get the array element type from the static object
        if isinstance(static_obj.ty, DataType_Array):
            result = self.map_basic_type(static_obj.ty.element_type)
            self._note_type(result)
            return result
        elif isinstance(static_obj.ty, DataType_Single):
            result = self.map_basic_type(static_obj.ty.basic_type)
            self._note_type(result)
            return result
        self._note_type("uint8_t")
        return "uint8_t"  # Fallback

    def make_static_local_declaration(self, name: str, static_obj) -> Tuple[Optional[str], Optional[str]]:
        ty = getattr(static_obj, "ty", None)
        if isinstance(ty, DataType_Array):
            elem_type = self.map_basic_type(ty.element_type)
            dims = ty.dimensions if ty.dimensions else [1]
            dims_str = ''.join(f"[{dim}]" for dim in dims)
            return f"{elem_type} {name}{dims_str};", None
        if isinstance(ty, DataType_Single):
            base_type = self.map_basic_type(ty.basic_type)
            return f"{base_type} {name} = 0;", base_type
        return None, None

    def map_basic_type(self, basic_type: BasicType) -> str:
        """Map BasicType to C type (for array elements)"""
        if isinstance(basic_type, BasicType_ApUFixed):
            width = basic_type.width
            if width <= 8:
                return "uint8_t"
            elif width <= 16:
                return "uint16_t"
            elif width <= 32:
                return "uint32_t"
            else:
                return "uint64_t"
        elif isinstance(basic_type, BasicType_ApFixed):
            width = basic_type.width
            if width <= 8:
                return "int8_t"
            elif width <= 16:
                return "int16_t"
            elif width <= 32:
                return "int32_t"
            else:
                return "int64_t"
        elif isinstance(basic_type, BasicType_Float32):
            return "float"
        elif isinstance(basic_type, BasicType_Float64):
            return "double"
        self._note_type("uint8_t")
        return "uint8_t"  # Fallback

    def infer_c_type_from_expr(self, expr) -> str:
        if isinstance(expr, LitExpr):
            lit = expr.literal.lit
            if isinstance(lit, LiteralInner_Float):
                return "double"
            if isinstance(lit, LiteralInner_Fixed):
                value = lit.value
                if value < 0:
                    return "int32_t"
                if value <= 0xFF:
                    return "uint8_t"
                if value <= 0xFFFF:
                    return "uint16_t"
                if value <= 0xFFFFFFFF:
                    return "uint32_t"
                return "uint64_t"

        if isinstance(expr, IdentExpr):
            name = expr.name
            alias_key = self.register_alias_key(name)
            if alias_key:
                info = self.irf_read_infos.get(alias_key)
                if info and info.type_hints:
                    return self.choose_type_hint(info.type_hints)
            return self.var_types.get(name, "uint32_t")

        if isinstance(expr, IndexExpr):
            if isinstance(expr.expr, IdentExpr) and expr.expr.name == "_irf" and expr.indices:
                key, _ = self.register_key(expr.indices[0])
                info = self.irf_read_infos.get(key)
                if info and info.type_hints:
                    return self.choose_type_hint(info.type_hints)
                write_info = self.irf_write_infos.get(key)
                if write_info and write_info.type_hints:
                    return self.choose_type_hint(write_info.type_hints)
                return "uint32_t"
            if isinstance(expr.expr, IdentExpr):
                array_name = expr.expr.name
                if self.proc and array_name in self.proc.statics:
                    static_obj = self.proc.statics[array_name]
                    return self.map_static_array_type(static_obj)
            return "uint32_t"

        if isinstance(expr, BinaryExpr):
            left = self.infer_c_type_from_expr(expr.left)
            right = self.infer_c_type_from_expr(expr.right)
            return self._merge_numeric_types(left, right)

        if isinstance(expr, UnaryExpr):
            return self.infer_c_type_from_expr(expr.operand)

        if isinstance(expr, SliceExpr):
            inner_type = self.infer_c_type_from_expr(expr.expr)
            return inner_type

        if isinstance(expr, IfExpr):
            return self.infer_c_type_from_expr(expr.then_branch)

        if isinstance(expr, SelectExpr):
            result_type: Optional[str] = None
            for _, value in expr.arms:
                result_type = self._merge_numeric_types(
                    result_type, self.infer_c_type_from_expr(value)
                )
            result_type = self._merge_numeric_types(
                result_type, self.infer_c_type_from_expr(expr.default)
            )
            return result_type or "uint32_t"

        if isinstance(expr, CallExpr):
            if expr.name in {"sqrt", "exp", "log", "pow", "ceil", "floor"}:
                return "double"
            if expr.name in {"abs", "labs", "llabs"}:
                return "int32_t"
            return "uint32_t"

        if isinstance(expr, TupleExpr) and expr.elements:
            return self.infer_c_type_from_expr(expr.elements[0])

        return "uint32_t"

    def generate_stmt(self, stmt: Stmt):
        """Generate statement"""
        if isinstance(stmt, AssignStmt):
            self.generate_assign(stmt)
        elif isinstance(stmt, ExprStmt):
            # Expression statements (function calls, etc.)
            expr_str = self.generate_expr(stmt.expr)
            self.emit(f"{expr_str};")
        elif isinstance(stmt, DoWhileStmt):
            self.generate_dowhile(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.generate_return(stmt)
        elif isinstance(stmt, DirectiveStmt):
            # Skip directives - they're MLIR/HLS annotations
            pass
        else:
            self.emit(f"// TODO: {type(stmt).__name__}")

    def generate_assign(self, stmt: AssignStmt):
        """Generate assignment - in CADL, 'let' creates AssignStmt too"""

        # Check if this is an irf alias variable that's only used for scratchpad aliasing
        # e.g., let addr = _irf[rs1] where addr is only used in _mem or _burst accesses
        if isinstance(stmt.lhs, IdentExpr) and isinstance(stmt.rhs, IndexExpr):
            if isinstance(stmt.rhs.expr, IdentExpr) and stmt.rhs.expr.name == "_irf":
                lhs_name = stmt.lhs.name
                alias_key = self.irf_aliases.get(lhs_name)
                if alias_key:
                    # Check if any scratchpad alias uses this register
                    has_scratchpad = any(
                        alias.register_key == alias_key
                        for alias in self.scratchpad_aliases.values()
                    )
                    if has_scratchpad:
                        # This variable is only used for _mem/_burst accesses that have been optimized away
                        return

        # Special case: burst operations - eliminate them
        # CADL: bitmask[0 +: ] = _burst_read[base_addr +: 4]
        # High-level: eliminated (arrays already accessible)
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.rhs, RangeSliceExpr):
            if isinstance(stmt.rhs.expr, IdentExpr) and stmt.rhs.expr.name == "_burst_read":
                target_expr = stmt.lhs.expr
                if isinstance(target_expr, IdentExpr) and target_expr.name in self.scratchpad_aliases:
                    return
                return

        # Check if this is burst write - eliminate it
        # CADL: _burst_write[addr +: size] = array[0 +: ]
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            if stmt.lhs.expr.name == "_burst_write":
                rhs_expr = stmt.rhs.expr if isinstance(stmt.rhs, RangeSliceExpr) else None
                if isinstance(rhs_expr, IdentExpr) and rhs_expr.name in self.scratchpad_aliases:
                    return
                return

        # Check if LHS is static[i] = _mem[...] where static is aliased
        # If so, skip this because the alias already provides direct access
        if isinstance(stmt.lhs, IndexExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            lhs_static = stmt.lhs.expr.name
            if lhs_static in self.scratchpad_aliases:
                alias = self.scratchpad_aliases[lhs_static]
                if alias.index_to_offset is not None:
                    if isinstance(stmt.rhs, IndexExpr) and isinstance(stmt.rhs.expr, IdentExpr):
                        if stmt.rhs.expr.name == "_mem":
                            # This is static[i] = _mem[addr+off], skip it
                            return

        # Check if LHS is _irf[rd] write or _mem[addr] write
        # _irf[rd] = value → eliminate (will be handled as return value in future)
        # _mem[addr] = value → eliminate (memory writes handled externally in high-level mode)
        if isinstance(stmt.lhs, IndexExpr):
            if isinstance(stmt.lhs.expr, IdentExpr):
                if stmt.lhs.expr.name == "_irf" and stmt.lhs.indices:
                    key, _ = self.register_key(stmt.lhs.indices[0])
                    rhs_str = self.generate_expr(stmt.rhs)
                    if key in self.result_locals:
                        self.emit(f"{self.result_locals[key]} = {rhs_str};")
                    else:
                        self.emit(f"// _irf write ignored in explicit return mode: {rhs_str}")
                    return
                if stmt.lhs.expr.name == "_mem":
                    index_expr = stmt.lhs.indices[0] if stmt.lhs.indices else None
                    if index_expr:
                        # Check if RHS is static[i] that is aliased - if so, skip this
                        # because the alias already provides direct access
                        if isinstance(stmt.rhs, IndexExpr) and isinstance(stmt.rhs.expr, IdentExpr):
                            rhs_static = stmt.rhs.expr.name
                            if rhs_static in self.scratchpad_aliases:
                                alias = self.scratchpad_aliases[rhs_static]
                                if alias.index_to_offset is not None:
                                    # This _mem write is already captured in alias mapping
                                    # The actual write is done by rewriting result[i] -> rs2[offset]
                                    return
                        mem_lhs = self._generate_mem_index(index_expr, stmt.lhs)
                        if mem_lhs:
                            rhs_str = self.generate_expr(stmt.rhs)
                            self.emit(f"{mem_lhs} = {rhs_str};")
                            return
                    addr_str = self.generate_expr(stmt.lhs.indices[0]) if stmt.lhs.indices else "0"
                    rhs_str = self.generate_expr(stmt.rhs)
                    self.emit(f"// _mem[{addr_str}] = {rhs_str};  // eliminated in high-level mode")
                    return

        # Regular expression generation
        lhs = self.generate_expr(stmt.lhs)
        rhs = self.generate_expr(stmt.rhs)

        # Check if this is a variable declaration (lhs is simple identifier)
        # In CADL, 'let x = ...' becomes AssignStmt(IdentExpr('x'), ...)
        if isinstance(stmt.lhs, IdentExpr) and stmt.lhs.name not in self.declared_vars:
            # First assignment to this variable - declare it
            self.register_declaration(stmt.lhs.name)

            # Use explicit type annotation if available, otherwise infer from RHS
            if stmt.type_annotation:
                inferred_type = self.map_type(stmt.type_annotation)
                alias_key = self.irf_aliases.get(stmt.lhs.name) if stmt.lhs.name in self.irf_aliases else None
                if alias_key:
                    info = self.irf_read_infos.get(alias_key)
                    if info and info.pointer_element_types:
                        elem_type = self.choose_type_hint(info.pointer_element_types)
                        inferred_type = f"{elem_type} *"
            else:
                inferred_type = self.var_types.get(stmt.lhs.name) or self.infer_c_type_from_expr(stmt.rhs)
            self.var_types[stmt.lhs.name] = inferred_type
            self.emit(f"{inferred_type} {lhs} = {rhs};")
        else:
            # Regular assignment
            self.emit(f"{lhs} = {rhs};")

    def generate_return(self, stmt: ReturnStmt):
        """Generate return statement"""
        exprs = [self.generate_expr(expr) for expr in stmt.exprs]

        if not exprs:
            self.emit("return;")
            return

        if self.return_spec.kind == "scalar":
            self.emit(f"return {exprs[0]};")
            return

        if self.return_spec.kind == "struct":
            fields = []
            for idx, expr_code in enumerate(exprs):
                if idx < len(self.return_spec.fields):
                    field_name = self.return_spec.fields[idx][0]
                else:
                    field_name = f"value{idx}"
                fields.append(f".{field_name} = {expr_code}")
            init = ", ".join(fields)
            self.emit(f"return ({self.return_spec.type_name}){{{init}}};")
            return

        # Fallback for unexpected cases
        self.emit(f"return {exprs[0]};")

    def generate_dowhile(self, stmt: DoWhileStmt):
        if self.try_emit_for_loop(stmt):
            return

        for binding in stmt.bindings:
            var_name = binding.id
            init_val = self.generate_expr(binding.init) if binding.init else "0"
            c_type = self.map_type(DataType_Single(binding.ty))
            if var_name in self.declared_vars:
                self.emit(f"{var_name} = {init_val};")
            else:
                self.emit(f"{c_type} {var_name} = {init_val};")
                self.register_declaration(var_name)

        self.emit("while (1) {")
        self.indent_level += 1
        self.push_decl_scope()
        for body_stmt in stmt.body:
            self.generate_stmt(body_stmt)
        for binding in stmt.bindings:
            if binding.next is not None:
                next_expr = self.generate_expr(binding.next)
                self.emit(f"{binding.id} = {next_expr};")
        cond_str = self.generate_expr(stmt.condition)
        self.emit(f"if (!({cond_str})) break;")
        self.pop_decl_scope()
        self.indent_level -= 1
        self.emit("}")

    def generate_expr(self, expr: Expr) -> str:
        """Generate expression"""
        if isinstance(expr, LitExpr):
            return self.generate_literal(expr)
        elif isinstance(expr, IdentExpr):
            name = expr.name
            if name in self.declared_vars:
                return name
            if name in self.static_scalar_names:
                return name
            alias_key = self.register_alias_key(name)
            if alias_key and alias_key in self.irf_read_params:
                return self.irf_read_params[alias_key]
            return name
        elif isinstance(expr, BinaryExpr):
            return self.generate_binop(expr)
        elif isinstance(expr, UnaryExpr):
            return self.generate_unop(expr)
        elif isinstance(expr, SliceExpr):
            return self.generate_slice(expr)
        elif isinstance(expr, RangeSliceExpr):
            return self.generate_range_slice(expr)
        elif isinstance(expr, IndexExpr):
            return self.generate_index(expr)
        elif isinstance(expr, IfExpr):
            return self.generate_if_expr(expr)
        elif isinstance(expr, CallExpr):
            return self.generate_call(expr)
        elif isinstance(expr, SelectExpr):
            return self.generate_select(expr)
        elif isinstance(expr, TupleExpr):
            return self.generate_tuple(expr)
        else:
            return f"/* TODO: {type(expr).__name__} */"

    def generate_literal(self, expr) -> str:
        """Generate literal"""
        lit = expr.literal
        if isinstance(lit.lit, LiteralInner_Fixed):
            return str(lit.lit.value)
        elif isinstance(lit.lit, LiteralInner_Float):
            return str(lit.lit.value)
        elif isinstance(lit.lit, LiteralInner_Bool):
            return "true" if lit.lit.value else "false"
        else:
            return "0"

    def generate_binop(self, expr) -> str:
        """Generate binary operation"""
        left = self.generate_expr(expr.left)
        right = self.generate_expr(expr.right)
        op_enum = expr.op

        # Map CADL operators to C
        # The op is a BinaryOp enum, get its value
        op_str = op_enum.value if hasattr(op_enum, 'value') else str(op_enum)

        return f"({left} {op_str} {right})"

    def generate_unop(self, expr) -> str:
        """Generate unary operation"""
        operand = self.generate_expr(expr.operand)
        op_enum = expr.op

        # The op is a UnaryOp enum, get its value
        op_str = op_enum.value if hasattr(op_enum, 'value') else str(op_enum)

        return f"({op_str}{operand})"

    def generate_slice(self, expr) -> str:
        """Generate bit slice as C bit masking

        CADL: value[7:4] extracts bits 7 down to 4 (4 bits total)
        C: ((value >> 4) & 0xF)

        CADL: value[2:0] extracts bits 2 down to 0 (3 bits total)
        C: (value & 0x7)
        """
        base = self.generate_expr(expr.expr)

        # Try to extract constant indices
        start_val = self.try_eval_constant(expr.start)
        end_val = self.try_eval_constant(expr.end)

        if start_val is not None and end_val is not None:
            # CADL uses [high:low] notation
            high_bit = max(start_val, end_val)
            low_bit = min(start_val, end_val)
            width = high_bit - low_bit + 1
            mask = (1 << width) - 1

            if low_bit == 0:
                # Simple mask: value[2:0] → (value & 0x7)
                return f"({base} & 0x{mask:X})"
            else:
                # Shift then mask: value[7:4] → ((value >> 4) & 0xF)
                return f"(({base} >> {low_bit}) & 0x{mask:X})"
        else:
            # Non-constant indices - fall back to comment
            start_str = self.generate_expr(expr.start)
            end_str = self.generate_expr(expr.end)
            return f"/* TODO: dynamic bit slice {base}[{start_str}:{end_str}] */"

    def try_eval_constant(self, expr) -> Optional[int]:
        """Try to evaluate expression as a constant integer"""
        if isinstance(expr, LitExpr):
            if isinstance(expr.literal.lit, LiteralInner_Fixed):
                return expr.literal.lit.value
        if isinstance(expr, UnaryExpr):
            if expr.op == UnaryOp.NEG:
                inner = self.try_eval_constant(expr.operand)
                if inner is not None:
                    return -inner
            return None
        if isinstance(expr, BinaryExpr):
            left = self.try_eval_constant(expr.left)
            right = self.try_eval_constant(expr.right)
            if left is None or right is None:
                return None
            if expr.op == BinaryOp.ADD:
                return left + right
            if expr.op == BinaryOp.SUB:
                return left - right
            if expr.op == BinaryOp.MUL:
                return left * right
            if expr.op == BinaryOp.DIV and right != 0:
                return left // right
            if expr.op == BinaryOp.LSHIFT:
                return left << right
            if expr.op == BinaryOp.RSHIFT:
                return left >> right
        return None

    def generate_range_slice(self, expr) -> str:
        """Generate range slice expression

        CADL: arr[start +: length] or arr[start +: ]
        This is used for burst operations and array slicing.

        In C, we'll generate a pointer: &arr[start]
        The length is used in the assignment context (memcpy)
        """
        if isinstance(expr.expr, IdentExpr):
            base_name = expr.expr.name
            if base_name in self.scratchpad_aliases:
                alias = self.scratchpad_aliases[base_name]
                param = self.irf_read_params.get(alias.register_key)
                if param:
                    start = self.generate_expr(expr.start)
                    start_index = self._format_alias_index(alias, start)
                    return f"(&{param}[{start_index}])"

        base = self.generate_expr(expr.expr)
        start = self.generate_expr(expr.start)

        # Return a sub-array reference
        # This will be used in assignment context like:
        # memcpy(dest, &src[offset], length)
        return f"(&{base}[{start}])"

    def generate_index(self, expr) -> str:
        """Generate array indexing

        Special handling:
        - _irf[rs1] → rs1_value (convert to direct parameter)
        """
        if isinstance(expr.expr, IdentExpr) and expr.expr.name == "_irf" and expr.indices:
            key, _ = self.register_key(expr.indices[0])
            param = self.irf_read_params.get(key)
            if param:
                return param
            return f"/* irf_{key} */"

        if isinstance(expr.expr, IdentExpr):
            base_name = expr.expr.name
            if base_name in self.scratchpad_aliases and expr.indices:
                alias = self.scratchpad_aliases[base_name]
                param = self.irf_read_params.get(alias.register_key)
                if param:
                    index_code = self.generate_expr(expr.indices[0]) if expr.indices else "0"
                    static_index = self.try_eval_constant(expr.indices[0]) if expr.indices else None
                    index_code = self._format_alias_index(alias, index_code, static_index)
                    return f"{param}[{index_code}]"
            if base_name == "_mem" and expr.indices:
                mem_access = self._generate_mem_index(expr.indices[0], expr)
                if mem_access:
                    return mem_access

        base = self.generate_expr(expr.expr)

        if len(expr.indices) == 1:
            index = self.generate_expr(expr.indices[0])
            return f"{base}[{index}]"
        else:
            # Multi-dimensional - flatten to single dimension
            indices_str = "][".join(self.generate_expr(idx) for idx in expr.indices)
            return f"{base}[{indices_str}]"

    def generate_if_expr(self, expr) -> str:
        """Generate ternary operator"""
        cond = self.generate_expr(expr.condition)
        then_val = self.generate_expr(expr.then_branch)
        else_val = self.generate_expr(expr.else_branch)
        return f"({cond} ? {then_val} : {else_val})"

    def generate_call(self, expr: CallExpr) -> str:
        """Generate function call

        CADL function calls map to C function calls.
        Special handling for built-in functions if needed.
        """
        func_name = expr.name
        args_str = ", ".join(self.generate_expr(arg) for arg in expr.args)

        # Check for special built-in functions that need C equivalents
        builtin_map = {
            "abs": "abs",
            "min": "min",
            "max": "max",
            "sqrt": "sqrt",
            "exp": "exp",
            "log": "log",
            "pow": "pow",
            "floor": "floor",
            "ceil": "ceil",
        }

        c_func = builtin_map.get(func_name, func_name)
        return f"{c_func}({args_str})"

    def generate_select(self, expr: SelectExpr) -> str:
        """Generate chained ternary expression for CADL select."""
        result = self.generate_expr(expr.default)
        for condition, value in reversed(expr.arms):
            cond_str = self.generate_expr(condition)
            value_str = self.generate_expr(value)
            result = f"({cond_str} ? {value_str} : {result})"
        return result

    def generate_tuple(self, expr: TupleExpr) -> str:
        """Generate tuple expression

        When multiple elements are present, fall back to a comma-expression that
        evaluates all elements and yields the last one.
        """
        elements = [self.generate_expr(e) for e in expr.elements]
        if not elements:
            return "0"
        if len(elements) == 1:
            return elements[0]
        combined = ", ".join(elements)
        return f"/* tuple */({combined})"

    def map_type(self, dtype: DataType) -> str:
        """Map CADL type to C type (byte-aligned)"""
        if isinstance(dtype, DataType_Single):
            basic = dtype.basic_type
            if isinstance(basic, BasicType_ApUFixed):
                width = basic.width
                if width <= 8:
                    self._note_type("uint8_t")
                    return "uint8_t"
                elif width <= 16:
                    self._note_type("uint16_t")
                    return "uint16_t"
                elif width <= 32:
                    self._note_type("uint32_t")
                    return "uint32_t"
                else:
                    self._note_type("uint64_t")
                    return "uint64_t"
            elif isinstance(basic, BasicType_ApFixed):
                width = basic.width
                if width <= 8:
                    self._note_type("int8_t")
                    return "int8_t"
                elif width <= 16:
                    self._note_type("int16_t")
                    return "int16_t"
                elif width <= 32:
                    self._note_type("int32_t")
                    return "int32_t"
                else:
                    self._note_type("int64_t")
                    return "int64_t"
            elif isinstance(basic, BasicType_Float32):
                self._note_type("float")
                return "float"
            elif isinstance(basic, BasicType_Float64):
                self._note_type("double")
                return "double"

        # Default fallback
        self._note_type("uint32_t")
        return "uint32_t"


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Transpile CADL to high-level C code for Polygeist',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
High-level C code generation:
  - No cpu_mem or _irf base pointers (retain _mem pointer when loads remain)
  - _irf[rsX] → rsX_value (direct value parameters)
  - Burst operations eliminated
  - Clean function signatures for instruction matching

Examples:
  # Generate C code from CADL
  python -m cadl_frontend.transpile_to_c input.cadl -o output.c

  # Compile generated C code
  gcc -c -std=c11 -Wall output.c -o output.o

  # Use with Polygeist for MLIR generation
  polygeist-opt output.c -o output.mlir
        """
    )
    parser.add_argument('input', type=Path, help='Input CADL file')
    parser.add_argument('-o', '--output', type=Path, help='Output C file')

    args = parser.parse_args()

    # Transpile
    transpiler = CTranspiler()
    c_code = transpiler.transpile(args.input)

    # Write output
    if args.output:
        args.output.write_text(c_code)
        print(f"Generated high-level C code written to {args.output}")
    else:
        print(c_code)


if __name__ == '__main__':
    main()
