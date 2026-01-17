"""
CADL AST to MLIR Converter

This module provides a framework for converting CADL Abstract Syntax Trees
to MLIR Intermediate Representation, leveraging CIRCT dialects for hardware
synthesis and optimization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# MLIR Python bindings
import circt
import circt.ir as ir

# Loop transformation module
from .loop_transform import LoopTransformer
import circt.dialects.func as func
import circt.dialects.arith as arith
import circt.dialects.scf as scf
import circt.dialects.memref as memref

# CIRCT Python bindings
import circt.dialects.comb as comb
import circt.dialects.hw as hw
import circt.dialects.aps as aps

# CADL AST imports
from .ast import (
    Proc, Flow, Static, Regfile,
    Stmt, Expr, BasicType, DataType, CompoundType,
    BasicType_ApFixed, BasicType_ApUFixed, BasicType_Float32, BasicType_Float64,
    BasicType_String, BasicType_USize,
    DataType_Single, DataType_Array, DataType_Instance,
    CompoundType_Basic,
    LitExpr, IdentExpr, BinaryExpr, UnaryExpr, CallExpr, IndexExpr, SliceExpr, RangeSliceExpr, IfExpr, SelectExpr, AggregateExpr, ArrayLiteralExpr, StringLitExpr,
    AssignStmt, ReturnStmt, ForStmt, DoWhileStmt, ExprStmt, DirectiveStmt, WithBinding,
    BinaryOp, UnaryOp, FlowKind
)


@dataclass
class TypedValue:
    """Wrapper for MLIR value with CADL type information"""
    value: Union[ir.Value, str]
    cadl_type: Optional[DataType] = None

@dataclass
class SymbolScope:
    """Symbol scope for managing variable bindings in different contexts"""
    symbols: Dict[str, TypedValue] = field(default_factory=dict)
    parent: Optional[SymbolScope] = None

    def get(self, name: str) -> Optional[TypedValue]:
        """Get symbol with type info, checking parent scopes if not found"""
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.get(name)
        return None

    def set(self, name: str, value: Union[ir.Value, str], cadl_type: Optional[DataType] = None) -> None:
        """Set symbol value with type info in current scope"""
        self.symbols[name] = TypedValue(value, cadl_type)

    def has(self, name: str) -> bool:
        """Check if symbol exists in any scope"""
        return self.get(name) is not None


class CADLMLIRConverter:
    """
    Main converter class for transforming CADL AST to MLIR IR

    Maintains MLIR context, module, and symbol table for SSA form generation.
    Uses CIRCT dialects for hardware-oriented operations.
    """

    def __init__(self):
        # MLIR context and module
        self.context = ir.Context()
        self.module: Optional[ir.Module] = None

        # Load required dialects and configure context
        self._load_dialects()

        # Symbol table management for SSA form
        self.current_scope = SymbolScope()
        self.scope_stack: List[SymbolScope] = []

        # Builder for generating operations
        self.builder: Optional[ir.InsertionPoint] = None

        # Cache for global references within current function
        self.current_global_refs: Dict[str, ir.Value] = {}

        # Track global operations (name -> GlobalOp) for type inference
        self.global_ops: Dict[str, memref.GlobalOp] = {}

        # Track constant variables (name -> constant value)
        self.constant_vars: Dict[str, int] = {}

        # Track pending directives for next statement
        self.pending_directives: List[DirectiveStmt] = []

        # Loop transformer
        self.loop_transformer = LoopTransformer(self)

    def _load_dialects(self) -> None:
        """Load required MLIR and CIRCT dialects"""
        # Register CIRCT dialects with the context
        with self.context:
            circt.register_dialects(self.context)

    def push_scope(self) -> None:
        """Push new symbol scope onto stack"""
        self.scope_stack.append(self.current_scope)
        self.current_scope = SymbolScope(parent=self.current_scope)
        # Reset global reference cache for new function scope
        self.current_global_refs = {}

    def pop_scope(self) -> None:
        """Pop symbol scope from stack"""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
            # Clear global reference cache when exiting scope to avoid referencing
            # SSA values from inner scopes that are no longer valid
            self.current_global_refs = {}

    def get_symbol(self, name: str) -> Optional[Union[ir.Value, str]]:
        """Get SSA value for symbol name"""
        typed_val = self.current_scope.get(name)
        return typed_val.value if typed_val else None

    def get_symbol_type(self, name: str) -> Optional[DataType]:
        """Get CADL type for symbol name"""
        typed_val = self.current_scope.get(name)
        return typed_val.cadl_type if typed_val else None

    def set_symbol(self, name: str, value: Union[ir.Value, str], cadl_type: Optional[DataType] = None) -> None:
        """Set SSA value and CADL type for symbol name in current scope"""
        self.current_scope.set(name, value, cadl_type)

    def get_cpu_memory_instance(self) -> ir.Value:
        """Get reference to the global CPU memory instance"""
        if "_cpu_memory" not in self.global_ops:
            raise RuntimeError("Global CPU memory should be declared at module level before use")

        # Get reference to the global memory - type is read from the cached operation
        return self._get_global_reference("_cpu_memory")

    def _get_global_reference(self, global_name: str) -> ir.Value:
        """Get reference to global variable, with caching to avoid duplicates.

        Type is inferred from the cached GlobalOp that was created during declaration.
        """
        if global_name not in self.current_global_refs:
            # Get the type from the cached global op
            if global_name not in self.global_ops:
                raise RuntimeError(f"Global {global_name} not declared (no GlobalOp found)")

            # Infer type from the GlobalOp's type attribute
            # Note: global_op.type_ returns a TypeAttr, need to extract the actual Type
            type_attr = self.global_ops[global_name].type_
            if hasattr(type_attr, 'value'):
                memory_type = type_attr.value
            else:
                memory_type = type_attr

            # Create new global reference and cache it
            global_ref = memref.GetGlobalOp(memory_type, global_name)
            self.current_global_refs[global_name] = global_ref.result
        return self.current_global_refs[global_name]

    def _is_scalar_global(self, global_name: str) -> bool:
        """Check if a global is a scalar (rank-0 memref)"""
        if global_name not in self.global_ops:
            return False

        type_attr = self.global_ops[global_name].type_
        if hasattr(type_attr, 'value'):
            memory_type = type_attr.value
        else:
            memory_type = type_attr

        # Check if it's a rank-0 memref (scalar)
        if isinstance(memory_type, ir.MemRefType):
            return len(memory_type.shape) == 0
        return False

    def _declare_global_memory(self) -> None:
        """Declare global CPU memory at module level using memref.global"""
        if "_cpu_memory" not in self.global_ops:
            # Create global memory using memref.global with static size
            element_type = ir.IntegerType.get_signless(32)
            memory_size = 1024
            memory_type = memref.MemRefType.get([memory_size], element_type)

            # Create a global memref variable
            global_name = "_cpu_memory"
            global_op = memref.GlobalOp(global_name, memory_type)

            # Store the global op for type inference
            self.global_ops[global_name] = global_op

            # Store the global reference for symbol resolution
            self.set_symbol("_cpu_memory", global_name)

    def _function_uses_memory(self, function) -> bool:
        """Check if a function uses _mem operations"""
        # We'll need to analyze the function body to see if it contains _mem operations
        # For now, let's implement a simple visitor pattern
        if not function.body:
            return False

        return self._stmt_list_uses_memory(function.body)

    def _flow_uses_memory(self, flow) -> bool:
        """Check if a flow uses _mem operations"""
        if not flow.body:
            return False

        return self._stmt_list_uses_memory(flow.body)

    def _stmt_list_uses_memory(self, stmts) -> bool:
        """Check if a list of statements uses _mem operations"""
        for stmt in stmts:
            if self._stmt_uses_memory(stmt):
                return True
        return False

    def _stmt_uses_memory(self, stmt) -> bool:
        """Check if a statement uses _mem operations"""
        if isinstance(stmt, AssignStmt):
            # Check LHS for _mem assignment
            if isinstance(stmt.lhs, IndexExpr) and isinstance(stmt.lhs.expr, IdentExpr):
                if stmt.lhs.expr.name == "_mem":
                    return True
            # Check RHS for _mem read
            if self._expr_uses_memory(stmt.rhs):
                return True
        elif isinstance(stmt, ExprStmt):
            return self._expr_uses_memory(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            return any(self._expr_uses_memory(expr) for expr in stmt.exprs)
        elif isinstance(stmt, DoWhileStmt):
            if self._stmt_list_uses_memory(stmt.body):
                return True
            if self._expr_uses_memory(stmt.condition):
                return True
        # Add other statement types as needed
        return False

    def _expr_uses_memory(self, expr) -> bool:
        """Check if an expression uses _mem operations"""
        if isinstance(expr, IndexExpr) and isinstance(expr.expr, IdentExpr):
            if expr.expr.name == "_mem":
                return True
        elif isinstance(expr, BinaryExpr):
            return self._expr_uses_memory(expr.left) or self._expr_uses_memory(expr.right)
        elif isinstance(expr, UnaryExpr):
            return self._expr_uses_memory(expr.operand)
        elif isinstance(expr, CallExpr):
            return any(self._expr_uses_memory(arg) for arg in expr.args)
        # Add other expression types as needed
        return False

    def convert_cadl_type(self, cadl_type: Union[BasicType, DataType, CompoundType]) -> ir.Type:
        """
        Convert CADL type to MLIR type

        Maps CADL type system to appropriate MLIR types.
        Both signed and unsigned CADL types map to signless MLIR integers.
        Signedness is handled by operation semantics (e.g., divsi vs divui).
        """
        if isinstance(cadl_type, BasicType_ApFixed):
            # Use signless integers for both signed and unsigned
            return ir.IntegerType.get_signless(cadl_type.width)

        elif isinstance(cadl_type, BasicType_ApUFixed):
            return ir.IntegerType.get_signless(cadl_type.width)

        elif isinstance(cadl_type, BasicType_Float32):
            return ir.F32Type.get()

        elif isinstance(cadl_type, BasicType_Float64):
            return ir.F64Type.get()

        elif isinstance(cadl_type, BasicType_String):
            raise NotImplementedError("String types are not supported in MLIR conversion")

        elif isinstance(cadl_type, BasicType_USize):
            return ir.IndexType.get()

        elif isinstance(cadl_type, DataType_Single):
            return self.convert_cadl_type(cadl_type.basic_type)

        elif isinstance(cadl_type, DataType_Array):
            element_type = self.convert_cadl_type(cadl_type.element_type)
            return memref.MemRefType.get(cadl_type.dimensions, element_type)

        elif isinstance(cadl_type, CompoundType_Basic):
            return self.convert_cadl_type(cadl_type.data_type)

        else:
            raise NotImplementedError(f"Unsupported CADL type: {type(cadl_type)}")

    def convert_proc(self, proc: Proc) -> ir.Module:
        """
        Convert CADL Proc to MLIR Module

        Creates top-level MLIR module containing all functions, flows,
        and global variables from the processor definition.
        """
        # Store proc reference for later access
        self.proc = proc

        with self.context, ir.Location.unknown():
            # Create module
            self.module = ir.Module.create()

            with ir.InsertionPoint(self.module.body):
                self.builder = ir.InsertionPoint.current

                # Check if any flows use memory
                any_uses_memory = any(self._flow_uses_memory(flow) for flow in proc.flows.values())

                # Declare global memory if needed
                if any_uses_memory:
                    self._declare_global_memory()

                # Convert static variables to global declarations
                for static in proc.statics.values():
                    self._convert_static(static)

                # Convert flows (as functions for now)
                for flow in proc.flows.values():
                    self._convert_flow(flow)

                # TODO: Convert register files to appropriate MLIR constructs

        return self.module

    def _convert_static(self, static: Static) -> None:
        """Convert static variable to MLIR global"""
        mlir_type = self.convert_cadl_type(static.ty)

        # Create a global variable using memref.global
        global_name = static.id

        # Get initial value if provided
        initial_value = None
        initial_values_list = None
        if static.expr:
            if isinstance(static.expr, LitExpr):
                # Single literal value
                initial_value = static.expr.literal.lit.value
            elif isinstance(static.expr, AggregateExpr):
                # Array initialization like {1474560, 870484, ...}
                initial_values_list = []
                for elem_expr in static.expr.elements:
                    if isinstance(elem_expr, LitExpr):
                        initial_values_list.append(elem_expr.literal.lit.value)
                    else:
                        # For non-literal elements, we'll skip initialization for now
                        initial_values_list = None
                        break

        # Determine the correct memref type based on the CADL type
        if isinstance(mlir_type, ir.MemRefType):
            # mlir_type is already a memref (for arrays), use it directly
            memref_type = mlir_type
        else:
            # mlir_type is a scalar element type, wrap it in memref<>
            memref_type = ir.MemRefType.get([], mlir_type)

        # Create a tensor type for the initial value if provided
        if initial_value is not None:
            # Single scalar initialization
            if isinstance(mlir_type, ir.MemRefType):
                # This shouldn't happen - scalar initialization requires scalar type
                raise RuntimeError(f"Scalar initialization provided for array type: {mlir_type}")
            else:
                # Create attribute from the integer value for scalars
                element_attr = ir.IntegerAttr.get(mlir_type, initial_value)
                attr = ir.DenseElementsAttr.get_splat(ir.RankedTensorType.get([], mlir_type), element_attr)
                global_op = memref.GlobalOp(global_name, memref_type, initial_value=attr, constant=True)
        elif initial_values_list is not None:
            # Array initialization with list of values
            if isinstance(mlir_type, ir.MemRefType):
                # Create dense elements attribute for array initialization
                element_type = mlir_type.element_type
                shape = mlir_type.shape

                # Create integer attributes for each value
                element_attrs = []
                for val in initial_values_list:
                    element_attrs.append(ir.IntegerAttr.get(element_type, val))

                # Create tensor type and dense elements attribute
                tensor_type = ir.RankedTensorType.get(shape, element_type)
                dense_attr = ir.DenseElementsAttr.get(element_attrs, tensor_type)

                # Create global with initialization
                global_op = memref.GlobalOp(global_name, memref_type, initial_value=dense_attr, constant=True)
            else:
                # This shouldn't happen - array initialization requires MemRefType
                raise RuntimeError(f"Array initialization provided for non-array type: {mlir_type}")
        else:
            # Create uninitialized global
            global_op = memref.GlobalOp(global_name, memref_type)

        # Add var_name attribute matching the global name
        global_op.attributes["var_name"] = ir.StringAttr.get(global_name)

        # Store the global op for type inference
        self.global_ops[global_name] = global_op

        # Add custom attributes if present
        if static.attrs:
            for attr_name, attr_value in static.attrs.items():
                # Convert CADL attribute value to MLIR attribute
                mlir_attr = self._convert_attribute_value(attr_value)
                if mlir_attr is not None:
                    global_op.attributes[attr_name] = mlir_attr

        # Store the global reference for symbol resolution
        self.set_symbol(static.id, global_name)

    def _convert_attribute_value(self, expr: Optional[Expr]) -> Optional[ir.Attribute]:
        """
        Convert CADL expression to MLIR attribute for use in operation attributes

        Handles:
        - StringLitExpr -> StringAttr
        - LitExpr with integer -> IntegerAttr
        - ArrayLiteralExpr -> ArrayAttr with typed elements
        - None -> UnitAttr (for presence-only attributes)
        """
        if expr is None:
            # Attribute without value, use UnitAttr
            return ir.UnitAttr.get()

        if isinstance(expr, StringLitExpr):
            # String attribute
            return ir.StringAttr.get(expr.value)

        if isinstance(expr, LitExpr):
            # Numeric attribute
            literal = expr.literal
            if hasattr(literal.lit, 'value'):
                value = literal.lit.value
                if isinstance(value, int):
                    # Integer attribute
                    mlir_type = self.convert_cadl_type(literal.ty)
                    return ir.IntegerAttr.get(mlir_type, value)
                elif isinstance(value, float):
                    # Float attribute
                    mlir_type = self.convert_cadl_type(literal.ty)
                    return ir.FloatAttr.get(mlir_type, value)

        if isinstance(expr, ArrayLiteralExpr):
            # Array attribute - convert each element and create ArrayAttr
            element_attrs = []
            for elem_expr in expr.elements:
                elem_attr = self._convert_attribute_value(elem_expr)
                if elem_attr is not None:
                    element_attrs.append(elem_attr)
            return ir.ArrayAttr.get(element_attrs)

        if isinstance(expr, IdentExpr):
            # Identifier - treat as string symbol
            return ir.StringAttr.get(expr.name)

        # For other expression types, try to convert to string representation
        return ir.StringAttr.get(str(expr))

    def _convert_flow(self, flow: Flow) -> ir.Operation:
        """Convert CADL Flow to MLIR function (for now)"""
        # For now, treat flows as functions
        # TODO: Implement hardware-specific flow conversion

        # Convert input types
        arg_types = [self.convert_cadl_type(dtype) for _, dtype in flow.inputs]

        # Check if flow uses _mem (but don't add it as argument anymore)
        uses_memory = self._flow_uses_memory(flow)

        # Flows typically return void or single value
        ret_types = []  # TODO: Determine from flow analysis

        # Create function type
        func_type = ir.FunctionType.get(arg_types, ret_types)

        # Create function operation with flow name
        func_op = func.FuncOp(f"flow_{flow.name}", func_type)

        # Add all attributes from flow to MLIR function
        if flow.attrs and flow.attrs.attrs:
            for attr_name, attr_expr in flow.attrs.attrs.items():
                if attr_expr and isinstance(attr_expr, LitExpr):
                    # Extract the literal value
                    attr_value = attr_expr.literal.lit.value
                    # Create integer attribute for any attribute
                    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), attr_value)
                    func_op.attributes[attr_name] = attr
                elif attr_expr is None:
                    # Simple attribute without value (like #[inline])
                    # Create a unit attribute (boolean true)
                    func_op.attributes[attr_name] = ir.UnitAttr.get()
                # TODO: Handle other expression types for attributes if needed

        # Add entry block
        entry_block = func_op.add_entry_block()

        with ir.InsertionPoint(entry_block):
            # Push new scope for flow
            self.push_scope()

            # Bind flow inputs to symbols
            for i, (name, _) in enumerate(flow.inputs):
                arg_value = entry_block.arguments[i]
                self.set_symbol(name, arg_value)

            # If flow uses memory, it will be declared via aps.memdeclare when first accessed
            # No need to set up CPU memory instance here anymore

            # Convert flow body
            if flow.body:
                self._convert_stmt_list(flow.body)

            # Check if last operation is already a return
            # If not, add a default return
            block_ops = list(entry_block.operations)
            if not block_ops or not isinstance(block_ops[-1], func.ReturnOp):
                func.ReturnOp([])

            # Pop flow scope
            self.pop_scope()

        return func_op

    def _convert_stmt_list(self, stmts: List[Stmt]) -> None:
        """Convert list of statements maintaining SSA form"""
        for stmt in stmts:
            self._convert_stmt(stmt)

    def _convert_type_if_needed(self, value: ir.Value, type_annotation: Optional[DataType]) -> ir.Value:
        """
        Convert value to target type if type annotation specifies a different type.

        Handles integer width conversions and sign conversions:
        - Extension: extui (unsigned) or extsi (signed)
        - Truncation: trunci
        - Sign conversion: bitcast for same-width sign changes

        Args:
            value: The MLIR value to potentially convert
            type_annotation: Optional CADL type annotation from assignment

        Returns:
            Converted value if conversion needed, otherwise original value
        """
        if type_annotation is None:
            return value

        # Convert target CADL type to MLIR type
        target_type = self.convert_cadl_type(type_annotation)
        source_type = value.type

        # Check if conversion is needed
        if source_type == target_type:
            return value

        # Handle integer type conversions
        if isinstance(source_type, ir.IntegerType) and isinstance(target_type, ir.IntegerType):
            source_width = source_type.width
            target_width = target_type.width

            # Step 1: Handle sign conversion if needed (same or different width)
            # Check if target type annotation indicates different signedness
            current_value = value
            if isinstance(type_annotation, DataType_Single):
                basic_type = type_annotation.basic_type
                # Determine if we need to track signedness information
                # For now, since both map to signless, no conversion needed
                # This is a placeholder for future signed type tracking

            # Step 2: Handle width conversion
            if source_width < target_width:
                # Extension needed - use the CADL type to determine sign extension
                if isinstance(type_annotation, DataType_Single):
                    basic_type = type_annotation.basic_type
                    if isinstance(basic_type, BasicType_ApFixed):
                        # Signed extension (for i16 -> i32, etc.)
                        return arith.ExtSIOp(target_type, current_value).result
                    else:
                        # Unsigned extension (for u16 -> u32, etc.)
                        return arith.ExtUIOp(target_type, current_value).result
                else:
                    # Default to unsigned extension
                    return arith.ExtUIOp(target_type, current_value).result
            elif source_width > target_width:
                # Truncation needed
                return arith.TruncIOp(target_type, current_value).result
            # else: same width, no conversion needed (sign conversion handled above)

            return current_value

        # For other type conversions, return value as-is for now
        # TODO: Add float conversions, etc.
        return value

    def _convert_value_to_target_type(self, value: ir.Value, target_type: ir.Type, use_sign_extend: bool = False) -> ir.Value:
        """
        Convert value to target MLIR type for store operations.

        This ensures that the value being stored matches the target memory element type.
        Handles integer width conversions:
        - Extension: extui (default) or extsi (if use_sign_extend=True)
        - Truncation: trunci

        Args:
            value: The MLIR value to potentially convert
            target_type: The target MLIR type (e.g., element type of memref)
            use_sign_extend: Whether to use sign extension for widening (default: False)

        Returns:
            Converted value if conversion needed, otherwise original value
        """
        source_type = value.type

        # Check if conversion is needed
        if source_type == target_type:
            return value

        # Handle integer type conversions
        if isinstance(source_type, ir.IntegerType) and isinstance(target_type, ir.IntegerType):
            source_width = source_type.width
            target_width = target_type.width

            if source_width < target_width:
                # Extension needed
                if use_sign_extend:
                    return arith.ExtSIOp(target_type, value).result
                else:
                    return arith.ExtUIOp(target_type, value).result
            elif source_width > target_width:
                # Truncation needed
                return arith.TruncIOp(target_type, value).result
            # else: same width, no conversion needed

        # For other type conversions, return value as-is for now
        return value

    def _get_global_element_type(self, global_name: str) -> Optional[ir.Type]:
        """
        Get the element type of a scalar global variable.

        Args:
            global_name: The name of the global variable

        Returns:
            The element type of the global, or None if not found
        """
        if global_name not in self.global_ops:
            return None

        type_attr = self.global_ops[global_name].type_
        if hasattr(type_attr, 'value'):
            memory_type = type_attr.value
        else:
            memory_type = type_attr

        if isinstance(memory_type, ir.MemRefType):
            return memory_type.element_type
        return None

    def _convert_stmt(self, stmt: Stmt) -> None:
        """Convert single statement to MLIR operations"""
        if isinstance(stmt, ExprStmt):
            self._convert_expr(stmt.expr)

        elif isinstance(stmt, AssignStmt):
            # Check for burst operations first
            if self._is_burst_operation(stmt):
                self._convert_burst_operation(stmt)
                return

            # Track constant assignments for loop analysis
            if isinstance(stmt.lhs, IdentExpr):
                if isinstance(stmt.rhs, LitExpr):
                    # This is a constant assignment
                    if hasattr(stmt.rhs.literal.lit, 'value'):
                        self.constant_vars[stmt.lhs.name] = stmt.rhs.literal.lit.value
                else:
                    # This is a non-constant assignment - invalidate if previously constant
                    if stmt.lhs.name in self.constant_vars:
                        del self.constant_vars[stmt.lhs.name]

            # Convert RHS expression
            rhs_value = self._convert_expr(stmt.rhs)

            # Apply type conversion if type annotation is present
            rhs_value = self._convert_type_if_needed(rhs_value, stmt.type_annotation)

            # Handle LHS assignment
            if isinstance(stmt.lhs, IdentExpr):
                # Check if assigning to a global scalar variable
                symbol_value = self.get_symbol(stmt.lhs.name)
                if isinstance(symbol_value, str) and self._is_scalar_global(symbol_value):
                    # Global scalar assignment: use aps.globalstore
                    # First, ensure the value type matches the global's element type
                    target_type = self._get_global_element_type(symbol_value)
                    if target_type is not None:
                        rhs_value = self._convert_value_to_target_type(rhs_value, target_type)
                    symbol_ref = ir.FlatSymbolRefAttr.get(symbol_value)
                    aps.GlobalStore(rhs_value, symbol_ref)
                    # Don't update symbol table - it remains as a global reference
                else:
                    # Local variable assignment or non-scalar global
                    self.set_symbol(stmt.lhs.name, rhs_value, stmt.type_annotation)
            elif isinstance(stmt.lhs, IndexExpr):
                # Handle indexed assignment (e.g., _irf[rd] = value, _mem[addr] = value)
                self._convert_index_assignment(stmt.lhs, rhs_value)
            elif isinstance(stmt.lhs, RangeSliceExpr):
                # Handle range slice assignment
                self._convert_range_slice_assignment(stmt.lhs, rhs_value)
            else:
                raise NotImplementedError(f"Complex LHS assignment not yet supported: {type(stmt.lhs)}")

        elif isinstance(stmt, ReturnStmt):
            # Convert return expressions
            ret_values = [self._convert_expr(expr) for expr in stmt.exprs]
            func.ReturnOp(ret_values)

        elif isinstance(stmt, DoWhileStmt):
            # Use scf.while as advised - perfect semantic match
            self._convert_do_while(stmt)
            # Clear directives after applying to loop
            self.pending_directives = []

        # elif isinstance(stmt, ForStmt):
        #     # Convert for loops using scf.for
        #     self._convert_for_loop(stmt)

        elif isinstance(stmt, DirectiveStmt):
            # Directives are hints/pragmas - collect them for next statement
            # e.g., [[unroll(4)]] will be applied to the following loop
            self.pending_directives.append(stmt)

        else:
            raise NotImplementedError(f"Statement type not yet supported: {type(stmt)}")

    def _convert_expr(self, expr: Expr) -> ir.Value:
        """Convert expression to MLIR SSA value"""
        if isinstance(expr, LitExpr):
            # Convert literal to constant operation
            literal = expr.literal
            mlir_type = self.convert_cadl_type(literal.ty)

            if hasattr(literal.lit, 'value'):
                value = literal.lit.value
                return arith.ConstantOp(mlir_type, value).result
            else:
                raise NotImplementedError(f"Literal type not supported: {type(literal.lit)}")

        elif isinstance(expr, IdentExpr):
            # Look up symbol in symbol table
            value = self.get_symbol(expr.name)
            if value is None:
                raise ValueError(f"Undefined symbol: {expr.name}")

            # If it's a global variable reference, load from it
            if isinstance(value, str):
                # Check if it's a scalar global (use aps.globalload) or array (use aps.memload)
                if self._is_scalar_global(value):
                    # Scalar global: use aps.globalload with symbol reference
                    # Get element type from the rank-0 memref
                    type_attr = self.global_ops[value].type_
                    if hasattr(type_attr, 'value'):
                        memory_type = type_attr.value
                    else:
                        memory_type = type_attr
                    element_type = memory_type.element_type

                    # Create symbol reference attribute
                    symbol_ref = ir.FlatSymbolRefAttr.get(value)
                    return aps.GlobalLoad(element_type, symbol_ref).result
                else:
                    # Array global: use aps.memload (existing behavior)
                    global_ref = self._get_global_reference(value)
                    # Extract element type from the memref type
                    if hasattr(global_ref.type, 'element_type'):
                        element_type = global_ref.type.element_type
                    else:
                        element_type = ir.IntegerType.get_signless(32)
                    return aps.MemLoad(element_type, global_ref, []).result
            else:
                return value

        elif isinstance(expr, BinaryExpr):
            # Convert binary operations using appropriate dialects
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)

            # Pass the original expression to determine signedness from CADL types
            return self._convert_binary_op(expr.op, left, right, expr)

        elif isinstance(expr, UnaryExpr):
            # Convert unary operations
            operand = self._convert_expr(expr.operand)
            return self._convert_unary_op(expr.op, operand)

        elif isinstance(expr, IndexExpr):
            # Convert index operations - handle special cases for _irf, _mem, etc.
            return self._convert_index_expr(expr)

        elif isinstance(expr, SliceExpr):
            # Convert slice operations like z[31:31]
            return self._convert_slice_expr(expr)

        elif isinstance(expr, IfExpr):
            # Convert if expressions to conditional operations
            return self._convert_if_expr(expr)

        elif isinstance(expr, SelectExpr):
            # Convert select expressions to conditional operations
            return self._convert_select_expr(expr)

        else:
            raise NotImplementedError(f"Expression type not yet supported: {type(expr)}")

    def _to_signless(self, value: ir.Value) -> ir.Value:
        """
        Convert signed integer type to signless integer type for arithmetic operations.

        Most arith dialect operations require signless integers, so we convert
        signed types to signless before operations.
        """
        if isinstance(value.type, ir.IntegerType) and value.type.is_signed:
            signless_type = ir.IntegerType.get_signless(value.type.width)
            cast_op = ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                results=[signless_type],
                operands=[value]
            )
            return cast_op.results[0]
        return value

    def _promote_operands(self, left: ir.Value, right: ir.Value, expr: Optional[BinaryExpr] = None) -> tuple[ir.Value, ir.Value]:
        """
        Promote operands to same type for binary operations.

        When integer operands have different widths, extends the narrower one
        to match the wider one (following hardware description language semantics).

        Args:
            left: Left operand value
            right: Right operand value
            expr: Original BinaryExpr AST node (optional, for signedness checking)

        Returns:
            Tuple of (promoted_left, promoted_right) with matching types
        """
        # Check if both operands are integers with different widths
        if isinstance(left.type, ir.IntegerType) and isinstance(right.type, ir.IntegerType):
            left_width = left.type.width
            right_width = right.type.width

            if left_width < right_width:
                # Determine if left operand is signed from expression
                is_left_signed = expr and self._get_expr_signedness(expr.left)
                # Extend left to match right (use sign-aware extension)
                if is_left_signed:
                    return (arith.ExtSIOp(right.type, left).result, right)
                else:
                    return (arith.ExtUIOp(right.type, left).result, right)
            elif right_width < left_width:
                # Determine if right operand is signed from expression
                is_right_signed = expr and self._get_expr_signedness(expr.right)
                # Extend right to match left (use sign-aware extension)
                if is_right_signed:
                    return (left, arith.ExtSIOp(left.type, right).result)
                else:
                    return (left, arith.ExtUIOp(left.type, right).result)

        # No promotion needed (same type or non-integer types)
        return (left, right)

    def _is_signed_type(self, ty: Optional[Union[BasicType, DataType, CompoundType]]) -> bool:
        """Check if a CADL type is signed (i32, i64, etc.)"""
        if isinstance(ty, BasicType_ApFixed):
            return True
        elif isinstance(ty, DataType_Single):
            return self._is_signed_type(ty.basic_type)
        elif isinstance(ty, CompoundType_Basic):
            return self._is_signed_type(ty.data_type)
        return False

    def _get_expr_signedness(self, expr: Expr) -> bool:
        """Check if an expression represents a signed value"""
        if isinstance(expr, IdentExpr):
            cadl_type = self.get_symbol_type(expr.name)
            if cadl_type:
                return self._is_signed_type(cadl_type)
        return False

    def _convert_binary_op(self, op: BinaryOp, left: ir.Value, right: ir.Value, expr: Optional[BinaryExpr] = None) -> ir.Value:
        """Convert binary operation to appropriate MLIR operation

        Args:
            op: Binary operation type
            left: Left MLIR value
            right: Right MLIR value
            expr: Original BinaryExpr AST node (optional, for type checking)
        """
        # Promote operands to same type if they have different integer widths
        left, right = self._promote_operands(left, right, expr)

        # Check if left operand is signed by checking the original expression's CADL type
        is_signed = expr and self._get_expr_signedness(expr.left)

        # Arithmetic operations (prefer arith dialect for arithmetic)
        if op == BinaryOp.ADD:
            return arith.AddIOp(left, right).result
        elif op == BinaryOp.SUB:
            return arith.SubIOp(left, right).result
        elif op == BinaryOp.MUL:
            return arith.MulIOp(left, right).result
        elif op == BinaryOp.DIV:
            # Use signed or unsigned division based on operand signedness
            if is_signed:
                return arith.DivSIOp(left, right).result
            else:
                return arith.DivUIOp(left, right).result
        elif op == BinaryOp.REM:
            # Use signed or unsigned remainder based on operand signedness
            if is_signed:
                return arith.RemSIOp(left, right).result
            else:
                return arith.RemUIOp(left, right).result

        # Comparison operations
        elif op == BinaryOp.EQ:
            return arith.CmpIOp(arith.CmpIPredicate.eq, left, right).result
        elif op == BinaryOp.NE:
            return arith.CmpIOp(arith.CmpIPredicate.ne, left, right).result
        elif op == BinaryOp.LT:
            # Use signed or unsigned comparison based on operand signedness
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.slt, left, right).result
            else:
                return arith.CmpIOp(arith.CmpIPredicate.ult, left, right).result
        elif op == BinaryOp.LE:
            # Use signed or unsigned comparison based on operand signedness
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sle, left, right).result
            else:
                return arith.CmpIOp(arith.CmpIPredicate.ule, left, right).result
        elif op == BinaryOp.GT:
            # Use signed or unsigned comparison based on operand signedness
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sgt, left, right).result
            else:
                return arith.CmpIOp(arith.CmpIPredicate.ugt, left, right).result
        elif op == BinaryOp.GE:
            # Use signed or unsigned comparison based on operand signedness
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sge, left, right).result
            else:
                return arith.CmpIOp(arith.CmpIPredicate.uge, left, right).result

        # Logical operations (convert to i1 first, then perform logical op)
        elif op == BinaryOp.AND:
            # Logical AND - convert operands to i1 (boolean) first
            # Convert left to i1 by comparing with zero (sgt: signed greater than)
            if left.type != ir.IntegerType.get_signless(1):
                zero_left = arith.ConstantOp(left.type, 0).result
                left = arith.CmpIOp(arith.CmpIPredicate.sgt, left, zero_left).result
            # Convert right to i1 by comparing with zero (sgt: signed greater than)
            if right.type != ir.IntegerType.get_signless(1):
                zero_right = arith.ConstantOp(right.type, 0).result
                right = arith.CmpIOp(arith.CmpIPredicate.sgt, right, zero_right).result
            # Now both are i1, perform logical AND
            return arith.AndIOp(left, right).result
        elif op == BinaryOp.OR:
            # Logical OR - convert operands to i1 (boolean) first
            # Convert left to i1 by comparing with zero (sgt: signed greater than)
            if left.type != ir.IntegerType.get_signless(1):
                zero_left = arith.ConstantOp(left.type, 0).result
                left = arith.CmpIOp(arith.CmpIPredicate.sgt, left, zero_left).result
            # Convert right to i1 by comparing with zero (sgt: signed greater than)
            if right.type != ir.IntegerType.get_signless(1):
                zero_right = arith.ConstantOp(right.type, 0).result
                right = arith.CmpIOp(arith.CmpIPredicate.sgt, right, zero_right).result
            # Now both are i1, perform logical OR
            return arith.OrIOp(left, right).result

        # Bitwise operations (use arith dialect for standard operations)
        elif op == BinaryOp.BIT_AND:
            return arith.AndIOp(left, right).result
        elif op == BinaryOp.BIT_OR:
            return arith.OrIOp(left, right).result
        elif op == BinaryOp.BIT_XOR:
            return arith.XOrIOp(left, right).result

        # Shift operations (use arith dialect for standard operations)
        elif op == BinaryOp.LSHIFT:
            return arith.ShLIOp(left, right).result
        elif op == BinaryOp.RSHIFT:
            # Use arithmetic or logical shift based on operand signedness
            if is_signed:
                return arith.ShRSIOp(left, right).result  # Arithmetic (signed) shift
            else:
                return arith.ShRUIOp(left, right).result  # Logical (unsigned) shift

        else:
            raise NotImplementedError(f"Binary operation not yet supported: {op}")

    def _convert_unary_op(self, op: UnaryOp, operand: ir.Value) -> ir.Value:
        """Convert unary operation to appropriate MLIR operation"""
        if op == UnaryOp.NEG:
            # Arithmetic negation
            zero = arith.ConstantOp(operand.type, 0).result
            return arith.SubIOp(zero, operand).result
        elif op == UnaryOp.NOT:
            # Logical NOT - convert operand to i1 first, then invert
            # Convert to i1 by comparing with zero if not already i1 (sgt: signed greater than)
            if operand.type != ir.IntegerType.get_signless(1):
                zero = arith.ConstantOp(operand.type, 0).result
                operand = arith.CmpIOp(arith.CmpIPredicate.sgt, operand, zero).result
            # Now operand is i1, invert it (XOR with 1)
            one_i1 = arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result
            return arith.XOrIOp(operand, one_i1).result
        elif op == UnaryOp.BIT_NOT:
            # Bitwise NOT (invert all bits)
            all_ones = arith.ConstantOp(operand.type, -1).result
            return arith.XOrIOp(operand, all_ones).result

        # # Type cast operations
        # elif op == UnaryOp.SIGNED_CAST:
        #     # Cast to signed interpretation
        #     # For now, just return operand (type system handles interpretation)
        #     return operand
        # elif op == UnaryOp.UNSIGNED_CAST:
        #     # Cast to unsigned interpretation
        #     return operand
        # elif op == UnaryOp.F32_CAST:
        #     # Cast to f32
        #     if operand.type != ir.F32Type.get():
        #         return arith.SIToFPOp(ir.F32Type.get(), operand).result
        #     return operand
        # elif op == UnaryOp.F64_CAST:
        #     # Cast to f64
        #     if operand.type != ir.F64Type.get():
        #         return arith.SIToFPOp(ir.F64Type.get(), operand).result
        #     return operand

        else:
            raise NotImplementedError(f"Unary operation not yet supported: {op}")

    def _convert_do_while(self, stmt: DoWhileStmt) -> None:
        """
        Convert do-while loop to scf.while or scf.for operation

        CADL do-while semantics: body executes at least once, condition checked after.
        The condition uses variables defined in the body (like i_).

        If the loop matches a for-loop pattern, emit scf.for directly.
        Otherwise, use scf.while with a first_iteration flag to ensure at least one execution.
        """
        # First, validate that only loop-carried variables are modified in body
        self.loop_transformer.validate_loop_body_assignments(stmt)

        # Analyze the pattern and try to raise to scf.for
        for_pattern = self.loop_transformer.analyze_and_detect_for_pattern(stmt)

        if for_pattern:
            # Emit scf.for directly
            self.loop_transformer.emit_scf_for(stmt, for_pattern)
        else:
            # Fallback to scf.while
            self.loop_transformer.emit_scf_while(stmt)

    # def _convert_for_loop(self, stmt: ForStmt) -> None:
    #     """Convert for loop to appropriate MLIR constructs"""
    #     # Push new scope for loop
    #     self.push_scope()

    #     # Execute initialization
    #     self._convert_stmt(stmt.init)

    #     # For now, convert to scf.while (more general than scf.for)
    #     # TODO: Detect when we can use scf.for for better optimization

    #     # Create condition check function
    #     def create_while_body():
    #         # Check condition
    #         condition_val = self._convert_expr(stmt.condition)

    #         # Create while operation
    #         # For simplicity, use empty arguments for now
    #         while_op = scf.WhileOp([], [])

    #         # Before region: condition check
    #         before_block = while_op.before.blocks.append()
    #         with ir.InsertionPoint(before_block):
    #             scf.ConditionOp(condition_val, [])

    #         # After region: body + update
    #         after_block = while_op.after.blocks.append()
    #         with ir.InsertionPoint(after_block):
    #             # Execute loop body
    #             self._convert_stmt_list(stmt.body)

    #             # Execute update statement
    #             self._convert_stmt(stmt.update)

    #             # Yield (no arguments for this simple case)
    #             scf.YieldOp([])

    #     create_while_body()

    #     # Pop loop scope
    #     self.pop_scope()

    def _convert_index_expr(self, expr: IndexExpr) -> ir.Value:
        """
        Convert IndexExpr to appropriate MLIR operation based on the base expression

        Handles special cases:
        - _irf[rs] -> aps.CpuRfRead
        - _mem[addr] -> memref.LoadOp
        - regular array[idx] -> memref.LoadOp
        """
        # Check if this is a special builtin operation
        if isinstance(expr.expr, IdentExpr):
            base_name = expr.expr.name

            if base_name == "_irf":
                # Integer register file read: _irf[rs] -> aps.CpuRfRead
                if len(expr.indices) != 1:
                    raise ValueError("_irf access requires exactly one index")

                # Convert the register index
                reg_index = self._convert_expr(expr.indices[0])

                # Determine result type (assume i32 for now, could be made configurable)
                result_type = ir.IntegerType.get_signless(32)

                # Create APS register file read operation
                return aps.CpuRfRead(result_type, reg_index).result

            elif base_name == "_mem":
                # CPU memory read: _mem[addr] -> aps.memload %cpu_mem[%addr]
                if len(expr.indices) != 1:
                    raise ValueError("_mem access requires exactly one index")
                memref = self.get_cpu_memory_instance()
                indices = [self._convert_expr(expr.indices[0])]

            else:
                # Regular array/memref indexing - check if it's a global array access
                symbol_value = self.get_symbol(base_name)
                if isinstance(symbol_value, str):  # It's a global reference
                    memref = self._get_global_reference(symbol_value)
                else:
                    memref = self._convert_expr(expr.expr)
                indices = [self._convert_expr(idx) for idx in expr.indices]
        else:
            # Non-identifier base expression (e.g., function_call()[idx])
            memref = self._convert_expr(expr.expr)
            indices = [self._convert_expr(idx) for idx in expr.indices]

        # Common memload logic: determine element type and load
        if hasattr(memref.type, 'element_type'):
            element_type = memref.type.element_type
        else:
            element_type = ir.IntegerType.get_signless(32)

        return aps.MemLoad(element_type, memref, indices).result

    def _get_memref_element_type(self, memref_value: ir.Value) -> Optional[ir.Type]:
        """
        Get the element type of a memref value.

        Args:
            memref_value: An MLIR value of memref type

        Returns:
            The element type of the memref, or None if not a memref
        """
        if hasattr(memref_value.type, 'element_type'):
            return memref_value.type.element_type
        return None

    def _convert_index_assignment(self, lhs: IndexExpr, rhs_value: ir.Value) -> None:
        """
        Convert indexed assignment to appropriate MLIR operation

        Handles special cases:
        - _irf[rd] = value -> aps.CpuRfWrite
        - _mem[addr] = value -> memref.StoreOp
        - regular array[idx] = value -> memref.StoreOp
        """
        # Check if this is a special builtin operation
        if isinstance(lhs.expr, IdentExpr):
            base_name = lhs.expr.name

            if base_name == "_irf":
                # Integer register file write: _irf[rd] = value -> aps.CpuRfWrite
                if len(lhs.indices) != 1:
                    raise ValueError("_irf assignment requires exactly one index")

                # Convert the register index
                reg_index = self._convert_expr(lhs.indices[0])

                # Create APS register file write operation
                aps.CpuRfWrite(reg_index, rhs_value)

            elif base_name == "_mem":
                # CPU memory write: _mem[addr] = value -> aps.memstore %value, %cpu_mem[%addr]
                if len(lhs.indices) != 1:
                    raise ValueError("_mem assignment requires exactly one index")

                # Convert the memory address
                addr = self._convert_expr(lhs.indices[0])

                # Get CPU memory instance
                cpu_mem = self.get_cpu_memory_instance()

                # Ensure value type matches memref element type
                target_type = self._get_memref_element_type(cpu_mem)
                if target_type is not None:
                    rhs_value = self._convert_value_to_target_type(rhs_value, target_type)

                # Generate APS memstore operation
                aps.MemStore(rhs_value, cpu_mem, [addr])

            else:
                # Regular array/memref assignment
                if isinstance(lhs.expr, IdentExpr):
                    # Get the memref itself, not a loaded value
                    base_value = self._get_memref_for_symbol(lhs.expr.name)
                else:
                    base_value = self._convert_expr(lhs.expr)
                indices = [self._convert_expr(idx) for idx in lhs.indices]

                # Ensure value type matches memref element type
                target_type = self._get_memref_element_type(base_value)
                if target_type is not None:
                    rhs_value = self._convert_value_to_target_type(rhs_value, target_type)

                # Use APS memstore for regular array assignment
                aps.MemStore(rhs_value, base_value, indices)
        else:
            # Complex base expression
            if isinstance(lhs.expr, IdentExpr):
                # Get the memref itself, not a loaded value
                base_value = self._get_memref_for_symbol(lhs.expr.name)
            else:
                base_value = self._convert_expr(lhs.expr)
            indices = [self._convert_expr(idx) for idx in lhs.indices]

            # Ensure value type matches memref element type
            target_type = self._get_memref_element_type(base_value)
            if target_type is not None:
                rhs_value = self._convert_value_to_target_type(rhs_value, target_type)

            # Use APS memstore for general indexed assignment
            aps.MemStore(rhs_value, base_value, indices)

    def _is_burst_operation(self, stmt: AssignStmt) -> bool:
        """
        Detect if an assignment is a burst operation

        Burst load:  mem[start +: ] = _burst_read[cpu_addr +: length]
        Burst store: _burst_write[cpu_addr +: length] = mem[start +: ]
        """
        # Check for burst read (RHS is _burst_read with range slice)
        if isinstance(stmt.rhs, RangeSliceExpr) and isinstance(stmt.rhs.expr, IdentExpr):
            if stmt.rhs.expr.name == "_burst_read":
                return True

        # Check for burst write (LHS is _burst_write with range slice)
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            if stmt.lhs.expr.name == "_burst_write":
                return True

        return False

    def _convert_burst_operation(self, stmt: AssignStmt) -> None:
        """
        Convert burst read/write operations to MLIR aps.memburstload/memburststore

        Burst load:  buffer[offset +: ] = _burst_read[cpu_addr +: length]
                     -> aps.memburstload %cpu_addr, %buffer[%offset], %length

        Burst store: _burst_write[cpu_addr +: length] = buffer[offset +: ]
                     -> aps.memburststore %buffer[%offset], %cpu_addr, %length
        """
        # Burst load: RHS is _burst_read
        if isinstance(stmt.rhs, RangeSliceExpr) and isinstance(stmt.rhs.expr, IdentExpr):
            if stmt.rhs.expr.name == "_burst_read":
                self._convert_burst_load(stmt)
                return

        # Burst store: LHS is _burst_write
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            if stmt.lhs.expr.name == "_burst_write":
                self._convert_burst_store(stmt)
                return

        raise ValueError("Invalid burst operation pattern")

    def _extract_literal_value(self, expr: Expr) -> int:
        """
        Extract constant integer value from LitExpr.

        Note: Parser validation ensures burst lengths are LitExpr at parse time,
        so we can directly extract the value without MLIR conversion.
        """
        if not isinstance(expr, LitExpr):
            raise ValueError(f"Expected LitExpr, got {type(expr).__name__}")

        if not hasattr(expr.literal.lit, 'value'):
            raise ValueError(f"Literal has no value attribute")

        return expr.literal.lit.value

    def _convert_burst_load(self, stmt: AssignStmt) -> None:
        """
        Convert burst load: buffer[offset +: ] = _burst_read[cpu_addr +: length]
        to: aps.memburstload %cpu_addr, %buffer[%offset], %length

        Burst length must be a compile-time constant.
        """
        lhs = stmt.lhs  # buffer[offset +: ]
        rhs = stmt.rhs  # _burst_read[cpu_addr +: length]

        if not isinstance(lhs, RangeSliceExpr):
            raise ValueError("Burst load LHS must be a range slice expression")
        if not isinstance(rhs, RangeSliceExpr):
            raise ValueError("Burst load RHS must be a range slice expression")

        # Extract components from RHS (_burst_read[cpu_addr +: length])
        cpu_addr = self._convert_expr(rhs.start)
        if rhs.length is None:
            raise ValueError("Burst read must have explicit length")

        # Extract constant length (parser ensures it's a LitExpr)
        rhs_length_val = self._extract_literal_value(rhs.length)

        # Extract components from LHS (buffer[offset +: ])
        # Get memref for the buffer
        if isinstance(lhs.expr, IdentExpr):
            buffer_name = lhs.expr.name
            buffer_memref = self._get_memref_for_symbol(buffer_name)
        else:
            buffer_memref = self._convert_expr(lhs.expr)

        start_offset = self._convert_expr(lhs.start)

        # Validate length if specified on both sides
        if lhs.length is not None:
            lhs_length_val = self._extract_literal_value(lhs.length)
            if lhs_length_val != rhs_length_val:
                raise ValueError(f"Burst length mismatch: buffer[+:{lhs_length_val}] = _burst_read[+:{rhs_length_val}]")

        # Convert constant value back to MLIR constant for the operation
        i32_type = ir.IntegerType.get_signless(32)
        length = arith.ConstantOp(i32_type, rhs_length_val).result

        # Generate aps.memburstload operation
        # Arguments: cpu_addr, memrefs (as list), start, length
        # Wrap single memref in a list to support variadic memrefs
        burst_op = aps.MemBurstLoad(cpu_addr, [buffer_memref], start_offset, length)

        # Apply pending directives as attributes (same pattern as loop_transform.py)
        if self.pending_directives:
            for directive in self.pending_directives:
                attr_name = directive.name
                if directive.expr and isinstance(directive.expr, LitExpr):
                    value = directive.expr.literal.lit.value
                    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
                    burst_op.operation.attributes[attr_name] = attr
                else:
                    burst_op.operation.attributes[attr_name] = ir.BoolAttr.get(True)
            self.pending_directives = []

    def _convert_burst_store(self, stmt: AssignStmt) -> None:
        """
        Convert burst store: _burst_write[cpu_addr +: length] = buffer[offset +: ]
        to: aps.memburststore %buffer[%offset], %cpu_addr, %length

        Burst length must be a compile-time constant.
        """
        lhs = stmt.lhs  # _burst_write[cpu_addr +: length]
        rhs = stmt.rhs  # buffer[offset +: ]

        if not isinstance(lhs, RangeSliceExpr):
            raise ValueError("Burst store LHS must be a range slice expression")
        if not isinstance(rhs, RangeSliceExpr):
            raise ValueError("Burst store RHS must be a range slice expression")

        # Extract components from LHS (_burst_write[cpu_addr +: length])
        cpu_addr = self._convert_expr(lhs.start)
        if lhs.length is None:
            raise ValueError("Burst write must have explicit length")

        # Extract constant length (parser ensures it's a LitExpr)
        lhs_length_val = self._extract_literal_value(lhs.length)

        # Extract components from RHS (buffer[offset +: ])
        # Get memref for the buffer
        if isinstance(rhs.expr, IdentExpr):
            buffer_name = rhs.expr.name
            buffer_memref = self._get_memref_for_symbol(buffer_name)
        else:
            buffer_memref = self._convert_expr(rhs.expr)

        start_offset = self._convert_expr(rhs.start)

        # Validate length if specified on both sides
        if rhs.length is not None:
            rhs_length_val = self._extract_literal_value(rhs.length)
            if lhs_length_val != rhs_length_val:
                raise ValueError(f"Burst length mismatch: _burst_write[+:{lhs_length_val}] = buffer[+:{rhs_length_val}]")

        # Convert constant value back to MLIR constant for the operation
        i32_type = ir.IntegerType.get_signless(32)
        length = arith.ConstantOp(i32_type, lhs_length_val).result

        # Generate aps.memburststore operation
        # Arguments: memrefs (as list), start, cpu_addr, length
        # Wrap single memref in a list to support variadic memrefs
        burst_op = aps.MemBurstStore([buffer_memref], start_offset, cpu_addr, length)

        # Apply pending directives as attributes (same pattern as loop_transform.py)
        if self.pending_directives:
            for directive in self.pending_directives:
                attr_name = directive.name
                if directive.expr and isinstance(directive.expr, LitExpr):
                    value = directive.expr.literal.lit.value
                    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
                    burst_op.operation.attributes[attr_name] = attr
                else:
                    burst_op.operation.attributes[attr_name] = ir.BoolAttr.get(True)
            self.pending_directives = []

    def _convert_range_slice_assignment(self, lhs: RangeSliceExpr, rhs_value: ir.Value) -> None:
        """Handle regular range slice assignments (not burst operations)"""
        # For now, this is not a common case - range slices are primarily used for burst ops
        raise NotImplementedError("Non-burst range slice assignments not yet supported")

    def _get_memref_for_symbol(self, symbol_name: str) -> ir.Value:
        """Get memref value for a symbol (handling both local and global variables)"""
        symbol_value = self.get_symbol(symbol_name)

        if symbol_value is None:
            raise ValueError(f"Undefined symbol: {symbol_name}")

        if isinstance(symbol_value, str):
            # It's a global reference - get the memref
            # Find the static variable definition to get the type
            static_var = None
            for static in self.proc.statics.values():
                if static.id == symbol_name:
                    static_var = static
                    break

            if static_var:
                # Type is inferred from the GlobalOp in _get_global_reference
                return self._get_global_reference(symbol_value)
            else:
                raise ValueError(f"Cannot find static definition for global: {symbol_name}")
        else:
            # It's a local value
            return symbol_value

    def _convert_slice_expr(self, expr: SliceExpr) -> ir.Value:
        """
        Convert slice expression to MLIR bit extraction

        Handles expressions like z[31:31] (extract bit 31) or z[15:8] (extract bits 15 to 8)
        Uses comb.ExtractOp for constant indices, arith dialect for dynamic indices
        """
        # Convert the base expression
        base_value = self._convert_expr(expr.expr)

        # Convert start and end indices
        start_val = self._convert_expr(expr.start)
        end_val = self._convert_expr(expr.end)

        # For now, we'll assume constant indices (most common case)
        # and use comb.ExtractOp for bit extraction

        # Get the constant values if possible
        if (hasattr(expr.start, 'literal') and hasattr(expr.start.literal, 'lit') and
            hasattr(expr.end, 'literal') and hasattr(expr.end.literal, 'lit')):

            start_bit = expr.start.literal.lit.value
            end_bit = expr.end.literal.lit.value

            # Determine the width of the extracted slice
            if start_bit == end_bit:
                # Single bit extraction - result is i1
                result_type = ir.IntegerType.get_signless(1)
                # Use comb.extract to get a single bit
                return comb.ExtractOp(result_type, base_value, start_bit).result
            else:
                # Multi-bit extraction
                width = abs(start_bit - end_bit) + 1
                result_type = ir.IntegerType.get_signless(width)
                low_bit = min(start_bit, end_bit)
                # Use comb.extract to get multiple bits
                return comb.ExtractOp(result_type, base_value, low_bit).result
        else:
            # Dynamic slice indices - more complex, use shift and mask
            # This is a fallback for non-constant indices
            # For now, assume single bit extraction and return bit 0
            result_type = ir.IntegerType.get_signless(1)
            # Extract bit at dynamic position using shift and mask
            # result = (base_value >> start_val) & 1
            shifted = arith.ShRUIOp(base_value, start_val).result
            one = arith.ConstantOp(base_value.type, 1).result
            return arith.AndIOp(shifted, one).result

    def _convert_if_expr(self, expr: IfExpr) -> ir.Value:
        """
        Convert if expression to MLIR conditional operation

        Converts CADL if expressions like:
            if z_neg {x + y_shift} else {x - y_shift}

        Uses arith.SelectOp for conditional selection
        """
        # Convert the condition
        condition = self._convert_expr(expr.condition)

        # Convert then and else branches
        then_value = self._convert_expr(expr.then_branch)
        else_value = self._convert_expr(expr.else_branch)

        # Ensure condition is a single bit (i1)
        # If the condition is not i1, we need to check if it's greater than zero
        if condition.type != ir.IntegerType.get_signless(1):
            # Convert to boolean by comparing with zero (sgt: signed greater than)
            zero = arith.ConstantOp(condition.type, 0).result
            condition = arith.CmpIOp(arith.CmpIPredicate.sgt, condition, zero).result

        # Use arith.SelectOp for conditional selection
        # SelectOp selects then_value when condition is true, else_value when false
        return arith.SelectOp(condition, then_value, else_value).result

    def _convert_select_expr(self, expr: SelectExpr) -> ir.Value:
        """
        Convert select expression to MLIR conditional operations

        Converts CADL select expressions like:
            sel {
                x == 0: 10,
                x < 10: 20,
                x < 20: 30,
                1: 40,
            }

        To a chain of arith.SelectOp operations, evaluated from first to last.
        The first matching condition takes precedence (short-circuit evaluation).
        """
        # Start with the default value
        result = self._convert_expr(expr.default)

        # Process arms in reverse order to build the select chain
        # This ensures the first matching condition takes precedence
        for cond_expr, val_expr in reversed(expr.arms):
            condition = self._convert_expr(cond_expr)
            value = self._convert_expr(val_expr)

            # Ensure condition is i1
            if condition.type != ir.IntegerType.get_signless(1):
                zero = arith.ConstantOp(condition.type, 0).result
                condition = arith.CmpIOp(arith.CmpIPredicate.sgt, condition, zero).result

            # Select: if condition then value else result
            result = arith.SelectOp(condition, value, result).result

        return result


def convert_cadl_to_mlir(proc: Proc, run_cse: bool = True) -> ir.Module:
    """
    Main entry point for converting CADL Proc to MLIR Module

    Args:
        proc: CADL processor AST to convert
        run_cse: Whether to run Common Subexpression Elimination pass (default: True)

    Returns:
        MLIR module containing the converted representation
    """
    converter = CADLMLIRConverter()
    module = converter.convert_proc(proc)

    # Apply CSE optimization pass if requested
    if run_cse:
        with converter.context:
            from circt.passmanager import PassManager
            pm = PassManager.parse("builtin.module(cse)")
            # TODO: We should to cannanicalize in the end, but for now, we skip it
            # pm = PassManager.parse("builtin.module(canonicalize,cse,canonicalize,cse)") 
            pm.run(module.operation)

    return module