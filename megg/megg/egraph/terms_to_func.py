"""
MeggEGraph Expression Tree to MLIR Conversion.

This module converts expression trees extracted from MeggEGraph
to MLIR operations, replacing the previous term-based approach.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from megg.utils import MBlock, MModule, MOperation, OperationType, MValue, IRBuilder,MType
from megg.egraph.megg_egraph import ExpressionNode, MeggEGraph
from megg.egraph.func_to_terms import megg_to_mlir_type
from megg.egraph.datatype import VoidType


@dataclass
class ExprTreeToMLIR:
    """
    Convert MeggEGraph expression trees to MLIR operations.

    This replaces the previous TermsToFunc approach with a cleaner
    architecture based on extracted expression trees.

    New architecture: Supports top-level block wrapping for sequential execution order.
    """

    # Original function to reconstruct
    original_func: MOperation

    # Expression trees for function body (top-level block)
    body_exprs: List[ExpressionNode]

    # Output terms (for return value extraction)
    output_terms: List[Any] = field(default_factory=list)

    # Mapping from expression node IDs to MLIR SSA values
    # Cache expression reconstruction: (expr_node_id, block_id) -> MValue
    # Block-aware caching to avoid SSA scope violations from shared e-graph terms
    expr_to_ssa: Dict[Tuple[str, int], MValue] = field(default_factory=dict)

    # Reconstructed function
    reconstructed_func: Optional[MOperation] = None

    # Constants cache to avoid duplication
    constants_cache: Dict[tuple, MValue] = field(default_factory=dict)

    # Loop context tracking for block arguments
    loop_contexts: Dict[str, MBlock] = field(default_factory=dict)
    loop_block_stack: List[Tuple[MBlock, Set[str]]] = field(default_factory=list)
    builder: IRBuilder = IRBuilder()
    instr_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Track the current reconstruction context (which block we're reconstructing for)
    # This is used to prevent pure operations from being created in nested blocks
    current_block_context: Optional[MBlock] = None

    # Track the first nested operation (for/if/while) created in each block
    # Maps block ID to the MLIR operation object
    # Used to insert operations before nested structures
    block_first_nested_op: Dict[int, Any] = field(default_factory=dict)

    # Target module context (optional, for ensuring correct context when appending)
    target_module: Optional[Any] = None

    @classmethod
    def reconstruct(cls, original_func: MOperation, body_exprs: List[ExpressionNode], output_terms: List[Any] = None,
                    instr_properties: Optional[Dict[str, Dict[str, Any]]] = None,
                    target_module: Optional[Any] = None) -> MOperation:
        """
        Reconstruct a FuncOp from MeggEGraph expression trees.

        Args:
            original_func: Original MLIR function
            body_exprs: List of body expression trees (typically one top-level Block)
            output_terms: Optional list of output terms for return value extraction
            instr_properties: Optional custom instruction properties
            target_module: Optional target module to ensure correct MLIR context

        Returns:
            Reconstructed MLIR FuncOp
        """
        converter = cls(
            original_func=original_func,
            body_exprs=body_exprs,
            output_terms=output_terms or [],
            instr_properties=instr_properties or {},
            target_module=target_module
        )
        converter._reconstruct()
        return converter.reconstructed_func

    def _has_custom_instruction(self, exprs: List[ExpressionNode]) -> bool:
        """Check if expression tree contains custom instructions."""
        def traverse(expr: ExpressionNode) -> bool:
            if expr.op == 'Custom_instr':
                return True
            for child in expr.children:
                if traverse(child):
                    return True
            return False

        for expr in exprs:
            if traverse(expr):
                return True
        return False

    def _reconstruct(self):
        """Main reconstruction method."""
        import logging
        logger = logging.getLogger(__name__)

        # Extract function metadata
        func_name = self.original_func.symbol_name
        logger.info(f"[RECONSTRUCT] Starting reconstruction of {func_name}")
        func_type = self.original_func.get_function_type
        arg_types = func_type.get_inputs()
        outputs = func_type.get_results()
        logger.info(f"[RECONSTRUCT] Function has {len(arg_types)} args, {len(outputs)} outputs")

        # Check if function contains custom instructions
        logger.info(f"[RECONSTRUCT] Checking for custom instructions...")
        has_custom_instr = self._has_custom_instruction(self.body_exprs)
        logger.info(f"[RECONSTRUCT] Has custom instructions: {has_custom_instr}")

        # If has custom instructions, replace memref args with llvm.ptr for ABI compatibility
        if has_custom_instr:
            modified_arg_types = []
            for arg_type in arg_types:
                type_str = str(arg_type)
                if type_str.startswith('memref<'):
                    # Use LLVM opaque pointer instead of memref descriptor
                    # This prevents ABI mismatch (memref=5 params vs ptr=1 param)
                    modified_arg_types.append(self.builder.llvm_ptr())
                else:
                    modified_arg_types.append(arg_type)
            arg_types = modified_arg_types

        # Ensure we're using the correct MLIR context
        if self.target_module is not None:
            # Set builder context to match the target module
            with self.builder.set_context(self.target_module.get_context()):
                self.reconstructed_func, entry_block = self.builder.create_function(
                    func_name, arg_types, outputs)
        else:
            self.reconstructed_func, entry_block = self.builder.create_function(
                func_name, arg_types, outputs)
        # Map function arguments (store by argument index as string key)
        for i, arg in enumerate(entry_block.arguments):
            arg_key = f"arg_{i}"
            self.expr_to_ssa[arg_key] = arg

        # Set entry block as the initial context (top-level block)
        self.current_block_context = entry_block

        # Reconstruct function body (顶层 body_exprs 其实就是一个 Block)
        logger.info(f"[RECONSTRUCT] Reconstructing {len(self.body_exprs)} body expressions")
        last_expr_results: List[MValue] = []
        for idx, body_expr in enumerate(self.body_exprs):
            logger.info(f"[RECONSTRUCT] Processing body_expr {idx+1}/{len(self.body_exprs)}: op={body_expr.op}")
            result = self._reconstruct_expr(body_expr, entry_block)
            logger.info(f"[RECONSTRUCT] Completed body_expr {idx+1}/{len(self.body_exprs)}")
            if result is None:
                continue
            if isinstance(result, MValue):
                last_expr_results = [result]
            elif hasattr(result, 'results'):
                op_results = result.results
                if op_results:
                    last_expr_results = op_results

        # Ensure the entry block is properly terminated. If the function is
        # expected to return values but the extraction did not produce an
        # explicit func.return, use the last reconstructed SSA values as the
        # implicit return operands (common for whole-function custom_instr).
        logger.info(f"[RECONSTRUCT] Ensuring block termination...")
        self._ensure_block_terminated(entry_block, outputs, last_expr_results)
        logger.info(f"[RECONSTRUCT] Reconstruction complete for {func_name}")

    def _reconstruct_expr(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        Recursively reconstruct an expression tree to MLIR operations.

        Args:
            expr: Expression node to reconstruct
            block: Block to insert operations into

        Returns:
            MLIR SSA value representing the expression
        """
        # Check if already reconstructed (but skip cache for literals - they must be scope-aware)
        # Use block-aware cache key to avoid SSA scope violations from shared e-graph terms
        target_block = self.current_block_context if self.current_block_context else block
        cache_key = (expr.node_id, id(target_block))

        if expr.op != 'Lit' and cache_key in self.expr_to_ssa:
            cached = self.expr_to_ssa[cache_key]
            # Special marker for terminators that return None
            if cached == "PROCESSED":
                return None
            return cached

        # print(f"Reconstructing expr {expr.node_id} op={expr.op} type={expr.dtype}")
        # Dispatch based on operation
        op_handlers = {
            # Literals and arguments
            'Lit': self._reconstruct_literal,
            'Arg': self._reconstruct_argument,
            'Loop_index': self._reconstruct_loop_index,
            'Loop_iter_arg': self._reconstruct_loop_iter_arg,
            'Block_arg': self._reconstruct_block_arg,

            # Binary arithmetic
            'Add': self._reconstruct_binary_op,
            'Sub': self._reconstruct_binary_op,
            'Mul': self._reconstruct_binary_op,
            'Div': self._reconstruct_binary_op,
            'Rem': self._reconstruct_binary_op,

            # Logical operations (bitwise)
            'And': self._reconstruct_binary_op,
            'And_': self._reconstruct_binary_op,  # Alternative name from extraction
            'Or': self._reconstruct_binary_op,
            'Or_': self._reconstruct_binary_op,  # Alternative name from extraction
            'Xor': self._reconstruct_binary_op,
            'Xor_': self._reconstruct_binary_op,  # Alternative name from extraction

            # Shift operations
            'Shl': self._reconstruct_binary_op,
            'ShrSI': self._reconstruct_binary_op,
            'Shrsi': self._reconstruct_binary_op,  # Alternative capitalization
            'ShrUI': self._reconstruct_binary_op,
            'Shrui': self._reconstruct_binary_op,  # Alternative capitalization

            # Unary arithmetic
            'Neg': self._reconstruct_unary_op,

            # Comparison
            'Cmp': self._reconstruct_comparison,
            'Select': self._reconstruct_select,

            # Cast operations
            'Index_cast': self._reconstruct_cast,
            'SIToFP': self._reconstruct_cast,
            'Sitofp': self._reconstruct_cast,  # Alternative capitalization
            'UIToFP': self._reconstruct_cast,
            'Uitofp': self._reconstruct_cast,  # Alternative capitalization
            'FPToSI': self._reconstruct_cast,
            'Fptosi': self._reconstruct_cast,  # Alternative capitalization
            'FPToUI': self._reconstruct_cast,
            'Fptoui': self._reconstruct_cast,  # Alternative capitalization
            'ExtSI': self._reconstruct_cast,
            'Extsi': self._reconstruct_cast,  # Alternative capitalization
            'ExtUI': self._reconstruct_cast,
            'Extui': self._reconstruct_cast,  # Alternative capitalization
            'TruncI': self._reconstruct_cast,
            'Trunci': self._reconstruct_cast,  # Alternative capitalization
            'Bitcast': self._reconstruct_cast,

            # Control flow
            'If_': self._reconstruct_if,
            'While_': self._reconstruct_while,
            'For_': self._reconstruct_for,
            'For_with_carry': self._reconstruct_for_with_carry,
            'Affine_for': self._reconstruct_affine_for,
            'Affine_for_with_carry': self._reconstruct_affine_for_with_carry,
            'Block': self._reconstruct_block,
            'Yield_': self._reconstruct_yield,
            'Affine_yield': self._reconstruct_affine_yield,
            'Custom_instr': self._reconstruct_custom_instr,
            'Return_': self._reconstruct_return,
            'Condition': self._reconstruct_condition,

            # MemRef operations
            'Alloc': self._reconstruct_memref_alloc,
            'Alloca': self._reconstruct_memref_alloca,
            'Get_global': self._reconstruct_memref_get_global,
            'Store': self._reconstruct_memref_store,
            'Load': self._reconstruct_memref_load,

            # Vec operation
            "Vec": self._reconstruct_vec,
        }

        # print(f"Dispatching op: {expr.op} for node {expr.node_id}")
        handler = op_handlers.get(expr.op)
        if handler is None:
            raise ValueError(f"Unsupported operation: {expr.op!r}")

        result = handler(expr, block)

        # print(f"  Result of {expr.op} ({expr.node_id}): {result}")

        # Cache the result, but NOT for literals (constants must be scope-aware)
        # Use block-aware cache key (same as computed above)
        if result is not None and expr.op != 'Lit':
            self.expr_to_ssa[cache_key] = result
        elif expr.op in ['Yield_', 'Affine_yield', 'Return_', 'Condition']:
            # Mark terminators as processed to avoid re-execution
            self.expr_to_ssa[cache_key] = "PROCESSED"

        return result

    @staticmethod
    def _canonical_op(op: Optional[str]) -> str:
        if not op:
            return ''
        if op == 'For_':
            return 'For'
        if op == 'While_':
            return 'While'
        if op == 'Affine_for_':
            return 'Affine_for'
        return op

    def _reconstruct_literal(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct literal constants directly from metadata and dtype."""
        # For literals, always create in the requested block
        # This ensures they are available for yield operations
        target_block = block

        # 从 metadata 获取值
        lit_value = expr.metadata.get('value', None)
        if lit_value is None:
            raise ValueError(
                f"Literal node {expr.node_id} missing 'value' metadata")

        # Get MLIR type using the new wrapper
        mlir_type = megg_to_mlir_type(expr.dtype)

        # Check if it's a float or integer type based on the type string
        type_str = str(mlir_type)
        is_float = any(ft in type_str for ft in ['f16', 'f32', 'f64'])

        # Use context manager for insertion point - insert into target block
        with self.builder.set_insertion_point_to_end(target_block):
            if is_float:
                # Float constant - include block ID to prevent cross-scope reuse
                cache_key = ('float', float(lit_value), type_str, id(target_block))
                if cache_key in self.constants_cache:
                    return self.constants_cache[cache_key]
                # Use builder to create float constant
                const_value = self.builder.constant(
                    float(lit_value), mlir_type._type)
            else:
                # Integer constant - include block ID to prevent cross-scope reuse
                cache_key = ('int', int(lit_value), type_str, id(target_block))
                if cache_key in self.constants_cache:
                    return self.constants_cache[cache_key]
                # Use builder to create integer constant
                const_value = self.builder.constant(int(lit_value), mlir_type)

        # Cache SSA value with block-aware key
        self.constants_cache[cache_key] = const_value
        return const_value

    def _reconstruct_argument(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct function arguments."""
        # Extract argument index from children (first child is the index)
        # print(f"Reconstructing argument {expr}")

        arg_idx = expr.metadata.get('arg_info', None)
        if arg_idx is None:
            raise ValueError(
                f"Arg node {expr.node_id} missing 'arg_info' metadata")
        arg_key = f"arg_{arg_idx}"
        # print(f"  Argument index: {arg_idx} key={arg_key}")
        return self.expr_to_ssa.get(arg_key)

    def _reconstruct_loop_index(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct loop index variable."""
        loop_id_info = expr.metadata.get('arg_info', None)
        loop_block = None
        loop_key = None

        if loop_id_info is not None:
            loop_id = loop_id_info[0] if isinstance(
                loop_id_info, list) and len(loop_id_info) > 0 else loop_id_info
            loop_key = f"loop_{loop_id}"
            loop_block = self.loop_contexts.get(loop_key)

        if loop_block is None and self.loop_block_stack:
            loop_id_str = str(loop_id_info) if loop_id_info is not None else None
            selected_block = None

            # Search through loop_block_stack from outer to inner to find the matching loop
            # This ensures we find the correct loop even when nested
            for block_ctx, block_ids in self.loop_block_stack:
                if loop_id_str and loop_id_str in block_ids:
                    selected_block = block_ctx
                    break

            # CRITICAL FIX: Do NOT fallback to innermost loop if not found!
            # Previously this caused outer loop indices to be replaced with inner loop indices
            if selected_block is None:
                # Log warning and return None instead of using wrong loop
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not find loop_id {loop_id_str} in loop_block_stack. "
                             f"Available ids: {[ids for _, ids in self.loop_block_stack]}")
                return None  # Return None instead of using wrong loop!

            loop_block = selected_block
            if loop_key and loop_block:
                self.loop_contexts[loop_key] = loop_block

        if loop_block and len(loop_block.arguments) > 0:
            result = loop_block.arguments[0]
            return result

        return None

    def _reconstruct_block_arg(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct block argument (loop-carried values)."""
        arg_info = expr.metadata.get('arg_info', None)
        if not arg_info or len(arg_info) < 2:
            return None

        block_id = arg_info[0]
        arg_idx = arg_info[1]

        loop_key = f"loopblock_{block_id}"

        # Get the loop context block
        if loop_key in self.loop_contexts:
            loop_block = self.loop_contexts[loop_key]
            # Block arguments start after the induction variable
            actual_idx = int(arg_idx)

            if actual_idx < len(loop_block.arguments):
                return loop_block.arguments[actual_idx]

        return None

    def _reconstruct_loop_iter_arg(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct loop iteration argument (loop-carried values)."""
        arg_info = expr.metadata.get('arg_info', None)
        if not arg_info or len(arg_info) < 2:
            return None

        loop_id = arg_info[0]
        iter_arg_idx = arg_info[1]

        loop_key = f"loop_{loop_id}"
        # Get the loop context block
        if loop_key in self.loop_contexts:
            loop_block = self.loop_contexts[loop_key]
            # Loop iteration arguments start after the induction variable (index 0)
            # So actual index = iter_arg_idx + 1
            actual_idx = int(iter_arg_idx) + 1
            if actual_idx < len(loop_block.arguments):

                return loop_block.arguments[actual_idx]

        return None

    # ===== Binary Operations =====

    def _type_str_to_mtype(self, type_str: str):
        """Convert a type string (e.g., 'i32', 'index') to an MType object."""
        type_str = type_str.strip()
        if type_str == 'index':
            return self.builder.index()
        elif type_str == 'i32':
            return self.builder.i32()
        elif type_str == 'i64':
            return self.builder.i64()
        elif type_str == 'i1':
            return self.builder.i1()
        elif type_str == 'f32':
            return self.builder.f32()
        elif type_str == 'f64':
            return self.builder.f64()
        elif type_str.startswith('i') and type_str[1:].isdigit():
            width = int(type_str[1:])
            return self.builder.integer(width)
        else:
            # Default to i32 for unknown types
            return self.builder.i32()

    def _ensure_same_type(self, lhs: MValue, rhs: MValue, block: MBlock) -> Tuple[MValue, MValue]:
        """Ensure two operands have the same type, inserting index_cast if needed."""
        lhs_type_str = str(lhs.type)
        rhs_type_str = str(rhs.type)

        # If types match, no conversion needed
        if lhs_type_str == rhs_type_str:
            return lhs, rhs

        # Handle index <-> i32/i64 conversion
        with self.builder.set_insertion_point_to_end(block):
            if lhs_type_str == 'index' and 'i' in rhs_type_str:
                # Convert lhs (index) to rhs's integer type
                target_type = self._type_str_to_mtype(rhs_type_str)
                lhs = self.builder.index_cast(lhs, target_type)
            elif rhs_type_str == 'index' and 'i' in lhs_type_str:
                # Convert rhs (index) to lhs's integer type
                target_type = self._type_str_to_mtype(lhs_type_str)
                rhs = self.builder.index_cast(rhs, target_type)
            # If both are integers of different widths, prefer the wider one
            elif 'i' in lhs_type_str and 'i' in rhs_type_str:
                # Extract bit widths (simple heuristic)
                # For now, keep as-is; more sophisticated width handling could be added
                pass

        return lhs, rhs

    def _reconstruct_binary_op(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct binary arithmetic operations."""
        # Pure operations should be created in the current block context (top-level or loop body)
        # not in nested blocks like scf.if branches
        target_block = self.current_block_context if self.current_block_context else block

        if len(expr.children) < 2:
            return None

        lhs = self._reconstruct_expr(expr.children[0], block)
        rhs = self._reconstruct_expr(expr.children[1], block)

        if lhs is None or rhs is None:
            return None

        # Ensure operands have the same type
        lhs, rhs = self._ensure_same_type(lhs, rhs, target_block)

        # Insert into the target block
        with self.builder.set_insertion_point_to_end(target_block):
            if expr.op == 'Add':
                result = self.builder.add(lhs, rhs)
            elif expr.op == 'Sub':
                result = self.builder.sub(lhs, rhs)
            elif expr.op == 'Mul':
                result = self.builder.mul(lhs, rhs)
            elif expr.op == 'Div':
                result = self.builder.div(lhs, rhs)
            elif expr.op == 'Rem':
                result = self.builder.rem(lhs, rhs)
            elif expr.op in ('And', 'And_'):
                result = self.builder.and_(lhs, rhs)
            elif expr.op in ('Or', 'Or_'):
                result = self.builder.or_(lhs, rhs)
            elif expr.op in ('Xor', 'Xor_'):
                result = self.builder.xor_(lhs, rhs)
            elif expr.op == 'Shl':
                result = self.builder.shl(lhs, rhs)
            elif expr.op in ('ShrSI', 'Shrsi'):
                result = self.builder.shrsi(lhs, rhs)
            elif expr.op in ('ShrUI', 'Shrui'):
                result = self.builder.shrui(lhs, rhs)
            else:
                return None

        return result

    def _reconstruct_unary_op(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct unary arithmetic operations."""
        if len(expr.children) < 1:
            return None

        operand = self._reconstruct_expr(expr.children[0], block)
        if operand is None:
            return None

        mlir_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            if expr.op == 'Neg':
                # Negate by subtracting from zero
                # Check type to determine float vs int
                type_str = str(mlir_type)
                is_float = any(ft in type_str for ft in ['f16', 'f32', 'f64'])

                x = 0.0 if is_float else 0
                zero = self.builder.constant(x, mlir_type)
                result = self.builder.sub(zero, operand)
                return result
            else:
                raise RuntimeError(f"meet unsupport unary op: {expr.op}")

        return None

    def _reconstruct_comparison(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct comparison operations."""
        # print(
        #     f"Reconstructing comparison {expr.node_id} with children {[child.node_id for child in expr.children]}")
        if len(expr.children) < 2:
            return None

        lhs = self._reconstruct_expr(expr.children[0], block)
        rhs = self._reconstruct_expr(expr.children[1], block)

        # print(f"  LHS: {lhs}, RHS: {rhs}")

        if lhs is None or rhs is None:
            return None

        # Extract predicate from metadata
        predicate = expr.metadata.get('cmp_info', None)

        # print(f"  LHS: {lhs}, RHS: {rhs}, predicate: {predicate}")

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.cmp(predicate, lhs, rhs)

        return result

    def _reconstruct_select(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct arith.select operation (ternary conditional)."""
        if len(expr.children) < 3:
            return None

        # select(condition, true_value, false_value)
        cond = self._reconstruct_expr(expr.children[0], block)
        true_val = self._reconstruct_expr(expr.children[1], block)
        false_val = self._reconstruct_expr(expr.children[2], block)

        if cond is None or true_val is None or false_val is None:
            return None

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.select(cond, true_val, false_val)

        return result

    def _reconstruct_cast(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct cast operations."""
        if len(expr.children) < 1:
            return None

        operand = self._reconstruct_expr(expr.children[0], block)
        if operand is None:
            return None

        target_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            if expr.op == 'Index_cast':
                result = self.builder.index_cast(operand, target_type)
            elif expr.op in ('SIToFP', 'Sitofp'):
                result = self.builder.sitofp(operand, target_type)
            elif expr.op in ('UIToFP', 'Uitofp'):
                result = self.builder.uitofp(operand, target_type)
            elif expr.op in ('FPToSI', 'Fptosi'):
                result = self.builder.fptosi(operand, target_type)
            elif expr.op in ('FPToUI', 'Fptoui'):
                result = self.builder.fptoui(operand, target_type)
            elif expr.op in ('ExtSI', 'Extsi'):
                result = self.builder.extsi(operand, target_type)
            elif expr.op in ('ExtUI', 'Extui'):
                result = self.builder.extui(operand, target_type)
            elif expr.op in ('TruncI', 'Trunci'):
                result = self.builder.trunci(operand, target_type)
            elif expr.op == 'Bitcast':
                result = self.builder.bitcast(operand, target_type)
            else:
                return None
        return result

    # ===== Control Flow Operations =====

    def _pre_materialize_branch_dependencies(self, branch_expr: ExpressionNode, block: MBlock):
        """
        Pre-materialize all pure operations that a branch will use.
        This ensures operations are created in the outer block context before entering nested blocks.

        Strategy: Recursively traverse the branch expression tree and materialize all non-control-flow operations.
        """
        if not branch_expr:
            return

        target_block = self.current_block_context if self.current_block_context else block
        cache_key = (branch_expr.node_id, id(target_block))
        if cache_key in self.expr_to_ssa:
            # Already materialized in this block context
            return

        # Operations that should NOT be pre-materialized (they define control flow structure)
        control_flow_ops = {'If_', 'While_', 'For', 'For_with_carry', 'Affine_for', 'Affine_for_with_carry',
                           'Block', 'Vec', 'Yield_', 'Affine_yield', 'Return_', 'Condition',
                           'Store', 'Load', 'Alloc', 'Alloca', 'Get_global'}  # Memory ops with side effects

        if branch_expr.op in control_flow_ops:
            # For control flow and side-effect operations, recursively pre-materialize their dependencies
            for child in branch_expr.children:
                self._pre_materialize_branch_dependencies(child, block)
        else:
            # For pure operations (arithmetic, casts, comparisons), materialize in the current block
            self._reconstruct_expr(branch_expr, block)

    def _branch_yields_values(self, branch_expr: ExpressionNode) -> bool:
        """Check if a branch yields non-empty values."""
        # Navigate to the yield node in the branch
        # Branch structure: Block -> Vec -> [..., Yield]
        if branch_expr.op == 'Block' and branch_expr.children:
            vec_expr = branch_expr.children[0]
            if vec_expr.op == 'Vec' and vec_expr.children:
                # Find the yield (usually last child)
                for child in vec_expr.children:
                    if child.op in ['Yield_', 'Affine_yield']:
                        # Check if yield has values
                        if child.children and len(child.children) > 0:
                            yield_vec = child.children[0]
                            if yield_vec.op == 'Vec':
                                return len(yield_vec.children) > 0
                        return False
        # Fallback: if branch is directly a yield
        if branch_expr.op in ['Yield_', 'Affine_yield']:
            if branch_expr.children and len(branch_expr.children) > 0:
                yield_vec = branch_expr.children[0]
                if yield_vec.op == 'Vec':
                    return len(yield_vec.children) > 0
        return False

    def _reconstruct_if(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct if-then-else operations."""
        if len(expr.children) < 3:
            return None
        

        # Reconstruct condition
        cond = self._reconstruct_expr(expr.children[0], block)
        if cond is None:
            return None

        # Determine return types based on yield values in branches
        # If both branches have empty yields, no return value
        then_expr = expr.children[1]
        else_expr = expr.children[2]

        # IMPORTANT: Pre-materialize all operations that the branches will use
        # This ensures they are created in the current block context, not inside the if/else blocks
        old_prematerialize_context = self.current_block_context
        self.current_block_context = block
        self._pre_materialize_branch_dependencies(then_expr, block)
        self._pre_materialize_branch_dependencies(else_expr, block)
        self.current_block_context = old_prematerialize_context

        # Check if branches yield values
        has_then_yield_values = self._branch_yields_values(then_expr)
        has_else_yield_values = self._branch_yields_values(else_expr)

        if not has_then_yield_values and not has_else_yield_values:
            # No return values
            return_types = []
        else:
            # Has return values - use dtype from expression
            result_t = expr.dtype
            if not result_t or isinstance(result_t, VoidType):
                return_types = []
            else:
                result_type = megg_to_mlir_type(result_t)
                if result_type is None:
                    return_types = []
                else:
                    return_types = [result_type]

        # Check if else branch is actually empty (just a yield with no values)
        # (else_expr already defined above)
        has_non_empty_else = not (
            else_expr.op == 'Yield_' and
            (not else_expr.children or
             (len(else_expr.children) == 1 and
              else_expr.children[0].op == 'Vec' and
              len(else_expr.children[0].children) == 0))
        )

        # Create if operation using builder
        with self.builder.set_insertion_point_to_end(block):
            if_op_m = self.builder.scf_if(cond, return_types, has_else=has_non_empty_else)

        # Save old context
        old_context = self.current_block_context

        # Reconstruct then branch with proper context
        then_expr = expr.children[1]
        then_block = self.builder.get_if_then_block(if_op_m)
        self.current_block_context = then_block
        self._reconstruct_branch(then_expr, then_block)

        # Reconstruct else branch only if non-empty
        if has_non_empty_else:
            else_block = self.builder.get_if_else_block(if_op_m)
            self.current_block_context = else_block
            self._reconstruct_branch(else_expr, else_block)

        # Restore context
        self.current_block_context = old_context

        # Cache the if operation result before returning
        # This prevents the same if from being reconstructed multiple times
        if_result_block = old_context if old_context else block
        if_result_cache_key = (expr.node_id, id(if_result_block))

        # Return the result value(s)
        if len(return_types) == 0:
            result = None
        else:
            # Get the first result from if operation
            result = self.builder.get_operation_result(if_op_m, 0)

        # Cache the result (if not already cached)
        if if_result_cache_key not in self.expr_to_ssa:
            if result is not None:
                self.expr_to_ssa[if_result_cache_key] = result
            else:
                self.expr_to_ssa[if_result_cache_key] = "PROCESSED"

        return result

    def _reconstruct_condition(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct scf.condition operation."""
        # print(
        #     f"Reconstructing Condition {expr.node_id} with children {[child.node_id for child in expr.children]}")

        if len(expr.children) < 2:
            return None

        # 第一个子节点是条件值
        cond_val = self._reconstruct_expr(expr.children[0], block)
        # print(f"Condition value: {cond_val}")
        if cond_val is None:
            return None

        # 确保条件是 i1 类型 - 通过字符串表示来比较类型
        cond_type_str = str(cond_val.get_type())

        # Only convert if it's NOT already i1
        if cond_type_str != 'i1':
            cond_type = cond_val.get_type()
            with self.builder.set_insertion_point_to_end(block):
                zero = self.builder.constant(0, cond_type)
                cond_val = self.builder.cmp("ne", cond_val, zero)

        # print(f"Condition children: {[child.node_id for child in expr.children]}")
        # 第二个子节点应该是一个 Vec，包含要传递的所有值
        if len(expr.children) >= 2:
            vec_expr = expr.children[1]
            yield_values = self._collect_values_from_expr(vec_expr, block)
        else:
            yield_values = []
        # print(f"Yield values: {yield_values}")
        # 创建 scf.condition 操作
        with self.builder.set_insertion_point_to_end(block):
            self.builder.scf_condition(cond_val, yield_values)

        return None

    def _reconstruct_while(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct scf.while using first child as init_args, second as condition, third as body."""
        if len(expr.children) < 3:
            return None

        # === 1) 重构 init_args ===
        init_vals_expr = expr.children[0]
        init_vals = []
        if init_vals_expr.op == "Vec":
            # 如果init_args是Vec，重构所有子表达式
            for child in init_vals_expr.children:
                val = self._reconstruct_expr(child, block)
                if val is not None:
                    init_vals.append(val)
        else:
            # 如果init_args是单个表达式
            val = self._reconstruct_expr(init_vals_expr, block)
            if val is not None:
                init_vals.append(val)
        if not init_vals:
            return None

        # === 2) 构造 result_types ===
        result_types = []
        for i, v in enumerate(init_vals):
            # Check if it's MValue first
            if isinstance(v, MValue):
                result_types.append(v.get_type())
            else:
                # Check if it's MOperation by type name or has results
                type_name = type(v).__name__
                if type_name == 'MOperation' or hasattr(v, 'results'):
                    # It's an MOperation, extract first result
                    try:
                        op_results = v.results
                        if op_results and len(op_results) > 0:
                            init_vals[i] = op_results[0]
                            result_types.append(op_results[0].get_type())
                        else:
                            raise RuntimeError(
                                f"MOperation in init_vals has no results: {v}")
                    except AttributeError as e:
                        # The MOperation object has MLIRValue instead of MLIROperation
                        # This happens when v is actually wrapping a value, not an operation
                        # Try to use it directly as a value
                        if hasattr(v, '_op') and type(v._op).__name__ == 'MLIRValue':
                            # It's actually a value wrapped in MOperation, convert to MValue
                            init_vals[i] = MValue(v._op)
                            result_types.append(MValue(v._op).get_type())
                        else:
                            raise RuntimeError(
                                f"Expected MValue or MOperation in init_vals, got {type(v)} with error: {e}")
                else:
                    raise RuntimeError(
                        f"Expected MValue or MOperation in init_vals, got {type(v)}")

        # === 3) 创建while操作 ===
        with self.builder.set_insertion_point_to_end(block):
            while_op_m = self.builder.scf_while(result_types, init_vals)

        # === 4) 获取 before/after blocks ===
        before_block = self.builder.get_while_before_block(while_op_m)
        after_block = self.builder.get_while_after_block(while_op_m)

        # 注册循环上下文
        loop_ids = expr.metadata.get('while_info', [])
        if len(loop_ids) < 2:
            raise ValueError(
                f"While node {expr.node_id} missing 'while_info' metadata")
        before_block_id = loop_ids[0]
        after_block_id = loop_ids[1]
        self.loop_contexts[f"loopblock_{before_block_id}"] = before_block
        self.loop_contexts[f"loopblock_{after_block_id}"] = after_block

        # Save old context
        old_context = self.current_block_context

        # === 5) 构建 before block（条件） ===
        cond_expr = expr.children[1]
        self.current_block_context = before_block
        self._reconstruct_branch(cond_expr, before_block)

        # === 6) 构建 after block（循环体） ===
        body_expr = expr.children[2]
        self.current_block_context = after_block
        self._reconstruct_branch(body_expr, after_block)

        # Restore context
        self.current_block_context = old_context

        # === 7) 返回 while 的结果 ===
        # Return the first result by default (will be adjusted by caller if needed)
        result = self.builder.get_operation_result(while_op_m, 0)
        return result

    def _reconstruct_for(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct for loop without iter_args."""
        if len(expr.children) < 4:
            return None

        # Extract loop bounds
        lower = self._reconstruct_expr(expr.children[0], block)
        upper = self._reconstruct_expr(expr.children[1], block)
        step = self._reconstruct_expr(expr.children[2], block)

        if not (lower and upper and step):
            return None

        # FIXED: For_ now has 5 children: [lower, upper, step, loop_index_term, body]
        # child[3] is a Loop_index Term containing the loop_id
        loop_id_expr = expr.children[3] if len(expr.children) >= 5 else None
        body_expr_index = 4 if loop_id_expr else 3
        body_expr = expr.children[body_expr_index]

        loop_id = None
        if loop_id_expr is not None and hasattr(loop_id_expr, 'metadata'):
            # child[3] should be a Loop_index node with arg_info containing the loop_id
            loop_id = loop_id_expr.metadata.get('arg_info', None)

        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        # Create for operation
        with self.builder.set_insertion_point_to_end(block):
            for_op_m = self.builder.scf_for(lower, upper, step, [])

        # Register loop context - use builder method to get body block
        body_block = self.builder.get_for_body_block(for_op_m)

        # CRITICAL FIX: Only register the current loop's own ID
        # Each loop should only register its own ID to its body block, not IDs from nested loops
        # The local_loop_ids is used for the stack so nested loops can inherit outer loop IDs
        local_loop_ids: Set[str] = set()
        if loop_id is not None:
            local_loop_ids.add(str(loop_id))
            # Register current loop's ID to current body block
            self.loop_contexts[f"loop_{loop_id}"] = body_block

        if loop_key:
            self.loop_contexts[loop_key] = body_block

        # Update context so pure ops materialize inside the loop body
        old_context = self.current_block_context
        self.current_block_context = body_block

        # Track current loop nesting for fallback loop index resolution
        self.loop_block_stack.append((body_block, local_loop_ids))
        try:
            self._reconstruct_branch(
                body_expr,
                body_block,
                allow_inferred_yield_values=False,
            )
        finally:
            self.loop_block_stack.pop()
            self.current_block_context = old_context

        # scf.for without iter_args produces no SSA results
        return None

    def _reconstruct_for_with_carry(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        # """Reconstruct for loop with iter_args."""
        # print(f"Reconstructing For_with_carry {expr.node_id} with children {[child.node_id for child in expr.children]}")

        if len(expr.children) < 6:
            return None

        # Extract loop bounds
        lower = self._reconstruct_expr(expr.children[0], block)
        upper = self._reconstruct_expr(expr.children[1], block)
        step = self._reconstruct_expr(expr.children[2], block)

        if not (lower and upper and step):
            return None

        # Extract loop ID, which is saved in the metadata of children[3]
        loop_id = expr.children[3].metadata.get('arg_info', None)
        # print(f"  Loop ID info: {loop_id}")
        loop_key = f"loop_{loop_id}"

        # Extract iter_args (children[4] should be Vec[Term])
        iter_args = []
        init_vals_expr = expr.children[4]
        for child in init_vals_expr.children:
            val = self._reconstruct_expr(child, block)
            if val:
                iter_args.append(val)

        # Create for operation with iter_args
        with self.builder.set_insertion_point_to_end(block):
            for_op_m = self.builder.scf_for(lower, upper, step, iter_args)

        # Register loop context - use builder method to get body block
        body_block = self.builder.get_for_body_block(for_op_m)
        body_expr = expr.children[5]

        # CRITICAL FIX: Only register the current loop's own ID
        # Each loop should only register its own ID to its body block, not IDs from nested loops
        # The local_loop_ids is used for the stack so nested loops can inherit outer loop IDs
        local_loop_ids: Set[str] = set()
        if loop_id is not None:
            local_loop_ids.add(str(loop_id))
            # Register current loop's ID to current body block
            self.loop_contexts[f"loop_{loop_id}"] = body_block

        self.loop_contexts[loop_key] = body_block

        # Update context: loop body is a new context for pure operations
        old_context = self.current_block_context
        self.current_block_context = body_block
        self.loop_block_stack.append((body_block, local_loop_ids))

        try:
            # Reconstruct body
            self._reconstruct_branch(
                body_expr,
                body_block,
                allow_inferred_yield_values=False,
            )

            # CRITICAL FIX: For loops with iter_args, ensure yield returns correct values
            # Check if the body has a proper terminator with the right number of operands
            ops = body_block.get_operations()
            if ops:
                last_op = ops[-1]
                if self._is_terminator(last_op):
                    # Check if yield has the right number of operands
                    yield_operands = last_op.operands if hasattr(last_op, 'operands') else []
                    expected_count = len(iter_args)

                    if len(yield_operands) != expected_count:
                        # Yield has wrong number of operands - need to fix it
                        logger = __import__('logging').getLogger(__name__)
                        logger.warning(
                            f"Loop with {expected_count} iter_args has yield with "
                            f"{len(yield_operands)} operands. Attempting to infer correct yield values."
                        )

                        # Remove the incorrect yield
                        last_op.erase()

                        # Strategy: Use the iter_args from the loop's block arguments
                        # This ensures type correctness (iter_args have the correct types)
                        block_args = body_block.arguments if hasattr(body_block, 'arguments') else []
                        # First arg is induction variable, rest are iter_args
                        # So for N iter_args, we need block_args[1] to block_args[N]
                        if len(block_args) > expected_count:
                            yield_values = list(block_args[1:expected_count+1])
                        else:
                            yield_values = []

                        # Fallback: If we still don't have the right count, try to infer from results
                        if len(yield_values) != expected_count:
                            yield_values = []
                            for op in reversed(ops[:-1]):  # Exclude the yield we just removed
                                op_results = op.results if hasattr(op, 'results') else []
                                if op_results and len(op_results) > 0:
                                    # Check if the result type matches the expected iter_arg type
                                    # This is a heuristic - prefer results matching iter_arg types
                                    yield_values = [op_results[0]]
                                    break

                        # Create corrected yield
                        with self.builder.set_insertion_point_to_end(body_block):
                            self.builder.scf_yield(yield_values)
        finally:
            # Restore context
            self.loop_block_stack.pop()
            self.current_block_context = old_context

        return self.builder.get_operation_result(for_op_m, 0)
    
    def _reconstruct_affine_for(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        Reconstruct affine.for without iter_args.
        Signature: affine_for(lower_operands, upper_operands, step, index_var,
                             lower_map, upper_map, body_stmts, type)
        """
        if len(expr.children) < 7:
            return None

        # Extract parameters
        step_value = self._extract_int_value(expr.children[2])

        loop_metadata = expr.children[3].metadata if expr.children[3].metadata else {}
        loop_id = loop_metadata.get('arg_info', None)
        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        # Create affine.for operation (using constant bounds for now)
        lower_bound = 0
        upper_bound = 10  # TODO: extract from affine maps

        with self.builder.set_insertion_point_to_end(block):
            for_op = self.builder.affine_for(lower_bound, upper_bound, step_value)

        if loop_key:
            self.loop_contexts[loop_key] = for_op.body

        # Reconstruct body: body_stmts is Vec[Term] at children[6]
        body_stmts_expr = expr.children[6]
        if body_stmts_expr.op == 'Vec':
            for stmt in body_stmts_expr.children:
                self._reconstruct_expr(stmt, for_op.body)

        results = list(for_op.results_)
        return results[0] if results else None

    def _reconstruct_affine_for_with_carry(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        Reconstruct affine.for with iter_args.
        Signature: affine_for_with_carry(lower_operands, upper_operands, step, index_var,
                                        lower_map, upper_map, init_values, body_stmts, type)
        """
        if len(expr.children) < 8:
            return None

        # Extract parameters
        step_value = self._extract_int_value(expr.children[2])

        loop_metadata = expr.children[3].metadata if expr.children[3].metadata else {}
        loop_id = loop_metadata.get('arg_info', None)
        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        # Collect init values from Vec at children[6]
        init_vals = []
        init_vec_expr = expr.children[6]
        if init_vec_expr.op == 'Vec':
            for child in init_vec_expr.children:
                val = self._reconstruct_expr(child, block)
                if val:
                    init_vals.append(val)

        # Create affine.for operation (using constant bounds for now)
        lower_bound = 0
        upper_bound = 10  # TODO: extract from affine maps

        with self.builder.set_insertion_point_to_end(block):
            for_op = self.builder.affine_for(lower_bound, upper_bound, step_value)

        if loop_key:
            self.loop_contexts[loop_key] = for_op.body

        # Reconstruct body: body_stmts is Vec[Term] at children[7]
        body_stmts_expr = expr.children[7]
        if body_stmts_expr.op == 'Vec':
            for stmt in body_stmts_expr.children:
                self._reconstruct_expr(stmt, for_op.body)

        results = list(for_op.results_)
        return results[0] if results else None
    
    def _reconstruct_vec(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        Reconstruct a Vec node: this represents an ordered sequence of sub-terms.
        We reconstruct each child in order (so side-effects / ordering preserved).
        Return the last non-None SSA value (so Vec can be used where a single value is expected).
        """
        last_result = None
        for child in expr.children:
            res = self._reconstruct_expr(child, block)
            if res is not None:
                last_result = res
        return last_result

    def _collect_values_from_expr(self, expr: ExpressionNode, block: MBlock) -> List[MValue]:
        """
        Collect values for yield/return contexts.
        If expr is Vec -> reconstruct each element and return list of SSA values in order.
        Else -> reconstruct expr and return [value] (if not None), or [].
        """
        values: List[MValue] = []
        # Defensive: some ExpressionNode implementations might have op as None/empty
        op_name = getattr(expr, "op", None)
        if op_name == 'Vec':
            for child in expr.children:
                val = self._reconstruct_expr(child, block)
                # print(f"Vec child {child.node_id} -> {val}")
                if val is not None:
                    # Convert MOperation to MValue if needed
                    if isinstance(val, MValue):
                        values.append(val)
                    elif hasattr(val, 'results'):
                        # Has results property (MOperation)
                        op_results = val.results  # results is a @property, not a method
                        if op_results and len(op_results) > 0:
                            values.append(op_results[0])
                    # Else: ignore non-value objects
        else:
            val = self._reconstruct_expr(expr, block)
            if val is not None:
                # Convert MOperation to MValue if needed
                if isinstance(val, MValue):
                    values.append(val)
                elif hasattr(val, 'results'):
                    # Has results property (MOperation)
                    op_results = val.results  # results is a @property, not a method
                    if op_results and len(op_results) > 0:
                        values.append(op_results[0])
                # Else: ignore non-value objects
        return values

    def _ensure_block_terminated(
        self,
        block: MBlock,
        expected_results: List[MType],
        fallback_values: List[MValue],
    ) -> None:
        """Ensure a block ends with a terminator.

        If greedy extraction fails to emit an explicit func.return (e.g., when a
        custom instruction replaces the entire function body), reuse the latest
        SSA values as implicit return operands to keep MLIR well-formed.
        """
        ops = block.get_operations()
        if ops and self._is_terminator(ops[-1]):
            return

        with self.builder.set_insertion_point_to_end(block):
            if expected_results:
                if not fallback_values or len(fallback_values) != len(expected_results):
                    raise ValueError(
                        "Function expects return values but block has no terminator and no fallback operands"
                    )
                casted_vals = [
                    self._cast_value_to_type(val, expected_results[idx])
                    for idx, val in enumerate(fallback_values)
                ]
                self.builder.func_return(casted_vals)
            else:
                self.builder.func_return([])

    def _cast_value_to_type(self, value: MValue, target_type: MType) -> MValue:
        """Ensure a value matches the expected MLIR type."""
        if value is None or target_type is None:
            return value

        src_type_str = str(value.get_type())
        tgt_type_str = str(target_type)

        if src_type_str == tgt_type_str:
            return value

        def parse_int_width(type_str: str) -> Optional[int]:
            if type_str.startswith('i') and type_str[1:].isdigit():
                return int(type_str[1:])
            return None

        src_width = parse_int_width(src_type_str)
        tgt_width = parse_int_width(tgt_type_str)

        if src_width is not None and tgt_width is not None:
            if src_width > tgt_width:
                return self.builder.trunci(value, target_type)
            elif src_width < tgt_width:
                return self.builder.extsi(value, target_type)
            return value

        if tgt_type_str == 'index':
            return self.builder.index_cast(value, target_type)

        return value

    def _collect_loop_index_ids(self, expr: ExpressionNode, stop_ops: Optional[set[str]] = None, debug: bool = False) -> set[str]:
        """Collect loop index identifiers from an expression, stopping at nested loops."""
        if stop_ops is None:
            stop_ops = {
                'For', 'For_', 'For_with_carry',
                'Affine_for', 'Affine_for_with_carry',
                'While_', 'While',
            }

        loop_ids: set[str] = set()
        stack = [expr]
        while stack:
            node = stack.pop()
            op_name = getattr(node, 'op', None)
            if op_name is None:
                continue
            canonical_op = self._canonical_op(op_name)
            if canonical_op in stop_ops and node is not expr:
                # Do not traverse into nested loops
                if debug:
                    print(f"    [COLLECT_STOP] Stopped at nested {canonical_op}")
                continue
            if canonical_op == 'Loop_index':
                info = node.metadata.get('arg_info') if node.metadata else None
                if info is not None:
                    if debug:
                        print(f"    [COLLECT_FOUND] Loop_index with arg_info={info}")
                    loop_ids.add(str(info))
            stack.extend(node.children or [])
        return loop_ids

    def _reconstruct_block(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        Reconstruct a Block: currently each Block has exactly one child (a Vec).
        So we just reconstruct that child. The Vec will handle ordered expansion.
        """
        if not expr.children:
            return None
        # By convention only one child, usually a Vec
        return self._reconstruct_expr(expr.children[0], block)


    def _reconstruct_branch(
        self,
        branch_expr: ExpressionNode,
        target_block: MBlock,
        *,
        terminator_dialect: str = 'scf',
        allow_inferred_yield_values: bool = True,
    ):
        """Reconstruct a branch (then/else/body)."""
        # Remove any auto-generated empty yield terminators before reconstructing
        # MLIR's scf.if automatically creates empty yields which we need to replace
        initial_ops = target_block.get_operations()
        if initial_ops and self._is_terminator(initial_ops[-1]):
            initial_ops[-1].erase()

        # Reconstruct the branch expression
        self._reconstruct_expr(branch_expr, target_block)

        # Ensure proper terminator
        ops = target_block.get_operations()
        if ops:
            last_op = ops[-1]
            is_term = self._is_terminator(last_op)
        else:
            is_term = False

        if not ops or not is_term:
            if allow_inferred_yield_values:
                yield_values = []

                if ops and len(ops) > 0:
                    for op in reversed(ops):
                        op_results = op.results if hasattr(op, 'results') else []
                        if op_results and len(op_results) > 0:
                            yield_values = [op_results[0]]
                            break
            else:
                yield_values = []

            # Add yield with inferred values
            with self.builder.set_insertion_point_to_end(target_block):
                if terminator_dialect == 'affine':
                    self.builder.affine_yield(yield_values)
                else:
                    self.builder.scf_yield(yield_values)

    def _reconstruct_yield(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        A Yield always has a single Vec child, whose elements are the yielded values.
        """
        # Determine the yield values first
        if not expr.children:
            yield_values = []
        else:
            vec_expr = expr.children[0]
            yield_values = self._collect_values_from_expr(vec_expr, block)

        # Check if block already has a terminator (MLIR may auto-create yields for scf.if blocks)
        ops = block.get_operations()
        if ops and self._is_terminator(ops[-1]):
            # If terminator exists and we need to yield values, we can't skip
            if len(yield_values) > 0:
                # Fall through to create the yield anyway
                pass
            else:
                # Empty yield and block already has a terminator - skip
                return None

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.scf_yield(yield_values)
        return None

    def _reconstruct_affine_yield(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        if not expr.children:
            with self.builder.set_insertion_point_to_end(block):
                self.builder.affine_yield([])
            return None

        vec_expr = expr.children[0]
        values = self._collect_values_from_expr(vec_expr, block)

        with self.builder.set_insertion_point_to_end(block):
            self.builder.affine_yield(values)
        return None

    def _reconstruct_custom_instr(self, expr: ExpressionNode, block:MBlock) -> Optional[MValue]:
        """
        Reconstruct custom instruction as a call (or inline asm) with typed operands.
        """
        instr_name = expr.metadata.get('instr_name') if expr.metadata else None
        # call_symbol = expr.metadata.get('call_symbol') if expr.metadata else instr_name
        if not instr_name:
            raise ValueError(f"Custom instruction node {expr.node_id} missing 'instr_name' metadata")

        # Collect operand values
        operand_values: List[MValue] = []
        if expr.children:
            # 两种情况：
            # 1. children[0] 是 Vec 表达式（简单计算模式，通过 egglog rewrite）
            # 2. children 是直接的 operand 表达式列表（复杂控制流，通过 MeggEGraph.add_custom_instr_node）
            if len(expr.children) == 1 and expr.children[0].op == 'Vec':
                # 情况1：从 Vec 中收集
                operands_expr = expr.children[0]
                operand_values = self._collect_values_from_expr(operands_expr, block)
            else:
                # 情况2：直接从 children 列表收集
                for child_expr in expr.children:
                    val = self._reconstruct_expr(child_expr, block)
                    if val is not None:
                        operand_values.append(val)

        # Determine result type
        result_type = megg_to_mlir_type(expr.dtype) if expr.dtype is not None else None
        if result_type is not None and str(result_type) == 'index':
            # Use i32 for RV32 architecture (Chipyard compatibility)
            result_type = self.builder.i32()

        instr_info = self.instr_properties.get(instr_name, {}) if self.instr_properties else {}

        # Check if we have encoding information for RISC-V .insn format
        encoding = instr_info.get('encoding')

        # Build assembly string and adjust operands if needed
        num_inputs = len(operand_values)
        if encoding:
            # Generate RISC-V .insn format
            opcode = encoding.get('opcode', '0x0')
            funct3 = encoding.get('funct3', '0x7')  # Default 0x7 (0b111) for R-type
            funct7 = encoding.get('funct7', '0x0')

            # R-type instructions require at least 2 source operands
            # If pattern only uses 1 argument (e.g., horner3 uses arg0 but not arg1),
            # we need to pad with a dummy operand
            if num_inputs < 2:
                # Duplicate the first operand as dummy for the second source register
                if operand_values:
                    operand_values = list(operand_values) + [operand_values[0]]
                    num_inputs = 2

            # Build .insn r format: .insn r opcode, funct3, funct7, $0, $1, $2, ...
            # $0 is the output register (if result_type is not None)
            # $1, $2, ... are input registers
            operand_placeholders = []
            operand_idx = 0
            if result_type is not None:
                operand_placeholders.append(f"${operand_idx}")
                operand_idx += 1
            for _ in range(num_inputs):
                operand_placeholders.append(f"${operand_idx}")
                operand_idx += 1

            asm_string = f".insn r {opcode}, {funct3}, {funct7}, " + ", ".join(operand_placeholders)

        else:
            # No encoding - use instruction name as-is (original behavior)
            asm_string = instr_name

        # Build constraints: "=r" for output, "r" for each input
        # Note: num_inputs may have been updated above if we added dummy operands
        constraint_items: List[str] = []
        if result_type is not None:
            constraint_items.append("=r")
        constraint_items.extend(["r"] * num_inputs)

        # Handle clobbers: add memory clobber for encoded custom instructions
        clobbers = instr_info.get('clobbers') or []
        if encoding and 'memory' not in clobbers:
            clobbers = list(clobbers) + ['memory']

        if clobbers:
            constraint_items.extend([f"~{{{clobber}}}" for clobber in clobbers])
        constraints = ",".join(constraint_items)

        # Handle side effects: force for encoded custom instructions
        has_side_effects = instr_info.get('has_side_effects')
        if has_side_effects is None:
            has_side_effects = (result_type is None)
        if encoding:
            # Force side effects for custom instructions to prevent LLVM from optimizing them away
            has_side_effects = True

        # Create inline asm using IRBuilder
        with self.builder.set_insertion_point_to_end(block):
            converted_operands: List[MValue] = []
            for val in operand_values:
                converted_val = val
                type_str = str(converted_val.type)

                # Handle different pointer types for custom instructions
                if type_str == '!llvm.ptr':
                    # LLVM opaque pointer - directly convert to i32 for RV32
                    converted_val = self.builder.ptrtoint(converted_val, self.builder.i32())
                elif type_str.startswith('memref<'):
                    # Legacy memref - extract pointer and convert
                    converted_val = self.builder.extract_aligned_pointer_as_index(
                        converted_val)
                    type_str = str(converted_val.type)
                    if type_str == 'index':
                        # Use i32 for RV32 architecture (Chipyard compatibility)
                        converted_val = self.builder.index_cast(
                            converted_val, self.builder.i32())
                elif type_str == 'index':
                    # Direct index type - convert to i32
                    converted_val = self.builder.index_cast(
                        converted_val, self.builder.i32())

                converted_operands.append(converted_val)
            operand_values = converted_operands
            result = self.builder.inline_asm(
                asm_string=asm_string,
                constraints=constraints,
                operands=operand_values,
                result_type=result_type,
                has_side_effects=has_side_effects
            )

        return result if result_type is not None else None

    def _reconstruct_return(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """
        A Return always has a single Vec child, whose elements are the return values.
        """
        if not expr.children:
            with self.builder.set_insertion_point_to_end(block):
                self.builder.func_return([])
            return None

        vec_expr = expr.children[0]
        values = self._collect_values_from_expr(vec_expr, block)
        # print(f"Return values: {values}")

        # Get expected return types from function signature
        func_type = self.original_func.get_function_type
        expected_types = func_type.get_results()

        # If a value comes from a multi-result operation and we need a specific type,
        # extract the correct result index
        adjusted_values = []
        for i, val in enumerate(values):
            if i < len(expected_types):
                expected_type = expected_types[i]
                # Check if this value's operation has multiple results
                if hasattr(val, '_value') and hasattr(val._value, 'get_defining_op'):
                    defining_op = val._value.get_defining_op()
                    if defining_op and hasattr(defining_op, 'get_num_results'):
                        num_results = defining_op.get_num_results()
                        if num_results > 1:
                            # Multi-result operation - find matching result
                            # Find which result matches the expected type
                            for res_idx in range(num_results):
                                result_val = self.builder.get_operation_result(MOperation(defining_op), res_idx)
                                if str(result_val.get_type()) == str(expected_type):
                                    adjusted_values.append(result_val)
                                    break
                            else:
                                # No match found, use original
                                adjusted_values.append(val)
                        else:
                            adjusted_values.append(val)
                    else:
                        adjusted_values.append(val)
                else:
                    adjusted_values.append(val)
            else:
                adjusted_values.append(val)

        converted_values: List[MValue] = []
        for i, val in enumerate(adjusted_values):
            if i < len(expected_types):
                converted_values.append(
                    self._cast_value_to_type(val, expected_types[i])
                )
            else:
                converted_values.append(val)

        with self.builder.set_insertion_point_to_end(block):
            self.builder.func_return(converted_values)
        return None

    # ===== MemRef Operations =====

    def _reconstruct_memref_alloc(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.alloc."""
        memref_type = megg_to_mlir_type(expr.dtype)  # Returns MType
        # print(f"Alloc memref type: {memref_type}")

        with self.builder.set_insertion_point_to_end(block):
            # Pass MType directly - the builder will unwrap it
            alloc_op = self.builder.alloc(memref_type)

        return alloc_op

    def _reconstruct_memref_alloca(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.alloca (stack allocation)."""
        memref_type = megg_to_mlir_type(expr.dtype)  # Returns MType
        # print(f"Alloca memref type: {memref_type}")

        with self.builder.set_insertion_point_to_end(block):
            # Pass MType directly - the builder will unwrap it
            alloca_op = self.builder.alloca(memref_type)

        return alloca_op

    def _reconstruct_memref_get_global(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.get_global."""
        # Extract symbol name from children (first child contains the name)
        symbol_name = expr.metadata.get('global_name', None)
        print(f"Get_global symbol name from metadata: {symbol_name}")
        # Remove @ prefix if present
        if symbol_name and symbol_name.startswith('@'):
            symbol_name = symbol_name[1:]
        if expr.children:
            name_child = expr.children[0]
            # Extract from Boxed("@symbol") format
            if name_child.op.startswith('Boxed('):
                symbol_name = name_child.op[7:-2]  # Remove 'Boxed("' and '")'

        memref_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            # Use builder's get_global method
            get_global_result = self.builder.get_global(
                memref_type, symbol_name)

        return get_global_result

    def _reconstruct_memref_store(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.store."""
        if len(expr.children) < 3:
            return None

        value = self._reconstruct_expr(expr.children[0], block)
        memref_val = self._reconstruct_expr(expr.children[1], block)

        if not (value and memref_val):
            return None

        index_expr = expr.children[2]
        index_value = self._reconstruct_expr(index_expr, block)
        if not index_value:
            return None
        indices = [index_value]

        with self.builder.set_insertion_point_to_end(block):
            # Check if value type matches memref element type
            memref_type_str = str(memref_val.type)
            value_type_str = str(value.type)

            # Extract element type from memref (e.g., memref<16xi32> -> i32)
            if 'memref<' in memref_type_str:
                # Parse element type from memref type string
                # Format: memref<...xELEM_TYPE> or memref<?xELEM_TYPE>
                element_type_start = memref_type_str.rfind('x') + 1
                element_type_end = memref_type_str.find('>', element_type_start)
                if element_type_end == -1:
                    element_type_end = len(memref_type_str)
                element_type = memref_type_str[element_type_start:element_type_end]

                # If value type doesn't match element type, cast it
                if value_type_str != element_type:
                    # Handle index -> integer cast
                    if value_type_str == 'index' and element_type.startswith('i'):
                        # Get the element type as MType for casting
                        from megg.utils import IntType
                        # Parse bit width from element type (e.g., "i32" -> 32)
                        if element_type.startswith('i'):
                            bit_width = int(element_type[1:])
                            target_type = IntType(width=bit_width, signed=True)
                            mlir_target_type = megg_to_mlir_type(target_type)
                            value = self.builder.index_cast(value, mlir_target_type)

            # Use builder's store method to create memref.store operation
            self.builder.store(value, memref_val, indices)

        return None

    def _reconstruct_memref_load(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.load."""
        if len(expr.children) < 2:
            return None

        memref_val = self._reconstruct_expr(expr.children[0], block)
        if not memref_val:
            return None

        index_expr = expr.children[1]
        index_value = self._reconstruct_expr(index_expr, block)
        if not index_value:
            return None
        indices = [index_value]

        with self.builder.set_insertion_point_to_end(block):
            # Use builder's load method to create memref.load operation
            load_result = self.builder.load(memref_val, indices)

        return load_result

    def _extract_int_value(self, expr: ExpressionNode) -> int:
        """Extract integer value from expression node."""
        # For Int nodes, the value might be in metadata or encoded in node_id
        # This is a simplified extraction - adjust based on actual serialization
        if 'value' in expr.metadata:
            return int(expr.metadata['value'])

        # For LitTerm.int nodes, the value might be in the first child (i64 node)
        if expr.children:
            first_child = expr.children[0]
            # The op field might directly contain the value (e.g., op="10")
            try:
                return int(first_child.op)
            except ValueError:
                # Try to extract from i64(...) pattern in op
                if first_child.op.startswith('i64(') and first_child.op.endswith(')'):
                    try:
                        # Extract from "i64(...)"
                        value_str = first_child.op[4:-1]
                        return int(value_str)
                    except:
                        pass

        # Try to parse from node_id (egglog often encodes values in IDs)
        try:
            return int(expr.node_id.split('_')[-1])
        except:
            return 0

    def _extract_float_value(self, expr: ExpressionNode) -> float:
        """Extract float value from expression node."""
        if 'value' in expr.metadata:
            return float(expr.metadata['value'])

        # For LitTerm.float nodes, check first child
        if expr.children:
            first_child = expr.children[0]
            # The op field might directly contain the value
            try:
                return float(first_child.op)
            except ValueError:
                pass

        try:
            return float(expr.node_id.split('_')[-1])
        except:
            return 0.0

    def _extract_predicate(self, expr: ExpressionNode) -> str:
        """Extract predicate string from expression node."""
        if 'predicate' in expr.metadata:
            return expr.metadata['predicate']

        # Default predicates
        return "eq"

    def _is_terminator(self, operation) -> bool:
        """Check if operation is a terminator."""
        # Check the operation name/type directly, not the string representation
        # (str(operation) includes the entire operation body, which may contain terminators)
        try:
            # Get the operation type from MLIR
            if hasattr(operation, 'name'):
                op_type = str(operation.name).lower()
            elif hasattr(operation, 'operation'):
                # For wrapped operations
                op_type = str(operation.operation.name).lower()
            else:
                # Fallback: use first line of str(operation) which contains the operation name
                first_line = str(operation).split('\n')[0].strip().lower()
                op_type = first_line.split('(')[0].strip().strip('"')

            # Check if this is a terminator operation
            return ('yield' in op_type or 'return' in op_type or
                    'branch' in op_type or 'condition' in op_type)
        except Exception:
            # If we can't determine the type, be conservative and assume it's not a terminator
            return False
