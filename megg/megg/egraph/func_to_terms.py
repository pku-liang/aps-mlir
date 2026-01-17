"""
Direct transformation between MLIR FuncOp and Megg Terms.

This module provides direct conversion between func.FuncOp operations
and the e-graph Term representation, without intermediate data structures.
"""

from __future__ import annotations
from megg.egraph.datatype import (
    DataType, IntType, FloatType as MeggFloatType, IndexType,
    TupleType, VectorType, TensorType, MemRefType, VoidType
)
from megg.utils import MModule, MOperation, OperationType, MBlock, MValue, IRBuilder, MType
from megg.egraph.term import Term, LitTerm, id_of, eclass_ty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import egglog
from egglog import EGraph

# Global counter for generating unique session IDs
_session_counter = 0

def _next_session_id() -> str:
    """Generate a unique session ID for each FuncToTerms transformation."""
    global _session_counter
    _session_counter += 1
    return f"s{_session_counter}"


CONTROL_FLOW_OPS = {
    OperationType.SCF_FOR,
    OperationType.SCF_IF,
    OperationType.SCF_WHILE,
    OperationType.AFFINE_FOR,
    OperationType.AFFINE_IF,
}


@dataclass
class FuncToTerms:
    """
    Transform a FuncOp directly to Terms in an e-graph.

    Essential fields for round-trip translation:
    - ssa_to_term: def-use chain (SSA value → Term)
    - top_block: Term for the function body
    - ssa_to_id: numeric IDs for compiler integration
    - session_id: unique ID to avoid variable name collisions in egraph
    """

    func: MOperation
    egraph: EGraph
    next_id: int
    # Essential: Mapping from SSA values to numeric IDs (for compiler)
    ssa_to_id: Dict[MValue, int]
    # Essential: Mapping from SSA values to Terms (def-use chain)
    ssa_to_term: Dict[MValue, Term]
    # Essential: Top block
    top_block: Term
    # record loop info
    loop_to_term: Dict[int, Tuple[MOperation,Term]]
    loop_index: int = 0
    # Unique session ID to avoid variable name collisions when same egraph is reused
    session_id: str = field(default_factory=_next_session_id)

    @classmethod
    def transform(cls, func: MOperation, egraph: Optional[EGraph] = None) -> 'FuncToTerms':
        """
        Transform a FuncOp to Terms.

        Args:
            func: The function to transform
            egraph: E-graph to use (creates new one if not provided)

        Returns:
            FuncToTerms instance with the transformation
        """
        if egraph is None:
            egraph = EGraph(save_egglog_string=True)

        transformer = cls(
            func=func,
            egraph=egraph,
            next_id=0,
            ssa_to_id={},
            ssa_to_term={},
            loop_to_term={},
            top_block=None,
        )
        print(f"Starting transformation of function '{func.name}'")
        transformer._transform()
        return transformer

    def get_id_of_term(self, term: Term) -> Optional[int]:
        """Get the numeric ID of a Term if it's tracked."""
        for ssa_val, t in self.ssa_to_term.items():
            if t == term:
                return self.ssa_to_id.get(ssa_val, None)
        return None

    def get_tracked_terms(self) -> Dict[int, Term]:
        """Get all tracked terms with their numeric IDs."""
        terms = {}
        for ssa_val, term_id in self.ssa_to_id.items():
            if ssa_val in self.ssa_to_term:
                terms[term_id] = self.ssa_to_term[ssa_val]
        return terms

    def _transform(self):
        """Perform the transformation from FuncOp to Terms (only top block)."""
        regions = self.func.get_regions()
        if len(regions) == 0:
            return egglog.ruleset()  # Return empty ruleset

        blocks = regions[0].get_blocks()
        if len(blocks) == 0:
            return egglog.ruleset()  # Return empty ruleset

        block = blocks[0]

        # Process the function body as a single block
        self.top_block = self._block_to_term(block, is_function_block=True)

        # # Return type annotation ruleset
        # return self._create_type_annotation_ruleset()


    def _operation_to_term(self, op: MOperation, operand_terms: List[Term], loop_id: Optional[int] = None, num_iter_args: int = 0) -> Optional[Term]:
        """Convert an operation to a Term."""
        # Handle type based on number of results
        if len(op.results) == 0:
            ty_str = egglog.String("void")
        elif len(op.results) == 1:
            ty_str = mlir_type_to_egraph_ty_string(op.results[0].type)
        else:
            # Multiple results - create tuple type
            result_types = [r.type for r in op.results]
            ty_str = create_tuple_type_string(result_types)

        # print(f"Converting op: {op.name}, type: {ty_str}, operands: {len(operand_terms)}")

        # Get operation type from MOperation
        print(f"Operation: {op.name}, Type: {getattr(op, 'type', 'N/A')}")
        op_type = op.type if hasattr(op, 'type') else OperationType.UNKNOWN

        # Arithmetic operations - check by OperationType
        if op_type in [OperationType.ARITH_ADD, OperationType.ARITH_ADDF]:
            if len(operand_terms) == 2:
                return Term.add(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_SUB, OperationType.ARITH_SUBF]:
            if len(operand_terms) == 2:
                return Term.sub(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_MUL, OperationType.ARITH_MULF]:
            if len(operand_terms) == 2:
                return Term.mul(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_DIV, OperationType.ARITH_DIVF]:
            if len(operand_terms) == 2:
                return Term.div(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_REM, OperationType.ARITH_REMF]:
            if len(operand_terms) == 2:
                return Term.rem(operand_terms[0], operand_terms[1], ty_str)

        # Logical operations (bitwise)
        elif op_type == OperationType.ARITH_ANDI:
            if len(operand_terms) == 2:
                return Term.and_(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_ORI:
            if len(operand_terms) == 2:
                return Term.or_(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_XORI:
            if len(operand_terms) == 2:
                return Term.xor_(operand_terms[0], operand_terms[1], ty_str)

        # Shift operations
        elif op_type == OperationType.ARITH_SHLI:
            if len(operand_terms) == 2:
                return Term.shl(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_SHRSI:
            if len(operand_terms) == 2:
                return Term.shrsi(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_SHRUI:
            if len(operand_terms) == 2:
                return Term.shrui(operand_terms[0], operand_terms[1], ty_str)

        # Cast operations
        elif op_type == OperationType.ARITH_INDEX_CAST:
            if len(operand_terms) == 1:
                return Term.index_cast(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_SITOFP:
            if len(operand_terms) == 1:
                return Term.sitofp(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_UITOFP:
            if len(operand_terms) == 1:
                return Term.uitofp(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_FPTOSI:
            if len(operand_terms) == 1:
                return Term.fptosi(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_FPTOUI:
            if len(operand_terms) == 1:
                return Term.fptoui(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_EXTSI:
            if len(operand_terms) == 1:
                return Term.extsi(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_EXTUI:
            if len(operand_terms) == 1:
                return Term.extui(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_TRUNCI:
            if len(operand_terms) == 1:
                return Term.trunci(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_BITCAST:
            if len(operand_terms) == 1:
                return Term.bitcast(operand_terms[0], ty_str)

        # Comparison operations
        elif op_type == OperationType.ARITH_CMPI:
            if len(operand_terms) == 2:
                predicate = predicate_to_string(op.get_attr('predicate'))
                print(f"Comparison predicate: {predicate}")
                return Term.cmp(operand_terms[0], operand_terms[1], egglog.String(predicate), ty_str)
        elif op_type == OperationType.ARITH_CMPF:
            if len(operand_terms) == 2:
                predicate = predicate_to_string(
                    op.get_attr('predicate'), cmp_type='f')
                return Term.cmp(operand_terms[0], operand_terms[1], egglog.String(predicate), ty_str)

        # Select operation (ternary conditional: cond ? true_val : false_val)
        elif op_type == OperationType.ARITH_SELECT:
            if len(operand_terms) == 3:
                # select(condition, true_value, false_value)
                return Term.select(operand_terms[0], operand_terms[1], operand_terms[2], ty_str)

        elif op_type == OperationType.SCF_IF:
            # IfOp has condition, true_region, and false_region
            cond_term = operand_terms[0] if operand_terms else None

            # Get the then (true) branch
            then_term = None
            if len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                then_term = self._block_to_term(
                    op.regions[0].get_blocks()[0], loop_id, num_iter_args)

            # Get the else (false) branch
            else_term = None
            if len(op.regions) > 1 and len(op.regions[1].get_blocks()) > 0:
                else_term = self._block_to_term(
                    op.regions[1].get_blocks()[0], loop_id, num_iter_args)
            else:
                # If no else block, create a default empty yield term
                else_term = Term.yield_(
                    egglog.Vec[Term](), egglog.String("void"))

            if cond_term is not None and then_term is not None and else_term is not None:
                return Term.if_(cond_term, then_term, else_term, ty_str)
        elif op_type == OperationType.SCF_WHILE:
            # WhileOp has before_region (condition) and after_region (body)
            # Plus initial values passed as operands
            init_terms = operand_terms  # Initial values for the while loop

            # Get the condition block from before_region
            cond_term = None
            if len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                cond_term = self._block_to_term(op.regions[0].get_blocks()[0])

            # Get the body block from after_region
            body_term = None
            if len(op.regions) > 1 and len(op.regions[1].get_blocks()) > 0:
                body_term = self._block_to_term(op.regions[1].get_blocks()[0])

            if cond_term is not None and body_term is not None:
                # WhileOp doesn't have init values in the Term representation currently
                # Just pass condition and body
                return Term.while_(init_terms, cond_term, body_term, ty_str)
        elif op_type == OperationType.SCF_FOR:
            # ForOp has lower bound, upper bound, step, iter_args, and body with index
            if len(operand_terms) >= 3:
                lower = operand_terms[0]
                upper = operand_terms[1]
                step = operand_terms[2]

                # Get loop name from attribute
                loop_name = self.loop_index
                self.loop_index += 1
                # Generate a unique ID for this loop to track its index
                loop_id = id(op)
                loop_id_i64 = egglog.i64(loop_id)
                loop_index_term = Term.loop_index(
                    loop_id_i64, egglog.String("index"))

                # Check if there are iter_args (loop-carried values)
                iter_args = operand_terms[3:] if len(operand_terms) > 3 else []

                # Convert body with awareness of the loop index and iter_args

                # ForOp has body in regions[0]
                body_term = None
                if op.regions and len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                    body_term = self._block_to_term(
                        op.regions[0].get_blocks()[0], loop_id=loop_id, num_iter_args=len(
                            iter_args)
                    )
                elif hasattr(op, 'body') and len(op.body.get_blocks()) > 0:
                    body_term = self._block_to_term(
                        op.body.get_blocks()[0], loop_id=loop_id, num_iter_args=len(
                            iter_args)
                    )

                if body_term is not None:
                    if iter_args:
                        # Use for_with_carry for loops with loop-carried values
                        # for_with_carry expects index_var as Term
                        loop_term = Term.for_with_carry(
                            lower, upper, step,
                            loop_index_term,  # This is a Term
                            egglog.Vec[Term](*iter_args),
                            body_term, ty_str
                        )
                    else:
                        # Use regular for_ for simple loops
                        # FIXED: for_ now expects index_var as Term (like for_with_carry)
                        loop_term = Term.for_(
                            lower, upper, step,
                            loop_index_term,  # This is a Term (Loop_index node)
                            body_term, ty_str
                        )
                    self.loop_to_term[loop_name] = (op,loop_term)
                    print(f"Registered loop '{loop_name}' -> term")

                    return loop_term
        elif op_type == OperationType.FUNC_RETURN:
            # ReturnOp - for function returns (different from YieldOp)
            # Unified interface: always use Vec, even for 0 or 1 operands
            return Term.return_(egglog.Vec[Term](*operand_terms), ty_str)

        elif op_type == OperationType.SCF_YIELD:
            # YieldOp can yield multiple values (for loop-carried values)
            # Unified interface: always use Vec, even for 0 or 1 operands
            return Term.yield_(egglog.Vec[Term](*operand_terms), ty_str)

        elif op_type == OperationType.SCF_CONDITION:
            # ConditionOp is used in while loops to yield the condition and values
            # First operand is the condition, rest are the values to pass to the next iteration
            if len(operand_terms) >= 1:
                # For simplicity, treat it like a yield with the condition
                # In a proper implementation, we'd handle the condition separately
                # For now, just return a yield of the values (excluding the condition)
                return Term.yield_(egglog.Vec(*operand_terms[1:]), ty_str)
            return Term.yield_(egglog.Vec(), ty_str)

        elif op_type == OperationType.ARITH_CONSTANT:
            return self._constant_to_term(op)

        # MemRef operations
        elif op_type == OperationType.MEMREF_ALLOC:
            # memref.alloc() : memref<...>
            return Term.alloc(ty_str)
        elif op_type == OperationType.MEMREF_ALLOCA:
            # memref.alloca() : memref<...> (stack allocation)
            return Term.alloca(ty_str)
        elif op_type == OperationType.MEMREF_GET_GLOBAL:
            # memref.get_global @symbol_name : memref<...>
            # Extract the symbol name from attributes
            symbol_name = op.get_attr('name')
            print(f"memref.get_global symbol: {symbol_name}")
            return Term.get_global(egglog.String(symbol_name), ty_str)
        elif op_type == OperationType.MEMREF_STORE:
            # memref.store %value, %memref[%index] : memref<...>
            if len(operand_terms) < 3:
                raise ValueError(
                    "memref.store currently requires exactly one index term")
            value_term = operand_terms[0]  # Value to store
            memref_term = operand_terms[1]  # MemRef to store into
            index_terms = operand_terms[2:]
            if len(index_terms) != 1:
                raise ValueError(
                    f"memref.store expects one index term, got {len(index_terms)}")
            return Term.store(value_term, memref_term, index_terms[0], ty_str)
        elif op_type == OperationType.MEMREF_LOAD:
            # memref.load %memref[%index] : memref<...>
            if len(operand_terms) < 2:
                raise ValueError(
                    "memref.load currently requires exactly one index term")
            memref_term = operand_terms[0]  # MemRef to load from
            index_terms = operand_terms[1:]
            if len(index_terms) != 1:
                raise ValueError(
                    f"memref.load expects one index term, got {len(index_terms)}")
            return Term.load(memref_term, index_terms[0], ty_str)

        raise ValueError(f"MOperation {op.name} is not supported")

    def _is_pure_operation(self, op: MOperation) -> bool:
        """
        Check if an operation is pure (no side effects and no control flow).

        Pure operations can be freely optimized in the e-graph.
        Side-effect operations and control flow must be serialized in Vec to preserve order.
        """
        op_type = op.type if hasattr(op, 'type') else OperationType.UNKNOWN

        # Control flow and side-effect operations must be serialized
        side_effect_types = {
            # Control flow
            *CONTROL_FLOW_OPS,
            # Memory writes
            OperationType.MEMREF_STORE, OperationType.AFFINE_STORE,
            OperationType.MEMREF_ALLOC, OperationType.MEMREF_DEALLOC,
            OperationType.MEMREF_ALLOCA,
            # Terminators
            OperationType.FUNC_RETURN, OperationType.SCF_YIELD,
        }

        if op_type in side_effect_types:
            return False

        # Unknown operations are considered pure
        return True

    def _get_block_stmts(self, block: MBlock, loop_id: Optional[int] = None, num_iter_args: int = 0) -> egglog.Vec[Term]:
        """
        Process a block and return only the statement vector (Vec[Term]) without wrapping in Term.block.
        This is useful for loop bodies where we don't need the Block wrapper.
        """
        # Process block arguments (similar to _block_to_term)
        for i, arg in enumerate(block.arguments):
            if loop_id is not None and i < num_iter_args + 1:
                # loop block arguments
                if i == 0:
                    # loop index
                    term = Term.loop_index(egglog.i64(
                        loop_id), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(f"{self.session_id}_loop_{loop_id}_index", term)
                else:
                    # loop iter args
                    iter_arg_index = i - 1
                    term = Term.loop_iter_arg(egglog.i64(loop_id), egglog.i64(
                        iter_arg_index), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(
                        f"{self.session_id}_loop_{loop_id}_iter_arg_{iter_arg_index}", term)
            else:
                # regular block arguments
                term = Term.block_arg(id(block), egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_block_{id(block)}_arg_{i}", term)

            # register argument term
            self.ssa_to_term[arg] = term_ref
            arg_id = self._next_id()
            self.ssa_to_id[arg] = arg_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(arg_id)))

        # Collect serialized terms (side effects + terminators)
        serialized_terms = []

        for op in block.operations:
            op_ty = op.type
            # Only serialize terminators (yield, return, etc.) in nested blocks
            if op_ty in [OperationType.SCF_YIELD, OperationType.AFFINE_YIELD, OperationType.AFFINE_STORE, OperationType.MEMREF_STORE, OperationType.MEMREF_ALLOC, OperationType.MEMREF_ALLOCA]:
                operand_terms = self._get_operand_terms(
                    op, block, loop_id, num_iter_args)
                term = self._operation_to_term(
                    op, operand_terms, loop_id, num_iter_args)
                if term is not None:
                    print(f"[BLOCK_STMTS] Serializing {op.name} in loop body")
                    serialized_terms.append(term)
            else:
                # For pure operations, just track SSA mappings without serializing
                operand_terms = self._get_operand_terms(
                    op, block, loop_id, num_iter_args)
                term = self._operation_to_term(
                    op, operand_terms, loop_id, num_iter_args)
                if term is not None:
                    # Just track SSA values for all operations
                    for result in op.results:
                        self.ssa_to_term[result] = term
                    print(
                        f"[BLOCK_STMTS] Skipping pure op {op.name} in loop body")

        return egglog.Vec[Term](*serialized_terms)

    def _block_to_term(
        self,
        block: MBlock,
        loop_id: Optional[int] = None,
        num_iter_args: int = 0,
        is_function_block: bool = False
    ) -> Optional[Term]:
        """
        Unified block processing for all contexts (function, loop, if/else, etc.).

        Strategy:
        - Pure operations (add, mul, cmp, etc.) are represented directly as e-graph nodes
        - Side-effect operations (store, alloc) and terminators are wrapped in Block(Vec(...))

        This allows pure ops to be optimized via equality saturation while preserving
        execution order for side-effectful operations.

        Args:
            block: The MLIR block to convert
            loop_id: Optional loop ID for loop index tracking
            num_iter_args: Number of loop-carried values
            is_function_block: True if this is a function's entry block

        Returns:
            A Term.block(...) containing side-effect operations in order
        """
        if not block or not block.operations:
            return None

        # block arguments
        for i, arg in enumerate(block.arguments):
            if is_function_block:
                # function arguments
                # Use unique name to avoid shadowing when transforming multiple functions
                term = Term.arg(egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_func_{id(block)}_arg_{i}", term)
            elif loop_id is not None:
                # loop block arguments
                if i == 0:
                    # loop index
                    term = Term.loop_index(egglog.i64(
                        loop_id), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(f"{self.session_id}_loop_{loop_id}_index", term)
                else:
                    # loop iter args
                    iter_arg_index = i - 1
                    term = Term.loop_iter_arg(egglog.i64(loop_id), egglog.i64(
                        iter_arg_index), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(
                        f"{self.session_id}_loop_{loop_id}_iter_arg_{iter_arg_index}", term)
            else:
                # regular block arguments
                term = Term.block_arg(id(block), egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_block_{id(block)}_arg_{i}", term)

            # register argument term
            self.ssa_to_term[arg] = term_ref
            arg_id = self._next_id()
            self.ssa_to_id[arg] = arg_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(arg_id)))

        # Collect operations that must be serialized (side effects + terminators)
        serialized_terms = []
        if (terminator := block.get_terminator()):
            # print(f"Processing terminator: {len(terminator.operands)}")
            if len(terminator.operands) > 0:
                ret_type = terminator.operands[0].type
                block_type = mlir_type_to_egraph_ty_string(ret_type)
            else:
                block_type = egglog.String("void")
        else:
            block_type = egglog.String("void")  # Default type

        # Collect operations that MUST be serialized in block.vec
        # Only include:
        # 1. Side-effect operations (memref.store, control flow, etc.)
        # 2. Terminators (return, yield, condition)
        # Pure operations will be reconstructed on-demand during extraction

        # IMPORTANT: Cache operations list to avoid getting different object instances
        all_ops = list(block.operations)
        need_serialize_ops = set()

        for op in all_ops:
            op_ty = op.type
            print(f"Analyzing op for serialization: {op.name}, type: {op_ty}")
            # Only serialize non-pure operations and terminators
            if not self._is_pure_operation(op) or op_ty in [OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]:
                need_serialize_ops.add(op_ty)

        # 第二个pass，处理所有操作
        for op in all_ops:
            op_ty = op.type
            if op_ty in [OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]:
                if op_ty == OperationType.SCF_CONDITION:
                    # --- ConditionOp ---
                    if not op.operands:
                        raise RuntimeError("scf.ConditionOp has no operands")

                    cond_operand = op.operands[0]
                    if cond_operand in self.ssa_to_term:
                        cond_term = self.ssa_to_term[cond_operand]
                    else:
                        cond_term = Term.block_arg(id(block), egglog.i64(
                            0), mlir_type_to_egraph_ty_string(cond_operand.type))
                        cond_term = self.egraph.let(
                            f"{self.session_id}_block_{id(block)}_cond", cond_term)
                        self.ssa_to_term[cond_operand] = cond_term
                        cond_id = self._next_id()
                        self.ssa_to_id[cond_operand] = cond_id
                        self.egraph.register(egglog.set_(
                            id_of(cond_term)).to(egglog.i64(cond_id)))

                    # iter_vars
                    iter_terms = []
                    for i, iter_operand in enumerate(op.operands[1:]):
                        if iter_operand in self.ssa_to_term:
                            iter_term = self.ssa_to_term[iter_operand]
                        else:
                            iter_term = Term.block_arg(id(block), egglog.i64(
                                i), mlir_type_to_egraph_ty_string(iter_operand.type))
                            iter_term = self.egraph.let(
                                f"{self.session_id}_block_{id(block)}_cond_arg_{i}", iter_term)
                            self.ssa_to_term[iter_operand] = iter_term
                            iter_id = self._next_id()
                            self.ssa_to_id[iter_operand] = iter_id
                            self.egraph.register(egglog.set_(
                                id_of(iter_term)).to(egglog.i64(iter_id)))
                        iter_terms.append(iter_term)

                    cond_block_term = Term.condition(
                        cond_term, egglog.Vec(*iter_terms), block_type)
                    serialized_terms.append(cond_block_term)
                else:
                    # --- Yield or Return ---
                    operand_terms = self._get_operand_terms(
                        op, block, loop_id, num_iter_args)
                    term = self._operation_to_term(
                        op, operand_terms, loop_id, num_iter_args)
                    if term is not None:
                        serialized_terms.append(term)

                continue

            # Get operand terms for this operation
            operand_terms = self._get_operand_terms(
                op, block, loop_id, num_iter_args)

            # Convert operation to term
            term = self._operation_to_term(
                op, operand_terms, loop_id, num_iter_args)

            if term is None:
                continue

            if is_function_block:
                serialize_op = (op.type in need_serialize_ops)
                # Function block: register in e-graph
                if len(op.results) == 0:
                    # No results (side effect operation) - must serialize
                    unique_id = self._next_id()
                    name_hint = f"{self.session_id}_side_effect_{id(op)}_{unique_id}"
                    term_ref = self.egraph.let(name_hint, term)
                    serialized_terms.append(term_ref)
                elif len(op.results) == 1:
                    result = op.results[0]
                    name_hint = f"{self.session_id}_op_{id(result)}"
                    term_ref = self.egraph.let(name_hint, term)
                    self.ssa_to_term[result] = term_ref
                    result_id = self._next_id()
                    self.ssa_to_id[result] = result_id
                    self.egraph.register(egglog.set_(
                        id_of(term_ref)).to(egglog.i64(result_id)))

                    # Only serialize if in need_serialize_ops
                    if serialize_op:
                        serialized_terms.append(term_ref)
                    else:
                        print(
                            f"[SKIP_FUNC_BLOCK] Skipping {op.name} (has result but not in need_serialize_ops)")
                else:
                    # Multiple results
                    # TODO: handle multiple results properly
                    main_result = op.results[0]
                    name_hint = f"{self.session_id}_op_{id(main_result)}"
                    term_ref = self.egraph.let(name_hint, term)
                    for i, result in enumerate(op.results):
                        self.ssa_to_term[result] = term_ref
                        result_id = self._next_id()
                        self.ssa_to_id[result] = result_id
                        # self.egraph.register(egglog.set_(id_of(term_ref)).to(egglog.i64(result_id)))

                    # Only serialize if in need_serialize_ops
                    if serialize_op:
                        serialized_terms.append(term_ref)
            else:
                # Nested blocks (loop bodies, if branches, etc.)
                # Just track SSA values for all operations
                for result in op.results:
                    self.ssa_to_term[result] = term

                # For nested blocks, serialize based on operation type (not need_serialize_ops)
                # because need_serialize_ops was computed for the parent block
                is_terminator = op_ty in [
                    OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]
                should_serialize = not self._is_pure_operation(
                    op) or is_terminator
                if should_serialize:
                    print(
                        f"[SERIALIZE_NESTED] Serializing {op.name} in nested block")
                    serialized_terms.append(term)
                else:
                    print(
                        f"[SERIALIZE_NESTED] Skipping pure op {op.name} in nested block")

        # Create block term only with serialized (side-effect + terminator) operations
        if serialized_terms:
            block_id = id(block)
            block_term = Term.block(egglog.i64(
                block_id), egglog.Vec(*serialized_terms), block_type)

            if is_function_block:
                # Use unique name to avoid shadowing when transforming multiple functions
                # in the same e-graph (e.g., original + optimized functions)
                block_name = f"{self.session_id}_func_body_{id(block)}"
            else:
                block_name = f"{self.session_id}_block_{id(block)}"

            term_ref = self.egraph.let(block_name, block_term)

            # 注册 block 的映射
            self.ssa_to_term[block] = term_ref
            block_id = self._next_id()
            self.ssa_to_id[block] = block_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(block_id)))

            # 特别记录函数体的 top_block
            if is_function_block:
                self.top_block = term_ref

            return term_ref
        else:
            # Empty block or only pure operations - create block with default void yield
            return Term.block(
                id(block),
                egglog.Vec(Term.yield_(egglog.Vec(), egglog.String("void"))),
                block_type
            )

    def _get_operand_terms(
        self,
        op: MOperation,
        block: MBlock,
        loop_id: Optional[int] = None,
        num_iter_args: int = 0
    ) -> List[Term]:
        """Extract operand terms for an operation, resolving block arguments."""
        operand_terms = []
        for operand in op.operands:

            # First try direct lookup
            found = False
            if operand in self.ssa_to_term:
                operand_terms.append(self.ssa_to_term[operand])
                found = True
            else:
                # Try to find by string representation since MValue comparison might not work
                operand_str = str(operand)
                for key, value in self.ssa_to_term.items():
                    if str(key) == operand_str:
                        operand_terms.append(value)
                        found = True
                        break

            if not found:
                # Try to find the operand as a block argument
                block_arg_index = None
                for i, block_arg in enumerate(block.arguments):
                    if str(operand) == str(block_arg):
                        block_arg_index = i
                        break

                if block_arg_index is not None:
                    # Create appropriate term based on context
                    if loop_id is not None and block_arg_index == 0:
                        # First block argument in a loop is the induction variable
                        arg_term = Term.loop_index(
                            egglog.i64(loop_id),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    elif loop_id is not None and 0 < block_arg_index <= num_iter_args:
                        # Subsequent arguments are loop-carried values
                        arg_term = Term.loop_iter_arg(
                            egglog.i64(loop_id),
                            egglog.i64(block_arg_index - 1),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    else:
                        # Regular block argument (including function parameters)
                        arg_term = Term.arg(
                            id(block),
                            egglog.i64(block_arg_index),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    operand_terms.append(arg_term)
        return operand_terms

    def _constant_to_term(self, op: MOperation) -> Optional[Term]:
        """Convert a constant operation to a Term."""
        # For MLIR constant operations, extract the value attribute using the new method
        if op.has_attr('value'):
            raw_value = op.get_attr('value')
            # result_type is now a string from MValue.type
            result_type_str = op.results[0].type if op.results else "i32"

            try:
                # Check if it's a float type by string matching
                if result_type_str in ['f16', 'f32', 'f64', 'float', 'double']:
                    # Float constant
                    if raw_value is not None:
                        float_val = float(raw_value)
                        return Term.lit(LitTerm.float(egglog.f64(float_val)), mlir_type_to_egraph_ty_string(result_type_str))
                elif result_type_str == 'index' or result_type_str.startswith('i') or result_type_str.startswith('u'):
                    # Integer constant (including index type)
                    if raw_value is not None:
                        int_val = int(raw_value)
                        return Term.lit(LitTerm.int(egglog.i64(int_val)), mlir_type_to_egraph_ty_string(result_type_str))
            except (ValueError, TypeError):
                pass

        # Fallback: create a default constant
        result_type_str = op.results[0].type if op.results else "i32"
        if result_type_str == 'index':
            ty_str = egglog.String(str(IndexType()))
            return Term.lit(LitTerm.int(egglog.i64(0)), ty_str)
        elif result_type_str in ['f16', 'f32', 'f64', 'float', 'double']:
            return Term.lit(LitTerm.float(egglog.f64(0.0)), mlir_type_to_egraph_ty_string(result_type_str))
        else:
            # Default to integer
            return Term.lit(LitTerm.int(egglog.i64(0)), mlir_type_to_egraph_ty_string(result_type_str))

    def _next_id(self) -> int:
        self.next_id += 1
        return self.next_id


def create_tuple_type_string(types: list) -> egglog.String:
    """Create a tuple type string for multiple values."""
    megg_types = []
    for t in types:
        megg_type = mlir_type_to_megg_type(t)
        if megg_type:
            megg_types.append(megg_type)

    if len(megg_types) == 0:
        return egglog.String("void")
    elif len(megg_types) == 1:
        return egglog.String(str(megg_types[0]))
    else:
        tuple_type = TupleType(element_types=tuple(megg_types))
        return egglog.String(str(tuple_type))


def _parse_type_from_string(type_str: str) -> Optional[DataType]:
    """Parse a type string to DataType (helper for MLIR -> Megg types)."""
    type_str = type_str.strip().lower()

    # Float types
    if type_str in ("f16", "half"):
        return MeggFloatType.float_type("f16")
    elif type_str in ("f32", "single", "float"):
        return MeggFloatType.float_type("f32")
    elif type_str in ("f64", "double"):
        return MeggFloatType.float_type("f64")

    # Index and void
    elif type_str == "index":
        return IndexType()
    elif type_str == "void":
        return VoidType()

    # Signed integer, e.g., i32
    elif type_str.startswith("i") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return IntType(width=width, signed=True)

    # Unsigned integer, e.g., u8
    elif type_str.startswith("u") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return IntType(width=width, signed=False)

    # MemRef type, e.g., memref<4x8xf32>
    elif type_str.startswith("memref<") and type_str.endswith(">"):
        inner = type_str[len("memref<"):-1]
        parts = inner.split("x")
        # 最后一个部分是 element_type，其余是 shape
        *shape_parts, elem_type_str = parts
        shape = tuple(int(d) for d in shape_parts)
        elem_type = _parse_type_from_string(elem_type_str)
        if elem_type is None:
            return None
        return MemRefType(shape=shape, element_type=elem_type)

    # Fallback: unknown type
    return None


def megg_to_mlir_type(megg_type: DataType) -> MType:
    """
    Convert Megg DataType to MLIR Type using C++ wrapper.

    Args:
        megg_type: Megg DataType instance

    Returns:
        Corresponding MLIR Type (MLIRType from C++ bindings)
    """
    builder = IRBuilder()
    # Get the string representation
    type_str = str(megg_type)

    # Float types
    if type_str in ("f16", "half"):
        # Note: f16 not directly supported in IRBuilder, use f32 as fallback
        return builder.f32()
    elif type_str in ("f32", "single", "float"):
        return builder.f32()
    elif type_str in ("f64", "double"):
        return builder.f64()

    # Index and void
    elif type_str == "index":
        return builder.index()
    elif type_str == "void":
        # Note: void type may need special handling
        # Using i1 as a placeholder for void
        return builder.integer(1)

    # Signed integer, e.g., i32
    elif type_str.startswith("i") and type_str[1:].isdigit():
        width = int(type_str[1:])
        if width == 32:
            return builder.i32()
        elif width == 64:
            return builder.i64()
        else:
            return builder.integer(width)

    # Unsigned integer, e.g., u8 (MLIR uses signless for unsigned)
    elif type_str.startswith("u") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return builder.integer(width)

    # MemRef type, e.g., memref<4x8xf32>
    elif type_str.startswith("memref<") and type_str.endswith(">"):
        inner = type_str[len("memref<"):-1]
        parts = inner.split("x")
        # Last part is element_type, rest are shape dimensions
        *shape_parts, elem_type_str = parts
        shape = [int(dim) for dim in shape_parts]
        # Recursively parse element type
        elem_type = _parse_type_from_string(elem_type_str)
        mlir_elem_type = megg_to_mlir_type(elem_type)
        return builder.memref(shape, mlir_elem_type)

    # Fallback: default to i32
    return builder.i32()


def mlir_type_to_megg_type(mlir_type) -> Optional[DataType]:
    """Convert an MLIR type to a Megg DataType."""
    try:
        type_str = str(mlir_type)

        # Try simple type parsing first
        simple_type = _parse_type_from_string(type_str)
        if simple_type:
            return simple_type

        # Handle tuple types (multiple return values)
        if isinstance(mlir_type, tuple):
            element_types = []
            for elem_type in mlir_type:
                elem_megg_type = mlir_type_to_megg_type(elem_type)
                if elem_megg_type:
                    element_types.append(elem_megg_type)
            if element_types:
                return TupleType(element_types=tuple(element_types))

        # Handle complex types (vector, tensor, memref)
        if 'vector' in type_str:
            return VectorType(element_type=IntType(32), size=1)
        elif 'tensor' in type_str:
            return TensorType(shape=(1,), element_type=IntType(32))
        elif 'memref' in type_str:
            # Parse memref type like "memref<2xi32>" or "memref<4x8xf32>"
            import re
            match = re.search(r'memref<(.*)>', type_str)
            if match:
                type_content = match.group(1)
                parts = type_content.split('x')
                if len(parts) >= 2:
                    shape_parts = parts[:-1]
                    element_type_str = parts[-1]
                    try:
                        shape = tuple(int(dim) for dim in shape_parts)
                        element_type = _parse_type_from_string(
                            element_type_str)
                        if element_type:
                            return MemRefType(shape=shape, element_type=element_type)
                    except ValueError:
                        pass
            return MemRefType(shape=(1,), element_type=IntType(32))
        elif 'none' in type_str.lower():
            return VoidType()

        # Default for unknown types
        return VoidType()
    except Exception:
        return VoidType()


def mlir_type_to_egraph_ty_string(mlir_type) -> egglog.String:
    """Convert an MLIR type to an e-graph type string."""
    megg_ty = mlir_type_to_megg_type(mlir_type)
    if megg_ty is None:
        raise ValueError(f"Unsupported MLIR type: {mlir_type}")
    return egglog.String(str(megg_ty))


def predicate_to_string(predicate_enum, cmp_type="i"):
    """
    Convert MLIR compare predicate enum to a human-readable string.

    Args:
        predicate_enum (int or str): The enum value of the predicate.
        cmp_type (str): "i" for integer compare (cmpi), "f" for float compare (cmpf).

    Returns:
        str: Corresponding string representation of the predicate.

    Raises:
        ValueError: If the predicate_enum or cmp_type is unknown.
    """
    if cmp_type == "i":
        # Integer comparison predicates
        mapping = {
            0: "eq",   # equal
            1: "ne",   # not equal
            2: "slt",  # signed less than
            3: "sle",  # signed less or equal
            4: "sgt",  # signed greater than
            5: "sge",  # signed greater or equal
            6: "ult",  # unsigned less than
            7: "ule",  # unsigned less or equal
            8: "ugt",  # unsigned greater than
            9: "uge",  # unsigned greater or equal
            "eq": "eq",
            "ne": "ne",
            "slt": "slt",
            "sle": "sle",
            "sgt": "sgt",
            "sge": "sge",
            "ult": "ult",
            "ule": "ule",
            "ugt": "ugt",
            "uge": "uge",
        }
    elif cmp_type == "f":
        # Floating-point comparison predicates
        mapping = {
            0: "false",      # always false
            1: "oeq",        # ordered and equal
            2: "ogt",        # ordered and greater than
            3: "oge",        # ordered and greater or equal
            4: "olt",        # ordered and less than
            5: "ole",        # ordered and less or equal
            6: "one",        # ordered and not equal
            7: "ord",        # ordered (no nans)
            8: "uno",        # unordered (any operand is nan)
            9: "ueq",        # unordered or equal
            10: "ugt",       # unordered or greater than
            11: "uge",       # unordered or greater or equal
            12: "ult",       # unordered or less than
            13: "ule",       # unordered or less or equal
            14: "une",       # unordered or not equal
            15: "true",      # always true
            "oeq": "oeq",
            "ogt": "ogt",
            "oge": "oge",
            "olt": "olt",
            "ole": "ole",
            "one": "one",
            "ord": "ord",
            "uno": "uno",
            "ueq": "ueq",
            "ugt": "ugt",
            "uge": "uge",
            "ult": "ult",
            "ule": "ule",
            "une": "une",
        }
    else:
        raise ValueError(f"Unknown cmp_type: {cmp_type}")

    if predicate_enum in mapping:
        return mapping[predicate_enum]
    else:
        raise ValueError(f"Unknown predicate_enum: {predicate_enum}")
