"""
Loop Transformation Module

This module handles the analysis and transformation of CADL do-while loops
to MLIR scf.while and scf.for operations.
"""

from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass

import circt.ir as ir
from circt.dialects import arith, scf

from .ast import (
    DoWhileStmt, BinaryExpr, BinaryOp, IdentExpr,
    LitExpr, AssignStmt, Expr
)
from .debug import dbg_print, dbg_debug, dbg_info, dbg_warning, DebugLevel

@dataclass
class ForLoopPattern:
    """Pattern information for raising do-while to scf.for"""
    induction_var: Any  # WithBinding for the induction variable
    induction_step: int  # Step value (positive or negative)
    lower_bound: int  # Initial value
    upper_bound: int  # Adjusted upper bound (exclusive for scf.for)
    comparison_op: Any  # BinaryOp for the condition
    other_bindings: List[Any]  # Other loop-carried variables


class LoopTransformer:
    """
    Handles transformation of CADL do-while loops to MLIR operations.

    This class analyzes loop patterns and emits appropriate MLIR code:
    - scf.for for counted loops with affine induction variables
    - scf.while for general do-while loops
    """

    def __init__(self, converter):
        """
        Initialize the loop transformer.

        Args:
            converter: The parent CADLMLIRConverter instance
        """
        self.converter = converter
        self.constant_vars = converter.constant_vars

    def validate_loop_body_assignments(self, stmt: DoWhileStmt) -> None:
        """
        Validate that only loop-carried variables (in 'with' bindings) and local variables
        (declared with 'let' inside the loop) are assigned in the loop body.

        Raises RuntimeError if an external variable is modified.
        """
        dbg_debug("Validating loop body assignments", module="loop_transform")

        # Collect loop-carried variable names (from bindings)
        loop_carried_vars = set()
        for binding in stmt.bindings:
            loop_carried_vars.add(binding.id)
            # Also allow assignment to the "next" variable if it's an identifier
            if isinstance(binding.next, IdentExpr):
                loop_carried_vars.add(binding.next.name)

        dbg_debug(f"Loop-carried variables: {loop_carried_vars}", module="loop_transform")

        # Collect local variables declared in the loop body (let statements)
        local_vars = set()
        for body_stmt in stmt.body:
            if isinstance(body_stmt, AssignStmt):
                # Check if this is a 'let' statement (has type annotation in AST)
                if isinstance(body_stmt.lhs, IdentExpr):
                    lhs_name = body_stmt.lhs.name
                    # If it's a new variable declaration, add to local_vars
                    # In CADL, 'let x = ...' creates a new binding
                    # We need to track this during parsing, but for now we'll be permissive
                    # and check if it was NOT previously defined
                    if lhs_name not in self.converter.current_scope.symbols:
                        local_vars.add(lhs_name)

        # Now check all assignments in body
        for body_stmt in stmt.body:
            if isinstance(body_stmt, AssignStmt):
                if isinstance(body_stmt.lhs, IdentExpr):
                    var_name = body_stmt.lhs.name

                    # Allow if it's a loop-carried variable or local variable
                    if var_name in loop_carried_vars or var_name in local_vars:
                        continue

                    # Check if this is a new local declaration (appears first time in this loop)
                    # If variable exists in outer scope, this is an illegal modification
                    if self.converter.get_symbol(var_name) is not None:
                        dbg_warning(
                            f"Invalid assignment to '{var_name}' in loop body",
                            module="loop_transform"
                        )
                        raise RuntimeError(
                            f"CADL semantic error: Variable '{var_name}' cannot be reassigned inside "
                            f"'do-while' loop body. Only loop-carried variables (in 'with' bindings) "
                            f"or locally declared variables (with 'let') can be assigned in the loop body.\n"
                            f"Hint: If '{var_name}' should change across iterations, add it to the 'with' clause."
                        )

    def analyze_and_detect_for_pattern(self, stmt: DoWhileStmt) -> Optional[ForLoopPattern]:
        """
        Analyze do-while loop to detect for-loop patterns.
        Returns ForLoopPattern if detectable, None otherwise.
        Prints minimal analysis results.
        """
        dbg_debug("Analyzing loop for scf.for pattern detection", module="loop_transform")

        can_be_for = True
        reasons = []
        induction_var = None
        induction_step = None

        # Find the induction variable (one with affine increment)
        dbg_debug(f"Checking {len(stmt.bindings)} bindings for induction variable", module="loop_transform")
        for binding in stmt.bindings:
            step = self._try_extract_step(binding, stmt.body)
            if step is not None:
                # Check if init is constant
                if self._is_constant_expr(binding.init):
                    induction_var = binding
                    induction_step = step
                    break

        if induction_var is None:
            can_be_for = False
            reasons.append("No affine induction variable found")
            dbg_debug("No induction variable found with constant step", module="loop_transform")
        else:
            dbg_debug(f"Found induction variable: {induction_var.id} (step={induction_step})", module="loop_transform")
            # Note: Other bindings (iter_args) can have non-constant inits.
            # scf.for accepts any SSA value as initial value for loop-carried variables.

            # Check simple bound - must reference the induction variable
            if isinstance(stmt.condition, BinaryExpr):
                # Reject == operator as it's not valid for counted loops
                if stmt.condition.op == BinaryOp.EQ:
                    can_be_for = False
                    reasons.append("Invalid comparison operator: == (not valid for counted loops)")
                elif stmt.condition.op in [BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE, BinaryOp.NE]:
                    # Check if condition involves the induction variable
                    condition_involves_iv = False
                    if isinstance(stmt.condition.left, IdentExpr):
                        if stmt.condition.left.name == induction_var.id or \
                           (isinstance(induction_var.next, IdentExpr) and
                            stmt.condition.left.name == induction_var.next.name):
                            condition_involves_iv = True
                    elif isinstance(stmt.condition.left, BinaryExpr):
                        # Handle cases like (i + 1) < N
                        condition_involves_iv = True

                    if not condition_involves_iv:
                        can_be_for = False
                        reasons.append("Condition doesn't reference induction variable")
                    elif self._is_constant_expr(stmt.condition.right):
                        # Check if bound variable is modified in loop body
                        if isinstance(stmt.condition.right, IdentExpr):
                            bound_var = stmt.condition.right.name
                            if self._is_variable_modified_in_body(stmt.body, bound_var):
                                can_be_for = False
                                reasons.append(f"Bound variable '{bound_var}' modified in loop body")
                    else:
                        can_be_for = False
                        reasons.append("Non-constant bound")
                else:
                    can_be_for = False
                    reasons.append("Non-comparison operator")
            else:
                can_be_for = False
                reasons.append("Complex condition")

        # Print minimal summary (one line per loop)
        if can_be_for:
            init_val = self._extract_constant_value(induction_var.init)
            bound_val = self._extract_constant_value(stmt.condition.right)
            comparison_op = stmt.condition.op
            adjusted_bound = bound_val

            # Adjust bound for scf.for based on comparison operator
            if comparison_op == BinaryOp.LE:
                adjusted_bound = bound_val + 1
            elif comparison_op == BinaryOp.GE:
                adjusted_bound = bound_val - 1

            other_vars = [b for b in stmt.bindings if b.id != induction_var.id]
            iter_args_str = f", iter_args=[{', '.join(b.id for b in other_vars)}]" if other_vars else ""

            dbg_info(
                f"Loop -> scf.for: {induction_var.id}={init_val}..{adjusted_bound} step {induction_step}{iter_args_str}",
                module="loop_transform"
            )

            # Return the pattern
            return ForLoopPattern(
                induction_var=induction_var,
                induction_step=induction_step,
                lower_bound=init_val,
                upper_bound=adjusted_bound,
                comparison_op=comparison_op,
                other_bindings=other_vars
            )
        else:
            dbg_info(
                f"Loop -> scf.while: {', '.join(reasons)}",
                module="loop_transform"
            )
            return None

    # Helper methods

    def _is_constant_expr(self, expr: Optional[Expr]) -> bool:
        """Check if expression is a constant literal or references a constant variable"""
        if expr is None:
            return False

        # Direct literal
        if isinstance(expr, LitExpr):
            return True

        # Variable that may hold a constant value
        if isinstance(expr, IdentExpr):
            # Check if this identifier refers to a constant variable
            try:
                # Check if we can trace it back to a constant
                return self._is_constant_value(expr.name)
            except:
                return False

        return False

    def _is_constant_value(self, var_name: str) -> bool:
        """Check if a variable was initialized with a constant value"""
        return var_name in self.constant_vars

    def _extract_constant_value(self, expr: Expr) -> Optional[int]:
        """Extract constant value from literal expression or constant variable"""
        if isinstance(expr, LitExpr):
            if hasattr(expr.literal.lit, 'value'):
                return expr.literal.lit.value
        elif isinstance(expr, IdentExpr):
            # Check if this variable holds a constant value
            if expr.name in self.constant_vars:
                return self.constant_vars[expr.name]
        return None

    def _try_extract_step(self, binding, body: List) -> Optional[int]:
        """Try to extract constant step from binding's next expression.
        Returns positive for increment, negative for decrement."""
        next_expr = binding.next

        # Pattern 1: next = i + 1 or i - 1
        if isinstance(next_expr, BinaryExpr):
            if next_expr.op == BinaryOp.ADD:
                if isinstance(next_expr.left, IdentExpr) and next_expr.left.name == binding.id:
                    if self._is_constant_expr(next_expr.right):
                        return self._extract_constant_value(next_expr.right)
            elif next_expr.op == BinaryOp.SUB:
                if isinstance(next_expr.left, IdentExpr) and next_expr.left.name == binding.id:
                    if self._is_constant_expr(next_expr.right):
                        # Negative step for decrement
                        return -self._extract_constant_value(next_expr.right)

        # Pattern 2: next = i_ (need to analyze body)
        if isinstance(next_expr, IdentExpr):
            next_var = next_expr.name
            # Look for assignment: i_ = i +/- constant in body
            step = self._find_step_in_body(body, binding.id, next_var)
            return step

        return None

    def _is_variable_modified_in_body(self, stmts: List, var_name: str) -> bool:
        """Check if a variable is modified (assigned to) in the statement list"""
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                if isinstance(stmt.lhs, IdentExpr) and stmt.lhs.name == var_name:
                    return True
        return False

    def _find_step_in_body(self, stmts: List, iv_name: str, next_var: str) -> Optional[int]:
        """Find assignment of form: next_var = iv +/- constant in statement list.
        Returns positive for increment, negative for decrement."""
        for stmt in stmts:
            # Check for assignment: i_ = i + 1 or i_ = i - 1
            if isinstance(stmt, AssignStmt):
                if isinstance(stmt.lhs, IdentExpr) and stmt.lhs.name == next_var:
                    # Check if rhs is i + constant
                    if isinstance(stmt.rhs, BinaryExpr):
                        if isinstance(stmt.rhs.left, IdentExpr) and stmt.rhs.left.name == iv_name:
                            if self._is_constant_expr(stmt.rhs.right):
                                const_val = self._extract_constant_value(stmt.rhs.right)
                                if stmt.rhs.op == BinaryOp.ADD:
                                    return const_val
                                elif stmt.rhs.op == BinaryOp.SUB:
                                    return -const_val

        return None

    def emit_scf_for(self, stmt: DoWhileStmt, pattern: ForLoopPattern) -> None:
        """
        Emit scf.for operation for loops matching the for-loop pattern.

        Handles:
        - Induction variable (loop counter)
        - Loop-carried variables (iter_args for state like crc)
        - Positive and negative step values
        """
        from circt.dialects import arith, scf
        from circt import ir

        # Extract pattern information
        iv_binding = pattern.induction_var
        lower_bound = pattern.lower_bound
        upper_bound = pattern.upper_bound
        step = pattern.induction_step

        dbg_debug(
            f"Emitting scf.for: {iv_binding.id}=[{lower_bound}..{upper_bound}] step {step}",
            module="loop_transform"
        )

        # Determine if this is a forward or backward loop
        is_forward = step > 0

        # Convert bounds and step to MLIR constants
        iv_type = self.converter.convert_cadl_type(iv_binding.ty)
        lower_const = arith.ConstantOp(iv_type, lower_bound).result
        upper_const = arith.ConstantOp(iv_type, upper_bound).result
        step_const = arith.ConstantOp(iv_type, abs(step)).result

        # Collect initial values for loop-carried variables (iter_args)
        init_values = []
        for binding in pattern.other_bindings:
            if binding.init:
                init_val = self.converter._convert_expr(binding.init)
            else:
                var_type = self.converter.convert_cadl_type(binding.ty)
                init_val = arith.ConstantOp(var_type, 0).result
            init_values.append(init_val)

        # Create scf.for operation
        # For backward loops, we still use the same scf.for but will need to adjust IV
        for_op = scf.ForOp(lower_const, upper_const, step_const, init_values)

        # Apply directives as attributes (generalized)
        if self.converter.pending_directives:
            from .ast import LitExpr
            for directive in self.converter.pending_directives:
                attr_name = directive.name
                if directive.expr:
                    # Convert expression to attribute value
                    if isinstance(directive.expr, LitExpr):
                        value = directive.expr.literal.lit.value
                        # Set as integer attribute on the operation
                        attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
                        for_op.operation.attributes[attr_name] = attr
                else:
                    # Boolean flag directive (no argument)
                    for_op.operation.attributes[attr_name] = ir.BoolAttr.get(True)
            # Clear directives after applying to loop (before processing body)
            self.converter.pending_directives = []

        # Body block
        body_block = for_op.region.blocks[0]
        with ir.InsertionPoint(body_block):
            self.converter.push_scope()

            # Bind induction variable
            # For backward loops, we need to compute the actual IV value
            if is_forward:
                # Forward loop: IV is directly the block argument
                iv_value = body_block.arguments[0]
            else:
                # Backward loop: IV = upper_bound - (current - lower_bound) - 1
                # This transforms ascending iteration into descending
                # Example: for i = 8 to 0 step -1
                #   iteration 0: iv = 8 - (0 - 0) = 8
                #   iteration 1: iv = 8 - (1 - 0) = 7
                #   ...
                current_iter = body_block.arguments[0]
                # Compute: upper_bound - current_iter + lower_bound
                diff = arith.SubIOp(upper_const, current_iter).result
                iv_value = arith.AddIOp(diff, lower_const).result

            self.converter.set_symbol(iv_binding.id, iv_value)

            # Bind loop-carried variables (iter_args)
            for i, binding in enumerate(pattern.other_bindings):
                # Block arguments: [IV, iter_arg_0, iter_arg_1, ...]
                self.converter.set_symbol(binding.id, body_block.arguments[i + 1])

            # Convert loop body
            self.converter._convert_stmt_list(stmt.body)

            # Collect next values for iter_args
            next_values = []
            for binding in pattern.other_bindings:
                if isinstance(binding.next, IdentExpr):
                    # Get the updated value from body (e.g., crc_ for crc)
                    next_val = self.converter.get_symbol(binding.next.name)
                else:
                    # Evaluate next expression
                    next_val = self.converter._convert_expr(binding.next)
                next_values.append(next_val)

            # Yield next values
            scf.YieldOp(next_values)
            self.converter.pop_scope()

        # Update symbols with final values after loop
        for i, binding in enumerate(pattern.other_bindings):
            if i < len(for_op.results):
                self.converter.set_symbol(binding.id, for_op.results[i])

        # Also update the induction variable to its final value
        # For forward: final IV = upper_bound
        # For backward: final IV = lower_bound (or upper_bound depending on semantics)
        # In CADL, the IV after loop should be the last value used
        if is_forward:
            self.converter.set_symbol(iv_binding.id, upper_const)
        else:
            self.converter.set_symbol(iv_binding.id, lower_const)

    def emit_scf_while(self, stmt: DoWhileStmt) -> None:
        """
        Emit scf.while operation for general do-while loops.

        CADL do-while semantics: body executes at least once, condition checked after.
        Uses scf.while with a first_iteration flag to ensure at least one execution.
        """
        from circt.dialects import arith, scf
        from circt import ir

        dbg_debug(
            f"Emitting scf.while with {len(stmt.bindings)} loop-carried variables",
            module="loop_transform"
        )

        # Handle with bindings (loop variables with init/next values)
        init_values = []
        loop_var_types = []

        for binding in stmt.bindings:
            # Convert initial value if provided
            if binding.init:
                init_val = self.converter._convert_expr(binding.init)
                init_values.append(init_val)
                loop_var_types.append(init_val.type)
            else:
                # Default initialization for the type
                var_type = self.converter.convert_cadl_type(binding.ty)
                zero_val = arith.ConstantOp(var_type, 0).result
                init_values.append(zero_val)
                loop_var_types.append(var_type)

        # Add a boolean flag to track first iteration
        bool_type = ir.IntegerType.get_signless(1)
        true_val = arith.ConstantOp(bool_type, 1).result
        init_values.append(true_val)
        loop_var_types.append(bool_type)

        # Create scf.while operation
        while_op = scf.WhileOp(loop_var_types, init_values)

        # Apply directives as attributes (same as emit_scf_for)
        if self.converter.pending_directives:
            from .ast import LitExpr
            for directive in self.converter.pending_directives:
                attr_name = directive.name
                if directive.expr:
                    if isinstance(directive.expr, LitExpr):
                        value = directive.expr.literal.lit.value
                        attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
                        while_op.operation.attributes[attr_name] = attr
                else:
                    while_op.operation.attributes[attr_name] = ir.BoolAttr.get(True)
            # Clear directives after applying to loop (before processing body)
            self.converter.pending_directives = []

        # Before region: check if we should continue
        before_block = while_op.before.blocks.append(*loop_var_types)
        with ir.InsertionPoint(before_block):
            # Get the first_iteration flag (last argument)
            first_iter = before_block.arguments[-1]

            # For do-while, we always continue on first iteration
            # Otherwise, we need to check the condition based on the current values

            # If it's the first iteration, always continue
            # Otherwise, the condition has already been evaluated with the new values
            # So we just pass through the values
            scf.ConditionOp(first_iter, before_block.arguments)

        # After region: execute loop body and check condition
        after_block = while_op.after.blocks.append(*loop_var_types)
        with ir.InsertionPoint(after_block):
            # Push scope for loop body
            self.converter.push_scope()

            # Update loop variables with block arguments (excluding the flag)
            for i, binding in enumerate(stmt.bindings):
                self.converter.set_symbol(binding.id, after_block.arguments[i])

            # Execute loop body - this defines i_, sum_, n_ etc.
            self.converter._convert_stmt_list(stmt.body)

            # Now evaluate the condition using variables defined in the body
            condition_val = self.converter._convert_expr(stmt.condition)

            # Compute next values for loop variables from bindings
            next_values = []
            for binding in stmt.bindings:
                if binding.next:
                    # The next expression references variables defined in the body
                    if isinstance(binding.next, IdentExpr):
                        next_val = self.converter.get_symbol(binding.next.name)
                    else:
                        next_val = self.converter._convert_expr(binding.next)
                    next_values.append(next_val)
                else:
                    # Keep current value if no next expression
                    current_val = self.converter.get_symbol(binding.id)
                    next_values.append(current_val)

            # Pass the condition as the new first_iteration flag
            # This way, the before region will check it on the next iteration
            next_values.append(condition_val)

            # Yield the next values
            scf.YieldOp(next_values)
            self.converter.pop_scope()

        # Make loop variables available in the parent scope (exclude the flag)
        for i, binding in enumerate(stmt.bindings):
            if while_op.results and i < len(while_op.results) - 1:
                self.converter.set_symbol(binding.id, while_op.results[i])
