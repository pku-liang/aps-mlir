"""
Simplified top-level Compiler for MLIR + e-graph optimization.

This module provides a simplified Compiler class that orchestrates
internal rewrites followed by external rewrites.
"""

from __future__ import annotations
import sys
import signal
from contextlib import contextmanager
from megg.egraph.extract import Extractor, MeggCost
from megg.egraph.megg_egraph import MeggEGraph, ExpressionNode
from megg.rewrites import register_internal_rewrites
from megg.egraph.term import Term, id_of
from megg.egraph.terms_to_func import ExprTreeToMLIR
from megg.egraph.func_to_terms import FuncToTerms
from megg.utils import MModule, OperationType, MOperation, get_loop_nest_level, count_nested_loops
from egglog import EGraph
import egglog
from pathlib import Path
from typing import Callable
from typing import Dict, List, Optional, Any, Tuple, Iterable, Set
from dataclasses import dataclass
import time
import json
import logging
from megg.rewrites.internal_rewrites import type_annotation_ruleset
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


@contextmanager
def timeout(seconds: int):
    """
    Context manager for timeout protection.

    Args:
        seconds: Timeout in seconds (0 means no timeout)

    Example:
        with timeout(30):
            # code that might hang
            ...
    """
    if seconds <= 0:
        # No timeout
        yield
        return

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

DEFAULT_EXTERNAL_PASSES: Tuple[str, ...] = ("loop-unroll", "loop-tile")


@dataclass
class FunctionContext:
    """Context information for a function being optimized."""
    func_name: str
    func_op: MOperation
    egraph: EGraph
    transformer: FuncToTerms
    generation: int = 0  # Track transformation generation
    parent_id: Optional[int] = None  # Track where this came from

    def create_child(self, new_func_op: MOperation, new_transformer: FuncToTerms) -> 'FunctionContext':
        """Create a new context from a transformation, preserving history."""
        return FunctionContext(
            func_name=self.func_name,
            func_op=new_func_op,
            egraph=self.egraph,
            transformer=new_transformer,
            generation=self.generation + 1,
            parent_id=id(self),
        )


def _get_var_name(var) -> str:
    """Return a consistent string identifier for egglog variables."""
    name = getattr(var, "name", None)
    if isinstance(name, str):
        return name

    text = str(var)
    if text.startswith('<Var ') and text.endswith('>'):
        return text[5:-1].strip()
    return text


@dataclass
class CompilationResult:
    """Result of the compilation process."""
    optimized_module: MModule
    internal_rewrites: int
    external_rewrites: int
    custom_rewrites: int
    time_elapsed: float
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PhaseSnapshot:
    """Snapshot of compiler state at a specific phase (for debugging/visualization)."""
    phase_name: str
    phase_index: int
    egraph_stats: Dict[str, Any]  # {func_name: {total_eclasses, total_nodes, ...}}
    cumulative_stats: Dict[str, Any]  # {internal_rewrites, external_rewrites, custom_rewrites}
    details: Optional[Dict[str, Any]] = None  # Phase-specific details (rules applied, etc.)
    timestamp: float = 0.0


# Type alias for debug callback
DebugCallback = Callable[['PhaseSnapshot'], None]


class StateManager:
    """Manages compilation state and safeguards."""

    def __init__(self, max_iterations: int = 10, time_limit: float = 60.0):
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.start_time = 0.0
        self.internal_rewrites = 0
        self.external_rewrites = 0
        self.custom_rewrites = 0
        # Detailed statistics for JSON output
        self.internal_rewrites_details: List[Dict[str, Any]] = []
        self.external_rewrites_details: List[Dict[str, Any]] = []
        self.initial_egraph_statistics: Dict[str, Any] = {}
        self.egraph_statistics: Dict[str, Any] = {}

    def start_compilation(self):
        """Start compilation tracking."""
        self.start_time = time.time()
        self.internal_rewrites = 0
        self.external_rewrites = 0
        self.custom_rewrites = 0
        self.internal_rewrites_details = []
        self.external_rewrites_details = []
        self.initial_egraph_statistics = {}
        self.egraph_statistics = {}
        logger.info("Starting compilation")

    def check_time_limit(self) -> bool:
        """Check if time limit exceeded."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_limit:
            logger.warning(f"Time limit ({self.time_limit}s) reached")
            return True
        return False

    def record_internal_rewrites(self, count: int, details: Optional[List[Dict[str, Any]]] = None):
        """Record internal rewrites applied."""
        self.internal_rewrites += count
        if details:
            self.internal_rewrites_details.extend(details)

    def record_external_rewrites(self, count: int, details: Optional[List[Dict[str, Any]]] = None):
        """Record external rewrites applied."""
        self.external_rewrites += count
        if details:
            self.external_rewrites_details.extend(details)

    def record_custom_rewrites(self, count: int):
        """Record custom instruction rewrites applied."""
        self.custom_rewrites += count

    def record_initial_egraph_statistics(self, stats: Dict[str, Any]):
        """Record e-graph statistics at initialization (before any rewrites)."""
        self.initial_egraph_statistics = stats

    def record_egraph_statistics(self, stats: Dict[str, Any]):
        """Record e-graph statistics after saturation."""
        self.egraph_statistics = stats

    def get_result(self, module: MModule, success: bool = True, error_msg: Optional[str] = None) -> CompilationResult:
        """Generate final compilation result."""
        elapsed = time.time() - self.start_time
        return CompilationResult(
            optimized_module=module,
            internal_rewrites=self.internal_rewrites,
            external_rewrites=self.external_rewrites,
            custom_rewrites=self.custom_rewrites,
            time_elapsed=elapsed,
            success=success,
            error_message=error_msg
        )


class RewriteEngine:
    """Handles internal and external rewrite execution."""

    def __init__(self, compiler: Compiler):
        self.compiler = compiler

    def apply_internal_rewrites(self) -> int:
        """Apply internal (egglog-based) algebraic rewrites to expand e-graph."""
        try:
            logger.info("Applying internal rewrites (algebraic laws)...")

            # Apply basic mathematical laws
            from megg.rewrites.internal_rewrites import basic_math_laws, constant_folding_laws

            rule_count = 0
            details = []

            # Basic math laws - expand e-graph with algebraic identities
            math_laws = basic_math_laws()
            const_laws = constant_folding_laws()
            type_rules = type_annotation_ruleset()

            # Run multiple rounds to allow associativity + constant folding
            # e.g., (x*2)*4 => x*(2*4) => x*8 requires 2+ rounds
            for func_name, egraph in self.compiler.egraphs.items():
                for round_idx in range(5):  # 5 rounds should be sufficient
                    # Run each ruleset and try to get statistics
                    try:
                        math_report = egraph.run(math_laws.saturate())
                        const_report = egraph.run(const_laws.saturate())
                        type_report = egraph.run(type_rules.saturate())

                        # Extract statistics from report using num_matches_per_rule
                        for report in [math_report, const_report, type_report]:
                            if hasattr(report, 'num_matches_per_rule'):
                                for rule_name, num_matches in report.num_matches_per_rule.items():
                                    if num_matches > 0:
                                        existing = next((d for d in details if d['rule'] == rule_name), None)
                                        if existing:
                                            existing['num_matches'] += num_matches
                                        else:
                                            details.append({
                                                'rule': rule_name,
                                                'num_matches': num_matches
                                            })
                    except Exception as e:
                        logger.debug(f"Failed to get report statistics: {e}")

            # Try to get rule count
            try:
                rule_count = len(math_laws.__egg_ruleset__.rules) + len(const_laws.__egg_ruleset__.rules)
            except:
                rule_count = 0

            # Record details in state manager
            if self.compiler.state_manager:
                self.compiler.state_manager.record_internal_rewrites(rule_count, details)
            logger.info(f"Applied {len(details)} internal rewrite rules with matches")
            return len(details)

        except Exception as e:
            logger.error(f"Internal rewrites failed: {e}")
            return 0

    def apply_custom_rewrites(self) -> int:
        """Apply custom instruction matching rewrites (final lightweight pass)."""
        try:
            if self.compiler.match_ruleset is None:
                logger.info("No custom instruction rewrites provided")
                return 0

            # Re-apply internal rewrites to simplify any new terms from external passes
            # (e.g., loop unroll generates (i*2)*4 which needs to be simplified to i*8)
            from megg.rewrites.internal_rewrites import basic_math_laws, constant_folding_laws
            math_laws = basic_math_laws()
            const_laws = constant_folding_laws()
            type_rules = type_annotation_ruleset()

            logger.info("Re-simplifying e-graph before custom matching...")
            for func_name, egraph in self.compiler.egraphs.items():
                for round_idx in range(5):
                    egraph.run(math_laws.saturate())
                    egraph.run(const_laws.saturate())
                    egraph.run(type_rules.saturate())

            print(f"Custom instruction ruleset: {self.compiler.match_ruleset}")
            logger.info(
                "Applying custom instruction matching (final pass)...")
            # logger.info(f"Custom ruleset: {self.compiler.match_ruleset.__egg_ruleset__}")
            rule_count = len(self.compiler.match_ruleset.__egg_ruleset__.rules)

            # Apply custom instruction rewrites to the expanded e-graph
            for func_name, egraph in self.compiler.egraphs.items():

                egraph.run(self.compiler.match_ruleset)
                logger.debug(
                    f"Applied custom rewrites to function: {func_name}")

            logger.info(
                f"Applied {rule_count} custom instruction patterns")
            return rule_count

        except Exception as e:
            logger.error(f"Custom instruction matching failed: {e}")
            return 0

    def _get_loop_characteristics(self, loop_op: MOperation) -> Optional[Dict[str, int]]:
        """Collect bounds, trip count, and structural info for a loop."""
        characteristics: Dict[str, int] = {}

        try:
            bounds = loop_op._op.get_scf_loop_bounds()
        except Exception:
            bounds = None

        if not bounds:
            return None

        try:
            lower = int(bounds['lower'])
            upper = int(bounds['upper'])
            step = int(bounds['step'])
        except Exception:
            return None

        if step == 0:
            return None

        trip_count = (upper - lower) // step if step else 0

        characteristics.update({
            'lower': lower,
            'upper': upper,
            'step': step,
            'trip_count': trip_count,
            'nest_level': get_loop_nest_level(loop_op),
            'nested_loop_count': count_nested_loops(loop_op),
        })

        return characteristics

    def _determine_loop_transformations(
        self,
        current_loop: Dict[str, int],
        hint: Dict[str, int],
        loop_index: int = -1,
        hint_name: str = "",
    ) -> List[Tuple[str, str, int]]:
        """Return all viable loop transformations based on the hint."""

        required = {'lower', 'upper', 'step',
                    'nest_level', 'nested_loop_count'}
        if not hint or not required.issubset(hint):
            return []

        custom_step = hint['step']
        if custom_step == 0:
            return []

        custom_trip = (hint['upper'] - hint['lower']) // custom_step
        if custom_trip <= 0:
            return []

        current_trip = current_loop.get('trip_count', 0)
        if current_trip <= 0:
            return []

        current_nested = current_loop.get('nested_loop_count', 0)
        target_nested = hint['nested_loop_count']

        if current_trip == custom_trip and current_nested == target_nested:
            print(f"  [External] loop_{loop_index} (trip={current_trip}, nested={current_nested}) == {hint_name} (trip={custom_trip}, nested={target_nested}) → no change needed")
            return []

        decisions: List[Tuple[str, str, int]] = []

        # Case 1: nested_loop_count increases
        # This can happen due to:
        # - Outer loop unroll: trip decreases, nested increases by same factor
        # - Tiling: trip decreases, introduces new nesting level
        if target_nested > current_nested:
            # Check if this is outer loop unroll (not unroll-jam!)
            # Signature: trip_count reduces by factor, nested_loop_count multiplies by factor
            # Example: trip 4→2 (factor=2), nested 1→2 (×2) = outer unroll factor=2
            if current_nested > 0 and custom_trip > 0 and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                expected_nested = current_nested * factor
                if factor > 1 and expected_nested == target_nested:
                    # This is outer loop unroll (body replicates, so nested loops repeat)
                    decisions.append(('loop-unroll', 'unroll_factor', factor))
                    print(f"  [External] loop_{loop_index} (trip={current_trip}, nested={current_nested}) vs {hint_name} (trip={custom_trip}, nested={target_nested}) → loop-unroll({factor}) [nested increases]")
                    return decisions

            # General tiling case (introduces new nesting level)
            if custom_trip > 0 and custom_trip < current_trip and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                decisions.append(('loop-tile', 'tile_size', factor))
                print(f"  [External] loop_{loop_index} (trip={current_trip}, nested={current_nested}) vs {hint_name} (trip={custom_trip}, nested={target_nested}) → loop-tile({factor}) [add nesting]")
            return decisions

        # Prefer unrolling when the current loop is deeper or the body needs to be widened.
        if target_nested < current_nested:
            if custom_trip > 0 and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                if factor > 1:
                    decisions.append(('loop-unroll', 'unroll_factor', factor))
                    if current_nested > 0:
                        decisions.append(('loop-unroll-jam', 'unroll_factor', factor))
                    print(f"  [External] loop_{loop_index} (trip={current_trip}, nested={current_nested}) vs {hint_name} (trip={custom_trip}, nested={target_nested}) → loop-unroll({factor}) [reduce nested]")
            return decisions

        # Nested loop counts are equal; choose based on trip count difference.
        if current_trip > custom_trip and current_trip % custom_trip == 0:
            factor = current_trip // custom_trip
            if factor > 1:
                decisions.append(('loop-unroll', 'unroll_factor', factor))
                if current_nested > 0:
                    decisions.append(('loop-unroll-jam', 'unroll_factor', factor))
                print(f"  [External] loop_{loop_index} (trip={current_trip}, nested={current_nested}) vs {hint_name} (trip={custom_trip}, nested={target_nested}) → loop-unroll({factor}) [trip {current_trip}→{custom_trip}]")
            if custom_trip > 0 and custom_trip <= current_trip and current_nested > 0:
                decisions.append(('loop-tile', 'tile_size', factor))
        elif current_trip < custom_trip:
            # Cannot increase iterations via tiling/unrolling; no suitable transform.
            print(f"  [External] loop_{loop_index} (trip={current_trip}) < {hint_name} (trip={custom_trip}) → no transform available")
            return []

        return decisions

    def _try_apply_external_pass(
        self,
        *,
        loop_index: int,
        decision: Tuple[str, str, int],
        func_context: FunctionContext,
        apply_loop_pass_by_name: Callable,
    ) -> Tuple[bool, Optional[MOperation], Optional[FuncToTerms]]:
        pass_name, param_name, param_value = decision
        if param_value <= 0:
            return False, None, None

        result = apply_loop_pass_by_name(
            func=func_context.func_op,
            loop_index=loop_index,
            pass_name=pass_name,
            func_to_terms=func_context.transformer,
            egraph=func_context.egraph,
            **{param_name: param_value}
        )

        # Handle None result (failure)
        if result is None:
            return False, None, None

        # Unpack result tuple: (dict, temp_egraph)
        result_dict, temp_egraph = result
        logger.info(f"  Result dict: {result_dict}")
        logger.info(f"  Success: {result_dict.get('success') if result_dict else None}")

        if not result_dict or not result_dict.get("success"):
            logger.warning(f"  Loop pass returned failure or None: result_dict={result_dict}")
            return False, None, None

        # Get the new function from result
        new_func_op = result_dict.get("function")
        if not new_func_op:
            logger.error("No function in apply_loop_pass result")
            return False, None, None

        logger.info(f"  Got new function: {new_func_op.symbol_name if hasattr(new_func_op, 'symbol_name') else 'unknown'}")

        # Re-transform the new function to get updated transformer
        # Add timeout protection (30 seconds) to prevent hanging on complex functions
        try:
            with timeout(30):
                new_transformer = FuncToTerms.transform(new_func_op, func_context.egraph)
            return True, new_func_op, new_transformer
        except TimeoutException:
            logger.error(f"FuncToTerms.transform timed out after 30s (function too complex)")
            return False, None, None
        except Exception as e:
            logger.error(f"Failed to re-transform function after pass: {e}")
            return False, None, None

    def apply_external_rewrites(self) -> int:
        try:
            from megg.rewrites.external_loop_pass import apply_loop_pass_by_name
            pass_count = 0
            external_details = []  # Track external pass details

            all_contexts: List[FunctionContext] = []
            contexts_by_func: Dict[str, List[FunctionContext]] = {}

            for func_name in list(self.compiler.transformers.keys()):
                egraph = self.compiler.egraphs.get(func_name)
                transformer = self.compiler.transformers.get(func_name)

                # Find original function operation (as template)
                original_func_op = None
                for op in self.compiler.original_module.get_functions():
                    if getattr(op, 'symbol_name', None) == func_name:
                        original_func_op = op
                        break

                # Skip if any component is missing
                if egraph is None or transformer is None or original_func_op is None:
                    continue

                # OPTIMIZATION: Check if we need loop passes before doing expensive affine extraction
                # Only extract affine-friendly version if there are loops that need transformation
                loops = transformer.loop_to_term
                needs_transformation = False

                if loops and self.compiler.loop_hints:
                    # Print loop hints from ISAX pattern
                    print(f"[Loop Hints] From ISAX pattern:")
                    for hint_name, hint_info in self.compiler.loop_hints.items():
                        hint_trip = (hint_info['upper'] - hint_info['lower']) // hint_info['step'] if hint_info.get('step', 0) != 0 else 0
                        print(f"  {hint_name}: trip={hint_trip}, nested={hint_info.get('nested_loop_count', 0)}, level={hint_info.get('nest_level', -1)}")

                    # Quick check: do any loops need transformation?
                    for loop_index, (loop_op, _) in list(loops.items()):
                        current_info = self._get_loop_characteristics(loop_op)
                        if not current_info:
                            continue

                        current_nest_level = current_info.get('nest_level', -1)
                        matching_hints = {
                            name: info for name, info in self.compiler.loop_hints.items()
                            if info.get('nest_level', -1) == current_nest_level
                        }

                        # Check if any hint would result in actual transformations
                        for ci_loop_name, ci_loop_info in matching_hints.items():
                            decisions = self._determine_loop_transformations(
                                current_info,
                                ci_loop_info,
                                loop_index=loop_index,
                                hint_name=ci_loop_name,
                            )
                            if decisions:  # Non-empty means transformation needed
                                needs_transformation = True
                                logger.info(f"Loop {loop_index} needs transformation based on hint '{ci_loop_name}': {decisions}")
                                break

                        if needs_transformation:
                            break

                # Only do affine extraction if we found loops that need transformation
                if needs_transformation:
                    # Extract affine-friendly version from e-graph for loop transformations
                    # This converts shl->mul, etc. to make patterns more recognizable
                    try:
                        from megg.egraph.megg_egraph import MeggEGraph
                        from megg.egraph.extract import Extractor, AffineCost
                        from megg.egraph.terms_to_func import ExprTreeToMLIR

                        logger.info(f"Extracting affine-friendly version of {func_name} for loop passes...")

                        # Convert egglog.EGraph to MeggEGraph
                        megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer=transformer)

                        # Extract using AffineCost (prefers mul over shl, etc.)
                        extractor = Extractor(megg_egraph, AffineCost())
                        extraction_result = extractor.find_best()

                        logger.info(f"✓ Extracted with cost: {extraction_result.cost}")

                        # Convert back to MLIR
                        func_op = ExprTreeToMLIR.reconstruct(
                            original_func=original_func_op,
                            body_exprs=[extraction_result.expr],
                            target_module=self.compiler.original_module
                        )

                        logger.info(f"✓ Reconstructed affine-friendly MLIR for {func_name}")

                        # CRITICAL: Create a standalone module for the affine-extracted function
                        # This ensures it has a proper parent module for cloning during loop passes
                        try:
                            import os
                            from datetime import datetime
                            from megg.utils import MModule

                            # Create standalone module
                            standalone_module = MModule()
                            standalone_module.append_to_module(func_op)

                            # Get the function back (now it has proper parent)
                            funcs = standalone_module.get_functions()
                            if funcs:
                                func_op = funcs[0]
                                logger.info(f"✓ Created standalone module for affine-extracted function")
                            else:
                                raise Exception("Failed to get function from standalone module")

                            # Save to tmp directory for debugging
                            from megg.utils import get_temp_dir
                            tmp_dir = get_temp_dir()

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
                            filename = f"affine_extracted_{func_name}_{timestamp}.mlir"
                            filepath = tmp_dir / filename

                            with open(filepath, 'w') as f:
                                f.write(str(standalone_module))

                            logger.info(f"📁 Saved affine-extracted MLIR: {filename}")

                        except Exception as module_error:
                            logger.error(f"Failed to create standalone module: {module_error}")
                            raise

                        # Create a fresh transformer for the affine-extracted function IN THE ORIGINAL EGRAPH
                        # This ensures all loop terms are registered in the original egraph
                        logger.info(f"Creating transformer for affine-extracted {func_name} in original egraph")
                        new_transformer = FuncToTerms.transform(func_op, egraph)  # Use original egraph!
                        logger.info(f"✓ Transformer created with {len(new_transformer.loop_to_term)} loops")

                        # Use the new transformer for loop analysis
                        transformer = new_transformer

                    except Exception as e:
                        logger.warning(f"Failed to extract affine version for {func_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        logger.warning("Falling back to original function")
                        func_op = original_func_op
                        # Keep using the old transformer
                else:
                    # No transformation needed - skip affine extraction, use original function
                    logger.info(f"Skipping affine extraction for {func_name} (no transformations needed)")
                    func_op = original_func_op

                # Create initial context (generation 0)
                initial_ctx = FunctionContext(
                    func_name=func_name,
                    func_op=func_op,  # Use affine-extracted function only if needed
                    egraph=egraph,
                    transformer=transformer,  # Use transformer with loop terms in original egraph
                    generation=0,
                    parent_id=None,
                )

                all_contexts.append(initial_ctx)
                contexts_by_func[func_name] = [initial_ctx]

            # Process contexts using index-based iteration
            # This allows us to dynamically add new contexts during iteration
            ctx_index = 0
            max_contexts = 50  # Limit total contexts to prevent runaway
            max_generation = 0  # Only process generation 0 (initial affine-extracted version)
                                # Skip generation 1+ to avoid re-transforming complex loop-optimized functions

            while ctx_index < len(all_contexts) and ctx_index < max_contexts:
                ctx = all_contexts[ctx_index]

                logger.info(f"Processing {ctx.func_name} generation {ctx.generation} "
                           f"(context {ctx_index + 1}/{len(all_contexts)})")

                # Skip contexts beyond max_generation to avoid complexity explosion
                if ctx.generation > max_generation:
                    logger.info(f"Skipping generation {ctx.generation} (max={max_generation})")
                    ctx_index += 1
                    continue

                # Get loops from current context
                loops = ctx.transformer.loop_to_term
                logger.info(f"Context has {len(loops)} loop(s) in loop_to_term")
                if not loops:
                    ctx_index += 1
                    continue

                # Add timeout protection for processing each context (60 seconds)
                try:
                    with timeout(60):
                        # Process each loop
                        for loop_index, (loop_op, _) in list(loops.items()):
                            logger.info(f"Processing loop {loop_index}")
                            current_info = self._get_loop_characteristics(loop_op)
                            if not current_info:
                                logger.info(f"  No characteristics for loop {loop_index}")
                                continue

                            logger.info(f"  Loop {loop_index} info: {current_info}")

                            # Filter loop hints by nest_level to avoid mismatches
                            # (e.g., outer loop matching inner hint, or vice versa)
                            current_nest_level = current_info.get('nest_level', -1)
                            logger.info(f"  Current nest_level: {current_nest_level}, available hints: {list(self.compiler.loop_hints.keys())}")

                            matching_hints = {
                                name: info for name, info in self.compiler.loop_hints.items()
                                if info.get('nest_level', -1) == current_nest_level
                            }
                            logger.info(f"  Matching hints for nest_level={current_nest_level}: {list(matching_hints.keys())}")

                            if not matching_hints:
                                logger.debug(f"No matching hints for loop at nest_level={current_nest_level}")
                                continue

                            # Try to match against loop hints (filtered by nest_level)
                            for ci_loop_name, ci_loop_info in matching_hints.items():
                                logger.info(f"  Trying to match with hint '{ci_loop_name}': {ci_loop_info}")
                                decisions = self._determine_loop_transformations(
                                    current_info,
                                    ci_loop_info,
                                    loop_index=loop_index,
                                    hint_name=ci_loop_name,
                                )

                                logger.info(f"  Decisions for '{ci_loop_name}': {decisions}")

                                if not decisions:
                                    continue

                                # Try to apply each decision
                                for decision in decisions:
                                    logger.info(f"  Applying decision: {decision}")
                                    success, new_func_op, new_transformer = self._try_apply_external_pass(
                                        loop_index=loop_index,
                                        decision=decision,
                                        func_context=ctx,
                                        apply_loop_pass_by_name=apply_loop_pass_by_name,
                                    )

                                    logger.info(f"  Apply result: success={success}")

                                    if not success:
                                        logger.info(f"  Skipping due to failure")
                                        continue

                                    logger.info(f"  Creating new context for generation {ctx.generation + 1}")

                                    # Create NEW context from transformation (preserve original)
                                    new_ctx = ctx.create_child(new_func_op, new_transformer)

                                    # Add new context to work list
                                    all_contexts.append(new_ctx)
                                    contexts_by_func[ctx.func_name].append(new_ctx)

                                    # Update the compiler's state to latest version
                                    self.compiler.transformers[ctx.func_name] = new_transformer

                                    pass_count += 1
                                    # Record external pass details
                                    pass_name, param_name, param_value = decision
                                    external_details.append({
                                        'pass': pass_name,
                                        'parameter': param_name,
                                        'value': param_value
                                    })
                                    logger.info(f"  Incremented pass_count to {pass_count}")

                                    logger.info(f"✓ Applied transformation, created generation {new_ctx.generation}")

                except TimeoutException:
                    logger.warning(f"Processing context {ctx_index + 1} timed out after 60s, skipping...")

                # Move to next context
                ctx_index += 1

            # Check if we hit the limits
            if len(all_contexts) >= max_contexts:
                logger.warning(f"Reached max_contexts limit ({max_contexts}), stopped processing")

            # Log summary
            for func_name, contexts in contexts_by_func.items():
                logger.info(f"Function {func_name}: {len(contexts)} versions "
                           f"(generations 0-{contexts[-1].generation if contexts else 0})")

            # Record external pass details in state manager
            if self.compiler.state_manager:
                self.compiler.state_manager.record_external_rewrites(pass_count, external_details)

            logger.info(f"Applied {pass_count} external pass instances")
            return pass_count

        except Exception as e:
            logger.error(f"External rewrites failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _find_loops_in_function(self, func_op: MOperation) -> List[tuple]:
        """Find all named loops in a function.

        Returns:
            List of (loop_name, loop_op) tuples
        """
        from megg.utils import OperationType

        loops = []  # List of (name, op) tuples
        loop_counter = 0

        def find_loops_recursively(op: MOperation):
            nonlocal loop_counter

            if hasattr(op, 'type'):
                if op.type in [OperationType.SCF_FOR, OperationType.AFFINE_FOR, OperationType.SCF_WHILE]:
                    loops.append((len(loops), op))
            # Recursively search nested regions
            for region in op.get_regions():
                for block in region.get_blocks():
                    for nested_op in block.get_operations():
                        find_loops_recursively(nested_op)

        regions = func_op.get_regions()
        if regions:
            for region in regions:
                for block in region.get_blocks():
                    for op in block.get_operations():
                        find_loops_recursively(op)

        return loops


class Compiler:
    """
    Simplified top-level compiler for MLIR + e-graph optimization.

    Executes internal rewrites first, then external rewrites.
    """

    def __init__(self, module: MModule, target_functions: Optional[List[str]] = None,
                 cost_function: Optional[Any] = None,
                 match_ruleset: Optional[egglog.Ruleset] = None,
                 skeletons: Optional[List] = None,
                 loop_hints: Optional[Dict[str, Dict[str, int]]] = None,
                 instr_encodings: Optional[Dict[str, Dict[str, str]]] = None,
                 debug_callback: Optional[DebugCallback] = None):
        """
        Initialize compiler with MLIR module.

        Args:
            module: Input MLIR module or MLIR text string
            target_functions: List of function names to optimize (None = all functions)
            cost_function: Custom cost function for extraction (default: AstSize)
                          Can be AstSize(), AstDepth(), OpWeightedCost(...), or custom
            match_ruleset: egglog ruleset for pattern tree matching
            skeletons: List of Skeleton for complex instruction matching
            loop_hints: Dictionary mapping loop names to optimization hints (unroll_factor, tile_size)
            instr_encodings: Dictionary mapping instruction names to encodings (opcode/funct3/funct7)
            debug_callback: Optional callback function called after each phase with PhaseSnapshot
        """
        self.original_module = module
        self.target_functions = target_functions or []
        self.cost_function = cost_function if cost_function is not None else MeggCost()
        self.loop_hints = loop_hints or {}  # Store loop optimization hints
        self.instr_encodings = instr_encodings or {}  # Store instruction encodings
        self.debug_callback = debug_callback  # Callback for phase snapshots
        self._phase_index = 0  # Track phase number for snapshots

        # Core components
        self.egraphs: Dict[str, EGraph] = {}
        self.transformers: Dict[str, FuncToTerms] = {}
        # Store MeggEGraphs with custom_instr nodes
        self.modified_megg_egraphs: Dict[str, MeggEGraph] = {}
        self.loop_transform_records: Dict[str, List[Dict[str, Any]]] = {}
        self.latest_func_ops: Dict[str, MOperation] = {}
        self._external_pass_modules: Dict[str, List[MModule]] = {}
        self.rewrite_engine = RewriteEngine(self)
        self.state_manager: Optional[StateManager] = None
        self.match_ruleset = match_ruleset
        self.skeletons = skeletons or []  # 新增：控制流骨架列表
        self.custom_instr_properties: Dict[str, Dict[str, Any]] = {}
        self._init_custom_instr_properties()
        # Convert MLIR functions to e-graph (if MLIR is available)
        self._initialize_egraph()

    def _init_custom_instr_properties(self):
        """Record side-effect metadata and encodings for known custom instructions."""
        self.custom_instr_properties = {}

        # First, process skeletons (complex control flow patterns)
        for skeleton in self.skeletons:
            props = {
                'has_side_effects': getattr(skeleton, 'has_side_effects', False),
                'clobbers': list(getattr(skeleton, 'clobbers', []) or []),
            }
            # Add encoding information if available
            if skeleton.instr_name in self.instr_encodings:
                props['encoding'] = self.instr_encodings[skeleton.instr_name]

            self.custom_instr_properties[skeleton.instr_name] = props

        # Then, ensure all instructions with encodings are registered
        # This handles simple computation patterns (egglog rewrites) that don't have skeletons
        for instr_name, encoding in self.instr_encodings.items():
            if instr_name not in self.custom_instr_properties:
                # Create default properties for simple computation patterns
                self.custom_instr_properties[instr_name] = {
                    'has_side_effects': False,  # Pure computation
                    'clobbers': [],
                    'encoding': encoding
                }

    def _initialize_egraph(self):
        """Convert MLIR functions to e-graph terms."""
        logger.info("Converting MLIR functions to e-graph terms")

        try:
            for op in self.original_module.get_functions():
                func_name = op.symbol_name
                logger.info(
                    f"Processing function: {func_name} in targets {self.target_functions}")
                if self.target_functions and func_name not in self.target_functions:
                    continue

                # Skip functions with unsupported operations (calls)
                if self.original_module.has_unsupported_ops(op):
                    logger.warning(
                        f"Skipping function {func_name} (unsupported operations)")
                    continue
                logger.info(f"Converting function: {func_name}")
                # Transform function to e-graph terms
                egraph = EGraph()

                transformer = FuncToTerms.transform(op, egraph)
                logger.info(f"FuncToTerms transformer: {transformer}")
                # use type rules to register type equality in egraph
                type_rules = type_annotation_ruleset()
                egraph.run(type_rules.saturate())
                # add egraph and transformer to compiler
                self.egraphs[func_name] = egraph
                self.transformers[func_name] = transformer
                self.latest_func_ops[func_name] = op
                self._external_pass_modules.setdefault(
                    func_name, []).append(self.original_module)

                logger.info(f"Processed function: {func_name}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Failed to initialize e-graph: {e}")
            logger.error(f"Traceback:\n{error_details}")
            print(f"ERROR: Failed to initialize e-graph: {e}", file=sys.stderr)
            print(f"Traceback:\n{error_details}", file=sys.stderr)

    def _emit_phase_snapshot(self, phase_name: str, details: Optional[Dict[str, Any]] = None):
        """Emit a snapshot of current compiler state to the debug callback."""
        if self.debug_callback is None:
            return

        try:
            # Collect e-graph statistics for each function
            egraph_stats = {}
            for func_name, egraph in self.egraphs.items():
                transformer = self.transformers.get(func_name)
                if transformer:
                    try:
                        megg = MeggEGraph.from_egraph(egraph, func_transformer=transformer)
                        egraph_stats[func_name] = megg.get_statistics()
                    except Exception as e:
                        egraph_stats[func_name] = {"error": str(e)}

            # Collect cumulative stats from state_manager
            cumulative_stats = {}
            if self.state_manager:
                cumulative_stats = {
                    "internal_rewrites": self.state_manager.internal_rewrites,
                    "external_rewrites": self.state_manager.external_rewrites,
                    "custom_rewrites": self.state_manager.custom_rewrites,
                    "internal_details": self.state_manager.internal_rewrites_details,
                    "external_details": self.state_manager.external_rewrites_details,
                }

            # Create snapshot
            snapshot = PhaseSnapshot(
                phase_name=phase_name,
                phase_index=self._phase_index,
                egraph_stats=egraph_stats,
                cumulative_stats=cumulative_stats,
                details=details,
                timestamp=time.time()
            )

            self._phase_index += 1

            # Call the callback
            self.debug_callback(snapshot)

        except Exception as e:
            logger.warning(f"Failed to emit phase snapshot: {e}")

    def schedule(self,
                 max_iterations: int = 10,
                 time_limit: float = 60.0,
                 internal_rewrites: bool = True,
                 external_passes: bool = True,
                 custom_rewrites: bool = True,
                 enable_safeguards: bool = True,
                 output_path: Optional[str] = None) -> CompilationResult:
        """
        Run optimization pipeline: expand e-graph, then match custom instructions.

        Pipeline stages:
        1. Internal rewrites: Algebraic laws and constant folding to expand e-graph
        2. External rewrites: MLIR passes to further expand e-graph
        3. Custom instruction matching: Final lightweight pass to match patterns

        Args:
            max_iterations: Maximum iterations (currently not used in simplified version)
            time_limit: Maximum time in seconds
            internal_rewrites: Enable internal algebraic rewrites (expand e-graph)
            external_passes: List of MLIR pass names (expand e-graph further)
            custom_rewrites: Enable custom instruction pattern matching (final pass)
            enable_safeguards: Enable time limits and error handling

        Returns:
            CompilationResult with optimized module and statistics
        """
        state_manager = StateManager(max_iterations, time_limit)
        self.state_manager = state_manager
        state_manager.start_compilation()

        try:
            logger.info("Starting optimization schedule")
            logger.info(
                "Goal: Expand e-graph with rewrites, then match custom instructions")

            # Collect initial e-graph statistics (before any rewrites)
            self._collect_initial_egraph_statistics()

            # Emit initial snapshot (Phase 0: After E-Graph Construction)
            self._emit_phase_snapshot("0_egraph_init", {"description": "E-Graph initialized from MLIR"})

            # Phase 1: Internal rewrites - expand e-graph with algebraic identities
            if internal_rewrites:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self.original_module, False, "Time limit exceeded during internal rewrites")
                from megg.utils import get_temp_dir
                tmp_dir = get_temp_dir()
                self.visualize_egraph(str(tmp_dir / "egraph_before_internal.svg"))
                internal_count = self.rewrite_engine.apply_internal_rewrites()
                state_manager.record_internal_rewrites(internal_count)
                self.visualize_egraph(str(tmp_dir / "egraph_after_internal.svg"))
                logger.info(
                    f"Phase 1 complete: {internal_count} internal rules applied (e-graph expanded)")

                # Emit snapshot after internal rewrites
                self._emit_phase_snapshot("1_internal_rewrites", {
                    "description": "Applied algebraic laws (commutativity, associativity, constant folding)",
                    "rules_applied": internal_count
                })

            # Phase 2: External rewrites - further expand e-graph with MLIR passes
            if external_passes:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during external rewrites")

                # Note: record_external_rewrites is called inside apply_external_rewrites
                # with details, so we don't call it again here to avoid double counting
                external_count = self.rewrite_engine.apply_external_rewrites()
                logger.info(
                    f"Phase 2 complete: {external_count} external passes applied (e-graph expanded)")

                # Emit snapshot after external rewrites
                self._emit_phase_snapshot("2_external_rewrites", {
                    "description": "Applied MLIR loop passes (unroll, tile)",
                    "passes_applied": external_count
                })

                # Phase 2.5: Re-apply internal rewrites to simplify expressions introduced by external passes
                # This is critical for pattern matching: loop transformations may introduce complex
                # arithmetic expressions (e.g., (x*2)<<2) that need to be simplified (e.g., x*4)
                if external_count > 0:
                    logger.info("Phase 2.5: Re-applying internal rewrites to simplify transformed code...")
                    additional_internal = self.rewrite_engine.apply_internal_rewrites()
                    state_manager.record_internal_rewrites(additional_internal)
                    logger.info(f"Phase 2.5 complete: {additional_internal} additional internal rules applied")

                    # Emit snapshot after re-simplification
                    self._emit_phase_snapshot("2.5_re_simplify", {
                        "description": "Re-applied internal rewrites after loop transforms",
                        "additional_rules": additional_internal
                    })

            # Phase 3: Custom instruction matching - final lightweight pass
            if custom_rewrites and self.match_ruleset is not None:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during custom rewrites")

                from megg.utils import get_temp_dir
                tmp_dir = get_temp_dir()
                self.visualize_egraph(str(tmp_dir / "egraph_before_custom.svg"))
                custom_count = self.rewrite_engine.apply_custom_rewrites()
                state_manager.record_custom_rewrites(custom_count)
                self.visualize_egraph(str(tmp_dir / "egraph_after_custom.svg"))
                logger.info(
                    f"Phase 3 complete: {custom_count} custom instruction patterns applied (final pass)")

                # Emit snapshot after custom rewrites
                self._emit_phase_snapshot("3_custom_rewrites", {
                    "description": "Applied egglog pattern matching rules",
                    "patterns_matched": custom_count
                })

            # Phase 3.5: Skeleton matching - complex instruction matching (新增)
            if custom_rewrites and self.skeletons:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during skeleton matching")

                skeleton_count = self._apply_skeleton_matching()
                state_manager.record_custom_rewrites(skeleton_count)
                logger.info(
                    f"Phase 3.5 complete: {skeleton_count} complex instructions matched via skeletons")

                # Emit snapshot after skeleton matching
                self._emit_phase_snapshot("3.5_skeleton_matching", {
                    "description": "Applied skeleton-based control flow matching",
                    "skeletons_matched": skeleton_count,
                    "skeleton_names": [s.instr_name for s in self.skeletons]
                })

            # Collect e-graph statistics after saturation (before extraction)
            self._collect_egraph_statistics()

            # Emit final snapshot before extraction
            self._emit_phase_snapshot("4_before_extraction", {
                "description": "E-Graph saturated, ready for extraction"
            })

            # Phase 4: Extract optimized module
            # WORKAROUND: Pass output_path to save immediately and avoid nanobind destructor crash
            optimized_module = self._extract_optimized_module(
                output_path=output_path)

            result = state_manager.get_result(optimized_module, True)
            logger.info(
                f"Optimization successful: {result.internal_rewrites} internal + {result.external_rewrites} external + {result.custom_rewrites} custom in {result.time_elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return state_manager.get_result(self.original_module, False, str(e))

    def _extract_optimized_module(self, output_path: Optional[str] = None) -> MModule:
        """Extract optimized MLIR module from e-graph using MeggEGraph.

        Args:
            output_path: If provided, save the module immediately to avoid nanobind crash
        """

        try:
            # Create a fresh module by parsing an empty module to ensure dialects are loaded
            from megg.utils import MModule, IRBuilder
            optimized_module = MModule("module {}")
            logger.debug(
                f"Created empty module, is generic: {('\"builtin.module\"()' in optimized_module.to_string())}")
            builder = IRBuilder()
            with builder.set_context(optimized_module.get_context()):
                if (
                    self.state_manager is not None
                    and self.state_manager.internal_rewrites == 0
                    and self.state_manager.external_rewrites == 0
                    and self.state_manager.custom_rewrites == 0
                    and not self.modified_megg_egraphs
                ):
                    logger.info(
                        "No rewrites applied; returning original module")
                    return self.original_module
                try:
                    for op in self.original_module.get_operations():
                        # 如果是 memref.global，直接加入 optimized_module
                        if op.type == OperationType.MEMREF_GLOBAL:
                            # Get the underlying MLIR operation
                            raw_op = op._op if hasattr(op, '_op') else op
                            optimized_module.append_to_module(raw_op)
                            logger.info(f"Preserved global memref: {op.name}")
                except Exception as e:
                    logger.warning(f"Failed to preserve global memrefs: {e}")

                # Reconstruct each function separately
                for func_name, transformer in self.transformers.items():
                    print(
                        f"Reconstructing function: {func_name}, top_block: {transformer.top_block}")
                    # Create MeggEGraph for this specific function
                    try:
                        # Check if we have a modified MeggEGraph (with custom_instr nodes)
                        if func_name in self.modified_megg_egraphs:
                            megg_egraph = self.modified_megg_egraphs[func_name]
                            logger.info(
                                f"Using modified MeggEGraph with custom instructions for {func_name}")
                            logger.info(
                                f"MeggEGraph statistics: {megg_egraph.get_statistics()}")
                        else:
                            # Create fresh MeggEGraph from egraph
                            egraph = self.egraphs[func_name]
                            megg_egraph = MeggEGraph.from_egraph(
                                egraph,
                                func_transformer=transformer
                            )
                            logger.info(
                                f"MeggEGraph for function {func_name}: {megg_egraph.get_statistics()}")
                        logger.info(
                            f"Root eclasses: {megg_egraph.root_eclasses}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to create MeggEGraph for {func_name}: {e}")
                        import traceback
                        logger.debug(
                            f"MeggEGraph creation traceback: {traceback.format_exc()}")
                        megg_egraph = None
                    try:
                        # Get original function from original_module (not transformer.func)
                        # transformer.func may point to a temporary function from external rewrites
                        original_func = None
                        for op in self.original_module.get_functions():
                            if getattr(op, 'symbol_name', None) == func_name:
                                original_func = op
                                break

                        if original_func is None:
                            logger.error(f"Could not find original function {func_name} in original_module")
                            continue

                        if megg_egraph:
                            # New MeggEGraph-based extraction path with custom cost function
                            logger.info(
                                f"Using MeggEGraph extraction for {func_name} with cost function {type(self.cost_function).__name__}")

                            # Create extractor with custom cost function
                            try:
                                # print(f"megg_egraph: {megg_egraph}")
                                extractor = Extractor(
                                    megg_egraph, self.cost_function)
                                logger.info(
                                    f"Extractor created, costs computed for {len(extractor.costs)} eclasses")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to create extractor: {e}")
                                extractor = None

                            # Extract expression trees for outputs
                            output_exprs = []
                            if extractor and megg_egraph.root_eclasses:
                                # Extract from root eclasses (one per output_term)
                                root_eclasses_to_extract = megg_egraph.root_eclasses
                                # print(megg_egraph)
                                for i, eclass_id in enumerate(root_eclasses_to_extract):
                                    try:
                                        # Extract using the custom cost function
                                        result = extractor.find_best(eclass_id)
                                        print(
                                            f"Extracted expression for output {i}: {result.expr} with cost {result.cost}")
                                        output_exprs.append(result.expr)
                                        logger.debug(
                                            f"Extracted expression from {eclass_id} with cost {result.cost}")
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to extract expression for eclass {eclass_id}: {e}")

                            # Determine which expressions describe the full function body.
                            body_exprs = list(output_exprs)
                            top_block_expr = None
                            top_block_term = getattr(
                                transformer, 'top_block', None)
                            if extractor and top_block_term is not None:
                                try:
                                    top_block_id = transformer.get_id_of_term(
                                        top_block_term)
                                except Exception:
                                    top_block_id = None

                                if top_block_id is not None:
                                    term_to_eclass = getattr(
                                        megg_egraph, 'term_id_to_eclass', {})
                                    top_block_eclass = term_to_eclass.get(
                                        top_block_id)
                                    if top_block_eclass:
                                        try:
                                            top_block_result = extractor.find_best(
                                                top_block_eclass)
                                            top_block_expr = top_block_result.expr
                                            logger.info(
                                                f"Extracted top-level block from eclass {top_block_eclass} with cost {top_block_result.cost} (op={top_block_expr.op})")
                                            child_ops = [
                                                child.op for child in top_block_expr.children]
                                            logger.info(
                                                f"Top-level block children ops: {child_ops}")
                                            if top_block_expr.children and top_block_expr.children[0].op == 'Vec':
                                                vec_children = top_block_expr.children[0].children
                                                vec_ops = [
                                                    child.op for child in vec_children]
                                                logger.info(
                                                    f"Top-level block Vec entries: {vec_ops}")
                                                for idx, child in enumerate(vec_children):
                                                    logger.info(
                                                        f"  Vec[{idx}] op={child.op} children={len(child.children)} metadata={child.metadata}")
                                                    if child.op in {'For', 'For_'}:
                                                        for jdx, grand in enumerate(child.children):
                                                            logger.info(
                                                                f"    For child[{jdx}] op={grand.op} metadata={grand.metadata}")
                                                            if grand.op == 'Block':
                                                                logger.info(
                                                                    f"      Block metadata: {grand.metadata}, children={len(grand.children)}")
                                        except Exception as e:
                                            logger.warning(
                                                f"Failed to extract top-level block expression: {e}")

                            if top_block_expr is not None:
                                body_exprs = [top_block_expr]

                            # Reconstruct MLIR from expression trees. output_exprs still
                            # represent the values returned from the function, while
                            # body_exprs now starts from the top-level Block when
                            # available so side-effecting ops are preserved.
                            if body_exprs:
                                logger.info(
                                    f"Reconstructing MLIR function {func_name} from {len(body_exprs)} body expression(s)")

                                # Add timeout protection (60 seconds) for reconstruction
                                try:
                                    with timeout(60):
                                        optimized_func = ExprTreeToMLIR.reconstruct(
                                            original_func,
                                            body_exprs=body_exprs,
                                            output_terms=output_exprs or body_exprs,
                                            instr_properties=self.custom_instr_properties,
                                            target_module=optimized_module
                                        )
                                    if optimized_func:
                                        optimized_module.append_to_module(
                                            optimized_func)
                                        logger.info(
                                            f"Reconstructed function {func_name} via MeggEGraph")
                                        logger.debug(
                                            f"After appending {func_name}, is generic: {'\"builtin.module\"()' in optimized_module.to_string()}")
                                    else:
                                        # Reconstruction returned None, use original
                                        optimized_module.append_to_module(original_func)
                                        logger.warning(f"Reconstruction returned None for {func_name}, using original")
                                except TimeoutException:
                                    logger.error(f"ExprTreeToMLIR.reconstruct timed out after 60s for {func_name}")
                                    # Fallback to original function
                                    optimized_module.append_to_module(original_func)
                                    logger.info(f"Using original function for {func_name} due to timeout")
                            else:
                                # Fallback: use original function
                                optimized_module.append_to_module(
                                    original_func)
                                logger.warning(
                                    f"No expression trees extracted for {func_name}, using original")
                        else:
                            # Fallback: use original function if MeggEGraph not available
                            optimized_module.append_to_module(original_func)
                            logger.info(
                                f"Using original function for {func_name}")

                    except Exception as e:
                        import traceback
                        logger.error(
                            f"Failed to reconstruct function {func_name}: {e}")
                        logger.error(f"Traceback:\n{traceback.format_exc()}")

                # Get the optimized module
                # NOTE: Printing module here can trigger nanobind destructor error
                # print(f"Optimized module:\n{optimized_module}")
                logger.info("Successfully extracted optimized module")

                # WORKAROUND: Save immediately before return to avoid nanobind destructor crash
                if output_path:
                    logger.info(
                        f"Saving optimized MLIR to {output_path} immediately...")
                    try:
                        # Ensure parent directory exists
                        import os
                        os.makedirs(os.path.dirname(output_path)
                                    or '.', exist_ok=True)
                        MModule.save_mlir(optimized_module, output_path)
                        logger.info(f"Successfully saved to {output_path}")
                    except Exception as e:
                        logger.error(f"Failed to save MLIR: {e}")

                # Store in member variable to keep alive
                self._optimized_module_cache = optimized_module

                # Return the optimized module
                return optimized_module

        except Exception as e:
            logger.error(f"Failed to extract optimized module: {e}")
            return self.original_module

    # Visualization and debugging APIs
    def visualize_egraph(self, output_path: str, format: str = "svg", **kwargs):
        """Generate e-graph visualization."""
        # just visualize the first egraph for simplicity
        egraph = next(iter(self.egraphs.values()), None)
        if egraph is None:
            logger.warning("No e-graph available for visualization")
            return None
        src = egraph._graphviz(**kwargs)
        src.format = format

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = src.render(filename=str(
            path.with_suffix('')), cleanup=True, quiet=True)
        logger.info(f"EGraph visualization saved to: {result}")
        return result

    def _apply_skeleton_matching(self) -> int:
        """
        应用控制流骨架匹配

        策略：
        1. 将egglog EGraph转换为MeggEGraph
        2. 使用SkeletonMatcher在MeggEGraph中进行局部搜索
        3. 对匹配到的结果，在原egraph中添加custom_instr并union

        Returns:
            匹配到的复杂指令数量
        """
        from megg.egraph.megg_egraph import MeggEGraph, SkeletonMatcher

        match_count = 0

        for func_name, egraph in self.egraphs.items():
            logger.info(
                f"Applying skeleton matching for function: {func_name}")

            # 将egglog EGraph序列化为MeggEGraph
            try:
                transformer = self.transformers.get(func_name)
                megg_egraph = MeggEGraph.from_egraph(
                    egraph,
                    n_inline_leaves=0,
                    split_primitive_outputs=False,
                    func_transformer=transformer
                )
                logger.debug(
                    f"Created MeggEGraph with {len(megg_egraph.eclasses)} eclasses")
            except Exception as e:
                logger.warning(
                    f"Failed to create MeggEGraph for {func_name}: {e}")
                continue

            # 创建matcher
            matcher = SkeletonMatcher(megg_egraph)

            # 对每个骨架进行匹配
            for skeleton in self.skeletons:
                logger.debug(f"Matching skeleton: {skeleton.instr_name}")
                try:
                    matches = matcher.match_skeleton(skeleton)
                    logger.info(
                        f"Found {len(matches)} matches for skeleton '{skeleton.instr_name}'")

                    # 如果有匹配，在 MeggEGraph 中添加 custom_instr nodes
                    if matches:
                        logger.info(
                            f"Adding custom_instr nodes for '{skeleton.instr_name}'")

                        try:
                            # 查找函数参数对应的 eclass IDs
                            arg_eclasses = self._find_arg_eclasses(
                                megg_egraph,
                                len(skeleton.arg_vars)
                            )
                            arg_name_to_eclass: Dict[str, str] = {}
                            for idx, arg_var in enumerate(skeleton.arg_vars):
                                var_name = _get_var_name(arg_var)
                                if idx < len(arg_eclasses):
                                    arg_name_to_eclass[var_name] = arg_eclasses[idx]

                            # 提取 result_type 字符串
                            result_type_str = self._extract_result_type(
                                skeleton.result_type)

                            for eclass_id, var_bindings in matches:
                                logger.info(
                                    f"Processing match at eclass {eclass_id}")
                                logger.info(f"  var_bindings: {var_bindings}")
                                logger.info(
                                    f"  arg_name_to_eclass: {arg_name_to_eclass}")
                                operand_eclasses: List[str] = []
                                for arg_var in skeleton.arg_vars:
                                    var_name = _get_var_name(arg_var)
                                    operand_ec = var_bindings.get(var_name)
                                    if operand_ec is None:
                                        operand_ec = arg_name_to_eclass.get(
                                            var_name)
                                    logger.info(
                                        f"  Variable '{var_name}' → eclass '{operand_ec}'")
                                    if operand_ec is None:
                                        raise ValueError(
                                            f"Unable to resolve operand for variable '{var_name}'")
                                    operand_eclasses.append(operand_ec)

                                logger.info(
                                    f"  Final operand_eclasses: {operand_eclasses}")

                                # 在匹配的 eclass 中添加 custom_instr node
                                node_id = megg_egraph.add_custom_instr_node(
                                    eclass_id=eclass_id,
                                    instr_name=skeleton.instr_name,
                                    operands=operand_eclasses,
                                    result_type=result_type_str
                                )
                                logger.info(
                                    f"Added custom_instr node {node_id} to eclass {eclass_id}")
                                match_count += 1

                        except ValueError as e:
                            logger.error(
                                f"Failed to find argument eclasses for '{skeleton.instr_name}': {e}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to add custom_instr nodes: {e}")
                            import traceback
                            traceback.print_exc()

                except Exception as e:
                    logger.warning(
                        f"Failed to match skeleton '{skeleton.instr_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Save modified MeggEGraph for this function
            # This preserves the custom_instr nodes for extraction
            self.modified_megg_egraphs[func_name] = megg_egraph
            logger.info(f"Saved modified MeggEGraph for function {func_name}")

        return match_count

    def _find_arg_eclasses(self, megg_egraph, num_args: int) -> List[str]:
        """
        Find eclasses for function arguments in MeggEGraph.

        Args:
            megg_egraph: MeggEGraph instance
            num_args: Expected number of arguments

        Returns:
            List of eclass IDs for arguments [arg0, arg1, ...]

        Raises:
            ValueError: If unable to find all arguments
        """
        arg_eclass_map: Dict[int, str] = {}

        for eclass in megg_egraph.eclasses.values():
            for node in eclass.nodes:
                if node.op != "Term.arg" or node.value is None:
                    continue

                try:
                    arg_idx = int(node.value)
                except (TypeError, ValueError):
                    logger.debug(
                        f"Skipping arg node with non-integer value {node.value}")
                    continue

                if not (0 <= arg_idx < num_args):
                    logger.debug(
                        f"Ignoring arg index {arg_idx} outside expected range")
                    continue

                if arg_idx not in arg_eclass_map:
                    arg_eclass_map[arg_idx] = eclass.eclass_id
                    logger.debug(
                        f"Found arg{arg_idx} at eclass {eclass.eclass_id}")

        missing = [idx for idx in range(num_args) if idx not in arg_eclass_map]
        if missing:
            raise ValueError(f"Missing eclasses for arguments: {missing}")

        ordered = [arg_eclass_map[idx] for idx in range(num_args)]
        logger.info(f"Found all {num_args} argument eclasses: {ordered}")
        return ordered

    def _extract_result_type(self, result_type) -> str:
        """
        Extract type string from egglog.String or other representations.

        Args:
            result_type: egglog.String("i32") or None

        Returns:
            "i32" or "void"
        """
        if result_type is None:
            return "void"

        # egglog.String __str__ returns 'String("i32")'
        type_str = str(result_type)

        # Extract content within quotes
        if type_str.startswith('String("') and type_str.endswith('")'):
            return type_str[8:-2]  # Remove 'String("' and '")'

        # Fallback: return as-is (may already be "i32" format)
        return type_str.strip('"')

    def _collect_initial_egraph_statistics(self):
        """Collect e-graph statistics at initialization (before any rewrites)."""
        try:
            total_eclasses = 0
            total_enodes = 0

            for func_name, egraph in self.egraphs.items():
                # Convert to MeggEGraph to get statistics
                transformer = self.transformers.get(func_name)
                if not transformer:
                    continue

                try:
                    from megg.egraph.megg_egraph import MeggEGraph
                    megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer=transformer)
                    stats = megg_egraph.get_statistics()

                    # MeggEGraph.get_statistics() returns 'total_eclasses' and 'total_nodes'
                    total_eclasses += stats.get('total_eclasses', 0)
                    total_enodes += stats.get('total_nodes', 0)

                    logger.info(f"Initial e-graph statistics for {func_name}: {stats}")
                except Exception as e:
                    logger.warning(f"Failed to get initial statistics for {func_name}: {e}")

            initial_stats = {
                'num_eclasses': total_eclasses,
                'num_enodes': total_enodes
            }

            if self.state_manager:
                self.state_manager.record_initial_egraph_statistics(initial_stats)

            logger.info(f"Total initial e-graph statistics: {initial_stats}")

        except Exception as e:
            logger.error(f"Failed to collect initial e-graph statistics: {e}")

    def _collect_egraph_statistics(self):
        """Collect e-graph statistics after saturation (before extraction to MeggEGraph)."""
        try:
            total_eclasses = 0
            total_enodes = 0

            for func_name, egraph in self.egraphs.items():
                # Convert to MeggEGraph to get statistics
                transformer = self.transformers.get(func_name)
                if not transformer:
                    continue

                try:
                    from megg.egraph.megg_egraph import MeggEGraph
                    megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer=transformer)
                    stats = megg_egraph.get_statistics()

                    # MeggEGraph.get_statistics() returns 'total_eclasses' and 'total_nodes'
                    total_eclasses += stats.get('total_eclasses', 0)
                    total_enodes += stats.get('total_nodes', 0)

                    logger.info(f"E-graph statistics for {func_name}: {stats}")
                except Exception as e:
                    logger.warning(f"Failed to get statistics for {func_name}: {e}")

            egraph_stats = {
                'num_eclasses': total_eclasses,
                'num_enodes': total_enodes
            }

            if self.state_manager:
                self.state_manager.record_egraph_statistics(egraph_stats)

            logger.info(f"Total e-graph statistics after saturation: {egraph_stats}")

        except Exception as e:
            logger.error(f"Failed to collect e-graph statistics: {e}")

    def generate_statistics_report(self) -> Dict[str, Any]:
        """Generate a comprehensive statistics report suitable for JSON output."""
        if not self.state_manager:
            return {}

        # Count internal rewrites that were actually used (num_matches > 0)
        used_internal_count = len([
            r for r in self.state_manager.internal_rewrites_details
            if r.get('num_matches', 0) > 0
        ])

        return {
            'internal_rewrites': used_internal_count,
            'external_rewrites': self.state_manager.external_rewrites_details,
            'custom_rewrites': self.state_manager.custom_rewrites,
            'initial_egraph_statistics': self.state_manager.initial_egraph_statistics,
            'egraph_statistics': self.state_manager.egraph_statistics,
            'time_elapsed': time.time() - self.state_manager.start_time if self.state_manager.start_time else 0
        }
