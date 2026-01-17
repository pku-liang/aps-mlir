from __future__ import annotations
from typing import Optional, Tuple
import logging
import os
from datetime import datetime

from megg.utils import MModule, MOperation, OperationType, MLIRPassManager
from megg.egraph.func_to_terms import FuncToTerms
from megg.egraph.term import Term
from egglog import EGraph

logger = logging.getLogger(__name__)


def _find_loop_in_function(func: MOperation, target_loop_index: int) -> Optional[MOperation]:
    """
    Find a loop in a function by counting loops in traversal order.

    Args:
        func: Function operation to search
        target_loop_index: Index of the loop to find (0-based)

    Returns:
        Loop operation if found, None otherwise
    """
    loop_count = 0
    all_loops = []  # Collect all loops for debugging

    def walk_operations(op: MOperation) -> Optional[MOperation]:
        nonlocal loop_count

        # Check if this operation is a loop
        op_type = op.type
        if op_type in (OperationType.SCF_FOR, OperationType.AFFINE_FOR):
            all_loops.append((loop_count, str(op)[:100]))  # Store loop info
            if loop_count == target_loop_index:
                logger.info(f"Found target loop {target_loop_index} at traversal position {loop_count}")
                return op
            loop_count += 1

        # Recursively search in nested regions
        if op.has_regions:
            for region in op.get_regions():
                for block in region.get_blocks():
                    for nested_op in block.get_operations():
                        result = walk_operations(nested_op)
                        if result is not None:
                            return result

        return None

    # Start walking from function body
    if func.has_regions:
        for region in func.get_regions():
            for block in region.get_blocks():
                for op in block.get_operations():
                    result = walk_operations(op)
                    if result is not None:
                        return result

    # Log all loops found for debugging
    logger.debug(f"Searched for loop {target_loop_index}, found {len(all_loops)} loops total:")
    for idx, loop_str in all_loops:
        logger.debug(f"  Loop {idx}: {loop_str}")

    return None


def apply_loop_pass_by_name(
    func: MOperation,
    loop_index: int,
    pass_name: str,
    func_to_terms: FuncToTerms,
    egraph: EGraph,
    unroll_factor: int = 4,
    tile_size: int = 4,
) -> bool:
    """
    Apply a loop transformation pass to a specific loop and union result back to e-graph.

    This function:
    1. Finds the loop operation by name
    2. Gets the original loop's e-graph term using its ID
    3. Applies MLIR pass to create optimized version
    4. Converts optimized loop to e-graph term
    5. Unions the original and optimized terms using egraph.union() API

    Args:
        func: The MLIR function operation containing the loop
        loop_name: Name/attribute identifying the specific loop to transform
        pass_name: MLIR pass to apply ("loop-unroll", "loop-tile", etc.)
        func_to_terms: FuncToTerms transformer tracking SSA → Term mappings
        egraph: E-graph for union operations
        unroll_factor: Factor for loop unrolling (default 4)
        tile_size: Tile size for loop tiling (default 4)

    Returns:
        Dict with optimized module/function on success, None on failure.

    Example:
        >>> apply_loop_pass_by_name(
        ...     func=my_func,
        ...     loop_name="loop_0",
        ...     pass_name="loop-unroll",
        ...     func_to_terms=transformer,
        ...     egraph=egraph,
        ...     unroll_factor=4
        ... )
    """
    try:
        # DEBUG: Check transformer state
        logger.info(f"apply_loop_pass_by_name called with loop_index={loop_index}")
        logger.info(f"  func_to_terms.loop_to_term has {len(func_to_terms.loop_to_term)} loops: {list(func_to_terms.loop_to_term.keys())}")

        # Step 2: Get the original loop's e-graph term using its ID
        loop_op, original_term = _get_loop_info(loop_index, func_to_terms)
        if original_term is None:
            logger.error(
                f"Failed to find e-graph term for loop '{loop_index}'")
            return None

        logger.info(f"Found loop '{loop_index}' with e-graph term: {original_term}")

        # Step 3: Apply MLIR pass to create optimized version
        optimized_result = _apply_mlir_pass_to_loop(
            func, loop_op, loop_index, pass_name, unroll_factor, tile_size
        )
        if not optimized_result:
            logger.error(
                f"Failed to apply pass '{pass_name}' to loop '{loop_index}'")
            return None

        optimized_module, optimized_func = optimized_result
        logger.info(
            f"Successfully applied '{pass_name}' to loop '{loop_index}'")

        # Save intermediate MLIR to tmp directory
        try:
            from megg.utils import get_temp_dir
            tmp_dir = get_temp_dir()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # YYYYmmdd_HHMMSS_ms
            param_value = unroll_factor if pass_name != "loop-tile" else tile_size
            filename = f"loop_{loop_index}_{pass_name}_{param_value}_{timestamp}.mlir"
            filepath = tmp_dir / filename

            with open(filepath, 'w') as f:
                f.write(str(optimized_module))

            logger.info(f"📁 Saved intermediate MLIR: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate MLIR: {e}")

        # Validate optimized function
        if not optimized_func:
            logger.error("Optimized function is None")
            return None

        # Step 4: Convert optimized function to e-graph terms
        # IMPORTANT: Transform in the ORIGINAL egraph, not a temp one
        # This ensures all intermediate symbols are registered in the original egraph
        # so that union will work correctly
        logger.info("Converting optimized function to e-graph terms...")

        try:
            optimized_transformer = FuncToTerms.transform(
                optimized_func, egraph)  # Use original egraph, not temp
        except Exception as e:
            logger.error(
                f"FuncToTerms.transform failed on optimized function: {e}", exc_info=True)
            return None

        # Validate that transformation succeeded
        if not optimized_transformer:
            logger.error("Optimized function transformation returned None")
            return None

        # Get the optimized loop term from the transformer
        # NOTE: After transformation (e.g., unrolling), the loop structure may have changed
        # and the original loop_index might not exist in the transformed code.
        # In that case, we union the entire function body instead of just the loop.
        optimized_loop_term = None
        if loop_index in optimized_transformer.loop_to_term:
            _, optimized_loop_term = optimized_transformer.loop_to_term[loop_index]
            logger.info(f"✓ Found transformed loop at index {loop_index}")
        else:
            logger.info(f"Loop {loop_index} no longer exists after transformation (likely unrolled/eliminated)")
            logger.info(f"  Available loops in transformed code: {list(optimized_transformer.loop_to_term.keys())}")
            logger.info("  This is expected for loop unrolling - the loop has been expanded")
            # For unrolled loops, we still want to union the results
            # The optimization is captured in the transformed function body
            # Return success - the egraph already has the optimized terms registered
            return ({"success": True, "function": optimized_func}, egraph)

        # Instead of extracting and re-registering, we can directly union the terms
        # Both original_term and optimized_loop_term are already registered in egraph
        logger.info(f"Unioning original loop term with optimized loop term...")
        logger.info(f"  Original term: {original_term}")
        logger.info(f"  Optimized term: {optimized_loop_term}")

        try:
            import egglog
            # Direct union without extract/let - both terms are already in egraph
            egraph.register(egglog.union(original_term).with_(optimized_loop_term))
            logger.info(
                "✓ Successfully unioned original and optimized loop terms")
        except Exception as e:
            logger.error(f"Failed to union terms: {e}", exc_info=True)
            logger.error(f"  This may indicate the terms are not compatible")
            return None

        # Step 7: Run e-graph to propagate the equivalence
        try:
            egraph.run(1)  # Run one iteration to propagate
            logger.info("✓ E-graph updated with new equivalence")
        except Exception as e:
            logger.error(f"Failed to run e-graph: {e}", exc_info=True)
            return None

        logger.info(
            f"✓ Successfully optimized and unioned loop '{loop_index}'")
        return {
            "success": True,
            "module": optimized_module,
            "function": optimized_func,
        }, egraph

    except Exception as e:
        logger.error(f"Error in apply_loop_pass_by_name: {e}", exc_info=True)
    return None

def _is_loop_operation(op: MOperation) -> bool:
    """Check if an operation is a loop."""
    if not hasattr(op, 'type'):
        return False

    loop_types = {
        OperationType.SCF_FOR,
        OperationType.SCF_WHILE,
        OperationType.AFFINE_FOR
    }

    return op.type in loop_types


def _get_operation_name(op: MOperation) -> Optional[str]:
    """Get the name/label of an operation from its attributes.

    This matches the C++ logic in loop_unroll.cpp and loop_tiling.cpp which check
    for a StringAttr named "name".
    """
    # Check for "name" attribute (matching C++ side logic)
    if op.has_attr("name"):
        try:
            name_value = op.get_attr("name")
            if name_value:
                return str(name_value)
        except Exception:
            pass

    # No name attribute found
    return None


def _get_loop_info(
    loop_index: int,
    transformer: FuncToTerms,
) -> Tuple[MOperation, Term]:
    loop_map = transformer.loop_to_term
    if loop_index not in loop_map:
        logger.error(f"Loop index {loop_index} not found in loop_map. Available indices: {list(loop_map.keys())}")
        raise AssertionError(f"Loop index {loop_index} not in loop_map")
    return loop_map[loop_index]


def _apply_mlir_pass_to_loop(
    func: MOperation,
    loop_op: MOperation,
    loop_index: int,
    pass_name: str,
    unroll_factor: int,
    tile_size: int,
) -> Optional[tuple]:
    """
    Apply an MLIR pass to a specific loop using MLIRPassManager.

    Args:
        func: Original function containing the loop
        loop_op: The loop operation to transform
        loop_index: Index of the loop (for fallback finding)
        pass_name: Pass to apply
        unroll_factor: Unroll factor for loop-unroll
        tile_size: Tile size for loop-tile

    Returns:
        Tuple of (module, function) with transformed loop, or None on failure.
        We return the module to keep it alive (function references become invalid if module is destroyed).
    """
    if unroll_factor == 1 or tile_size == 1:
        return None

    try:
        # CRITICAL FIX: Clone module directly from parent operation
        # Do NOT use cast_to_module() as it steals ownership and invalidates func

        parent_op = func.get_parent_op()

        # Debug: check if parent_op is valid
        if parent_op is None:
            logger.error(f"Function {func.symbol_name} has no parent operation")
            return None

        logger.debug(f"Parent op type: {type(parent_op)}")

        # Clone the parent module operation directly
        try:
            result = parent_op._op.clone_with_mapping()
        except Exception as clone_err:
            logger.error(f"Failed to clone module: {clone_err}")
            import traceback
            traceback.print_exc()
            return None

        if result is None:
            logger.error("Failed to clone module - result is None")
            return None

        cloned_module_op, cpp_op_map = result

        # Now cast the cloned module to MModule (safe because it's a new copy)
        module = MOperation(cloned_module_op).cast_to_module()
        if module is None:
            logger.error("Failed to cast cloned operation to module")
            return None

        # DEBUG: Print cloned module
        logger.info(f"=== Cloned module before applying pass ===")
        logger.info(f"{str(module)[:500]}")

        # Create pass manager
        pass_manager = MLIRPassManager(module)

        # Get the function from cloned module using the mapping
        # cpp_op_map is a C++ dict, use ._op to get the underlying C++ pointer
        new_func_cpp = cpp_op_map.get(func._op)
        if not new_func_cpp:
            # Fallback: Find function by name in cloned module
            func_name = func.symbol_name
            logger.info(f"Function not in cpp_op_map, searching by name: {func_name}")
            cloned_funcs = module.get_functions()
            logger.info(f"Cloned module has {len(cloned_funcs)} functions")
            func_op = None
            for f in cloned_funcs:
                logger.info(f"  Function: {f.symbol_name}")
                if f.symbol_name == func_name:
                    func_op = f
                    break
            if func_op is None:
                logger.error(f"Could not find function '{func_name}' in cloned module")
                return None
        else:
            func_op = MOperation(new_func_cpp)
            logger.info(f"Found function in cpp_op_map: {func_op.symbol_name}")

        # Get the corresponding loop in the cloned module
        new_loop_cpp = cpp_op_map.get(loop_op._op)
        if not new_loop_cpp:
            # Fallback: Find loop by traversing cloned function
            # Count loops to match the loop_index
            logger.info(f"Loop not in cpp_op_map, searching by index: {loop_index}")
            logger.info(f"Cloned function content:\n{str(func_op)[:1000]}")
            new_loop_op = _find_loop_in_function(func_op, loop_index)
            if new_loop_op is None:
                logger.error(f"Could not find loop {loop_index} in cloned module")
                logger.error(f"Full cloned function:\n{str(func_op)}")
                return None
        else:
            new_loop_op = MOperation(new_loop_cpp)
            logger.info(f"Found loop in cpp_op_map")

        # Apply the requested pass
        if pass_name == "loop-unroll":
            pass_manager.register_affine_loop_unroll_passes(
                function=func_op,
                loop_op=new_loop_op,
                unroll_factor=unroll_factor
            )
            success = pass_manager.run_pass("LoopUnroll")
            if not success:
                logger.warning(f"Loop unroll pass returned False")

        elif pass_name == "loop-tile":
            logger.info(
                f"Registering loop tile pass with size {tile_size}")
            pass_manager.register_affine_loop_tiling_pass(
                function=func_op,
                loop_op=new_loop_op,
                default_tile_size=tile_size
            )
            success = pass_manager.run_pass("LoopTiling")
            if not success:
                logger.warning(f"Loop tiling pass returned False")

        elif pass_name == "loop-unroll-jam":
            logger.info(
                f"Registering loop unroll-jam pass with factor {unroll_factor}")
            pass_manager.register_affine_loop_unroll_jam_passes(
                function=func_op,
                loop_op=new_loop_op,
                unroll_jam_factor=unroll_factor
            )
            success = pass_manager.run_pass("LoopUnrollJam")
            if not success:
                logger.warning(f"Loop unroll-jam pass returned False")
        else:
            logger.warning(
                f"Pass '{pass_name}' not yet supported, returning original function")
            return (module, func_op)

        pass_manager.clear_custom_passes()

        # Verify the module after transformation
        if not pass_manager.verify_module():
            logger.error("Module verification failed after pass application")
            return None

        # Re-fetch the function from module after transformation
        # The func_op reference might be stale after pass application
        # Match by symbol_name to ensure we get the correct function
        target_func_name = func.symbol_name
        transformed_func = None
        func_count = 0

        for f in module.get_functions():
            func_count += 1
            if hasattr(f, 'symbol_name') and f.symbol_name == target_func_name:
                transformed_func = f
                logger.info(f"Found target function '{target_func_name}' in transformed module")
                break

        logger.info(
            f"Module has {func_count} function(s) after transformation")

        if not transformed_func:
            logger.error(f"Target function '{target_func_name}' not found in module after transformation")
            return None

        # Check function structure
        regions = transformed_func.get_regions()
        logger.info(f"Transformed function has {len(regions)} region(s)")
        if regions:
            blocks = regions[0].get_blocks()
            logger.info(f"First region has {len(blocks)} block(s)")
            if blocks:
                ops = blocks[0].get_operations()
                logger.info(f"First block has {len(list(ops))} operation(s)")

        # DEBUG: Print transformed function to see if it still has affine.for
        logger.info(f"=== Transformed function after pass ===")
        logger.info(f"{str(transformed_func)[:2000]}")

        # IMPORTANT: Apply lower-affine pass on Python side to convert affine ops to SCF
        # This is more reliable than the C++ approach which has issues
        # Also apply canonicalize to simplify index_cast operations for better pattern matching
        logger.info("Applying --lower-affine --canonicalize to convert affine to SCF and simplify")
        try:
            import tempfile
            import subprocess
            import os

            # Write module to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                temp_input = f.name
                f.write(str(module))

            # Run mlir-opt --lower-affine
            mlir_opt_path = "mlir-opt"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                temp_output = f.name

            result = subprocess.run(
                [mlir_opt_path, "--lower-affine", "--canonicalize", temp_input, "-o", temp_output],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"mlir-opt --lower-affine --canonicalize failed: {result.stderr}")
                # Clean up temp files
                os.unlink(temp_input)
                os.unlink(temp_output)
                return (module, transformed_func)  # Return original if lowering fails

            # Read the lowered module
            with open(temp_output, 'r') as f:
                lowered_mlir = f.read()

            # Clean up temp files
            os.unlink(temp_input)
            os.unlink(temp_output)

            # Parse the lowered module
            from megg.utils import MModule
            lowered_module = MModule.parse(lowered_mlir)

            # Get the transformed function from lowered module
            lowered_funcs = lowered_module.get_functions()
            if not lowered_funcs:
                logger.error("No functions found in lowered module")
                return (module, transformed_func)

            lowered_func = None
            for func in lowered_funcs:
                if func.symbol_name == target_func_name:
                    lowered_func = func
                    break

            if not lowered_func:
                logger.error(f"Function {target_func_name} not found in lowered module")
                return (module, transformed_func)

            logger.info(f"✓ Successfully lowered affine operations to SCF and canonicalized")
            logger.info(f"=== Function after lower-affine + canonicalize ===")
            logger.info(f"{str(lowered_func)[:2000]}")

            # Return the lowered module and function
            return (lowered_module, lowered_func)

        except Exception as e:
            logger.error(f"Error applying lower-affine pass: {e}", exc_info=True)
            return (module, transformed_func)

        # Return both module and function to keep module alive
        # Original func is NOT affected by this operation (we cloned the module)
        # return (module, transformed_func)

    except Exception as e:
        logger.error(f"Error applying MLIR pass: {e}", exc_info=True)
        return None
