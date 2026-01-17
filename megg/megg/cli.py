#!/usr/bin/env python3
"""
megg-opt: Command-line tool for MLIR + egglog optimization using the new Compiler class.
"""
from __future__ import annotations
from megg.backend.llvm_backend import MLIRToLLVMBackend
from megg.compiler import Compiler
from megg.rewrites import match_rewrites
import sys
import logging
import atexit
import gc
import argparse
from megg.utils import MModule, MeggArgs, _merge_custom_module
logger = logging.getLogger(__name__)


def _cleanup_before_exit():
    gc.collect()
    sys.stderr.flush()

atexit.register(_cleanup_before_exit)


def main(megg_args: MeggArgs):
    try:
        logger.debug(f"Loading {megg_args.input}...")
        module = MModule.parse_from_file(megg_args.input)

        match_ruleset = None
        skeletons = None
        custom_module = None
        loop_hints = None
        if megg_args.custom_instructions:
            custom_module = MModule.parse_from_file(
                megg_args.custom_instructions)
            # build_ruleset_from_module现在返回(ruleset, skeletons)
            print("Building custom instruction ruleset...")
            match_ruleset, skeletons = match_rewrites.build_ruleset_from_module(
                custom_module)
            logger.debug(f"Loaded {len(skeletons)} control flow skeletons")
            
            # Extract loop information from custom_module to guide optimization
            from megg.utils.loop_analyzer import extract_loop_lengths
            loop_hints = extract_loop_lengths(custom_module)
            logger.info(loop_hints)


        # Parse target functions
        target_functions = None
        if megg_args.target_functions:
            target_functions = [f.strip()
                                for f in megg_args.target_functions.split(',')]

        # Run optimization
        if megg_args.rewrite_mode != 'none':
            logger.info("Starting optimization with Compiler...")
            logger.info(
                f"Optimization pipeline: Internal → External → Custom Instructions")
            logger.info(f"Max iterations: {megg_args.iter}")
            logger.info(f"Time limit: {megg_args.time_limit}s")
            logger.info(
                f"Extract cost model: {megg_args.extract_cost_model}")
            if target_functions:
                logger.info(f"Target functions: {target_functions}")
            if match_ruleset is not None:
                logger.info(
                    f"Custom instructions: Enabled ({len(match_ruleset.__egg_ruleset__.rules)} patterns)")
            # Create compiler instance
            compiler = Compiler(module, target_functions,
                                match_ruleset=match_ruleset,
                                skeletons=skeletons,
                                loop_hints=loop_hints)

            # Run optimization
            result = compiler.schedule(
                max_iterations=megg_args.iter,
                time_limit=megg_args.time_limit,
                internal_rewrites=(megg_args.rewrite_mode in [
                                   'internal', 'both']),
                external_passes=True if megg_args.rewrite_mode in [
                    'external', 'both'] else False,
                # Enable custom rewrites if ruleset provided
                custom_rewrites=(match_ruleset is not None),
                enable_safeguards=not megg_args.disable_safeguards,
                # Pass output path to save immediately (workaround for nanobind crash)
                output_path=megg_args.output
            )

            if not result.success:
                logger.error(f"Optimization failed: {result.error_message}")
                sys.exit(1)

            optimized_module = result.optimized_module

            if custom_module is not None:
                _merge_custom_module(optimized_module, custom_module)

            # Print optimization results
            logger.info(f"Optimization completed successfully!")
            logger.info(
                f"Phase 1 - Internal rewrites: {result.internal_rewrites} rules")
            logger.info(
                f"Phase 2 - External rewrites: {result.external_rewrites} passes")
            logger.info(
                f"Phase 3 - Custom instructions: {result.custom_rewrites} patterns")
            logger.info(f"Time elapsed: {result.time_elapsed:.2f}s")
            logger.info(
                f"Optimization complete! Internal: {result.internal_rewrites}, External: {result.external_rewrites}, Custom: {result.custom_rewrites}, Time: {result.time_elapsed:.2f}s")

            # Generate visualizations and dumps if requested
            if megg_args.dump_egraph:
                compiler.visualize_egraph(megg_args.dump_egraph)
                logger.debug(
                    f"E-graph visualization saved to {megg_args.dump_egraph}")

            if megg_args.dump_state:
                compiler.dump_state(megg_args.dump_state)
                logger.debug(
                    f"Compiler state dumped to {megg_args.dump_state}")

        else:
            # No optimization requested
            optimized_module = module
            logger.info("Optimization skipped (--rewrite-mode none)")
            
        # Process through LLVM backend if requested
        if megg_args.emit_llvm or megg_args.compile:
            logger.info("Processing through LLVM backend...")
            backend = MLIRToLLVMBackend(verbose=megg_args.verbose)

            # Determine output format
            if megg_args.compile == 'object':
                output_format = 'object'
            elif megg_args.compile == 'executable':
                output_format = 'executable'
            else:
                output_format = 'llvm-ir'

            # Process link libraries
            link_libs = None
            if megg_args.link_libs:
                link_libs = megg_args.link_libs.split(',')

            # Process through LLVM backend
            result = backend.process(
                optimized_module,
                output_format=output_format,
                output_path=megg_args.output,
                optimization_level=megg_args.opt_level,
                target=megg_args.target,
                link_libs=link_libs
            )

            logger.debug(
                f"LLVM backend processing complete! Output: {megg_args.output}")
        else:
            # Save output MLIR
            logger.debug(f"Saving optimized MLIR to {megg_args.output}...")
            MModule.save_mlir(optimized_module, megg_args.output)

            # Post-process with mlir-opt to convert generic form to pretty form
            import subprocess
            import os
            mlir_opt_path = "3rdparty/llvm-project/build/bin/mlir-opt"
            if os.path.exists(mlir_opt_path):
                try:
                    # Run mlir-opt to reformat the output
                    result = subprocess.run(
                        [mlir_opt_path, megg_args.output, "--mlir-print-op-generic=false"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        # Write the pretty-printed output back
                        with open(megg_args.output, 'w') as f:
                            f.write(result.stdout)
                        logger.debug("Output formatted with mlir-opt")
                except Exception as e:
                    logger.warning(f"Failed to format output with mlir-opt: {e}")
            
            logger.info(f"Optimized MLIR saved to {megg_args.output}")

        if 'optimized_module' in locals() and hasattr(optimized_module, 'clear'):
            optimized_module.clear()
        if 'module' in locals() and hasattr(module, 'clear'):
            module.clear()
        if 'compiler' in locals():
            if hasattr(compiler, 'egraphs'):
                compiler.egraphs.clear()
            if hasattr(compiler, 'transformers'):
                compiler.transformers.clear()
            del compiler
        if 'optimized_module' in locals():
            del optimized_module
        if 'module' in locals():
            del module
        gc.collect()
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.debug(f"Error during processing: {e}", file=sys.stderr)
        if megg_args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main_e2e(argv=None):
    """Entry point for end-to-end C compilation mode."""
    parser = argparse.ArgumentParser(
        description="megg-opt (E2E mode): End-to-end C compilation with Megg optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('input', help='Input C source file')
    parser.add_argument('--custom-instructions',
                        help='Optional custom instruction MLIR file (if not provided, only basic optimization will be applied)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output executable path')
    parser.add_argument('--encoding-json', type=str,
                        help='JSON file with instruction encodings (opcode/funct3/funct7)')
    parser.add_argument('--external', type=bool,default=True,
                        help='Should run external passes')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate files for debugging')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--handson', action='store_true',
                        help='Enable handson mode - saves phase snapshots for tutorial visualization')

    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Import E2E compiler
    from megg.e2e_compiler import E2ECompiler

    # Create and run compiler
    compiler = E2ECompiler(
        source_file=args.input,
        custom_instr_mlir=args.custom_instructions,
        output=args.output,
        encoding_json=args.encoding_json,
        external=args.external,
        keep_intermediate=args.keep_intermediate,
        verbose=args.verbose,
        handson=args.handson
    )

    success = compiler.compile()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check for --mode argument early to decide which path to take
    if '--mode' in sys.argv:
        mode_idx = sys.argv.index('--mode')
        if mode_idx + 1 < len(sys.argv):
            mode = sys.argv[mode_idx + 1]
            if mode == 'c-e2e':
                # Build clean argv without --mode for E2E parser
                clean_argv = sys.argv[1:mode_idx] + sys.argv[mode_idx+2:]
                main_e2e(clean_argv)
                sys.exit(0)

    # Default: Original MLIR mode
    parser = argparse.ArgumentParser(
        description="megg-opt: MLIR + egglog optimization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    MeggArgs.add_cli_args(parser)
    args = parser.parse_args()
    user_specified_output = args.output is not None
    megg_args = MeggArgs.from_cli_args(args)
    if not user_specified_output:
        megg_args.gen_output_path()
    main(megg_args)
    if not user_specified_output and megg_args.output:
        try:
            with open(megg_args.output, "r", encoding="utf-8") as f:
                sys.stdout.write(f.read())
        except OSError as exc:
            sys.stderr.write(f"Failed to print optimized module: {exc}\n")
