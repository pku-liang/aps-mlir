from __future__ import annotations
from egglog import egraph
import pathlib
import os
import tempfile
from pathlib import Path
from typing import Literal, List, Tuple, Callable, Generic, TypeVar, Any, Set
import logging
from dataclasses import dataclass
import argparse
from .mlir_utils import (
    MModule,
    MLIRPassManager,
    MOperation,
    MRegion,
    MBlock,
    MValue,
    OperationType,
    MType
)
from .ir_builder import IRBuilder, Singleton
from .loop_analyzer import extract_loop_lengths
logger = logging.getLogger(__name__)
T = TypeVar("T")

__all__ = [
    'MModule',
    'MLIRPassManager',
    'IRBuilder',
    'MOperation',
    'MRegion',
    'MBlock',
    'MValue',
    'OperationType',
    'visualize_egraph',
    'Dispatcher',
    'Singleton',
    'MeggArgs',
    '_merge_custom_module',
    'extract_loop_lengths',
    'get_loop_nest_level',
    'count_nested_loops',
    'get_temp_dir',
]


def get_temp_dir() -> Path:
    """Get the temporary directory for megg intermediate files.

    Priority:
    1. MEGG_TEMP_DIR environment variable (if set)
    2. /tmp/megg (default)

    Returns:
        Path to the temporary directory (created if not exists)
    """
    temp_dir_env = os.environ.get("MEGG_TEMP_DIR")

    if temp_dir_env:
        temp_dir = Path(temp_dir_env)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="megg_"))

    os.environ["MEGG_TEMP_DIR"] = str(temp_dir)
    return temp_dir

class Dispatcher(Generic[T]):
    def __init__(self, mapping: List[Tuple[Set[T], Callable]]):
        self.mapping = mapping

    def __call__(self, _type: Any, *args):
        for type_set, func in self.mapping:
            if any(isinstance(_type, t) for t in type_set):
                return func(*args)
        raise RuntimeError(f"No support class for type: {type(_type)}")


@dataclass
class MeggArgs:
    """Structured argument container for megg-opt CLI."""
    input: str
    output: str | None
    rewrite_mode: Literal["internal", "external", "both", "none"]
    mlir_passes: str | None
    target_functions: str | None
    iter: int
    time_limit: float
    extract_cost_model: Literal["json", "builtin"]
    dump_egraph: str | None
    dump_state: str | None
    verbose: bool
    log_level: str
    disable_safeguards: bool
    emit_llvm: bool
    compile: Literal["object", "executable"] | None
    opt_level: Literal["O0", "O1", "O2", "O3"]
    target: str | None
    link_libs: str | None
    custom_instructions: str | None

    def __post_init__(self):
        """Setup logging configuration after initialization."""
        if self.verbose:
            log_level = logging.DEBUG
        else:
            log_level = getattr(logging, self.log_level.upper())

        logging.basicConfig(
            level=log_level,
            format="%(name)s - [%(levelname)s]: %(message)s",
        )
        # Reduce noise from egglog logging
        logging.getLogger('egglog').setLevel(logging.CRITICAL + 1)
        logging.getLogger('egglog.egraph').setLevel(logging.CRITICAL + 1)
        logging.getLogger('egglog.actions').setLevel(logging.CRITICAL + 1)

    def gen_output_path(self, folder: str = "outputs"):
        """Generate output path if not specified."""
        if self.output is None:
            import os
            from pathlib import Path
            os.makedirs(folder, exist_ok=True)
            basename = Path(self.input).stem

            # Determine extension based on output format
            if self.emit_llvm:
                ext = '.ll'
            elif self.compile == 'object':
                ext = '.o'
            elif self.compile == 'executable':
                ext = ''
            else:
                ext = '.mlir'

            # Add suffix to indicate optimization mode
            suffix = ''
            if self.rewrite_mode != 'none':
                suffix = f'_{self.rewrite_mode}'

            self.output = f'{folder}/{basename}_opt{suffix}{ext}'

            if self.verbose:
                print(f"Using default output path: {self.output}")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """Add all CLI arguments to the parser."""
        # Core arguments
        parser.add_argument(
            'input',
            help='Input MLIR file'
        )

        parser.add_argument(
            '-o', '--output',
            required=False,
            help='Output file (default: outputs/<input_basename>_opt.<ext>)'
        )

        parser.add_argument(
            '--rewrite-mode',
            choices=['internal', 'external', 'both', 'none'],
            default='both',
            help='Rewrite mode (default: both, use "none" to skip optimization)'
        )

        parser.add_argument(
            '--mlir-passes',
            type=str,
            help='External passes on MLIR (semicolon-separated)'
        )

        parser.add_argument(
            '--target-functions',
            type=str,
            help='Comma-separated list of function names to optimize (default: all functions)'
        )

        # Optimization parameters
        parser.add_argument(
            '--iter',
            type=int,
            default=100,
            help='Max iterations (default: 100)'
        )

        parser.add_argument(
            '--time-limit',
            type=float,
            default=300.0,
            help='Max runtime in seconds (default: 300)'
        )

        parser.add_argument(
            '--extract-cost-model',
            choices=['json', 'builtin'],
            default='builtin',
            help='Cost model for extraction (default: builtin)'
        )

        # Debug and logging options
        parser.add_argument(
            '--dump-egraph',
            type=str,
            help='Path to save e-graph visualization (.dot file)'
        )

        parser.add_argument(
            '--dump-state',
            type=str,
            help='Path to save compiler state (.json file)'
        )

        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )

        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set log level (default: INFO)'
        )

        parser.add_argument(
            '--disable-safeguards',
            action='store_true',
            help='Disable termination and rollback safeguards'
        )

        # LLVM backend options
        parser.add_argument(
            '--emit-llvm',
            action='store_true',
            help='Emit LLVM IR instead of MLIR'
        )

        parser.add_argument(
            '--compile',
            choices=['object', 'executable'],
            help='Compile to object file or executable'
        )

        parser.add_argument(
            '--opt-level',
            choices=['O0', 'O1', 'O2', 'O3'],
            default='O2',
            help='Optimization level for LLVM compilation (default: O2)'
        )

        parser.add_argument(
            '--target',
            type=str,
            help='Target triple for compilation (e.g., x86_64-unknown-linux-gnu)'
        )

        parser.add_argument(
            '--link-libs',
            type=str,
            help='Libraries to link (comma-separated)'
        )

        parser.add_argument(
            '--custom-instructions', '--match-pattern',
            dest='custom_instructions',
            type=str,
            help='Path to MLIR module containing custom instruction definitions'
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> MeggArgs:
        """Create MeggArgs from parsed CLI arguments."""
        return cls(
            input=args.input,
            output=args.output,
            rewrite_mode=args.rewrite_mode,
            mlir_passes=args.mlir_passes,
            target_functions=args.target_functions,
            iter=args.iter,
            time_limit=args.time_limit,
            extract_cost_model=args.extract_cost_model,
            dump_egraph=args.dump_egraph,
            dump_state=args.dump_state,
            verbose=args.verbose,
            log_level=args.log_level,
            disable_safeguards=args.disable_safeguards,
            emit_llvm=args.emit_llvm,
            compile=args.compile,
            opt_level=args.opt_level,
            target=args.target,
            link_libs=args.link_libs,
            custom_instructions=args.custom_instructions,
        )


def _merge_custom_module(dest: MModule, src: MModule):
    """Append custom instruction definitions to the destination module."""
    existing_symbols = set()

    for op in dest.get_functions():
        if op.symbol_name:
            existing_symbols.add(op.symbol_name)

    for op in src.get_functions():
        symbol_name = op.symbol_name
        if symbol_name and symbol_name not in existing_symbols:
            dest.append_to_module(op)
            existing_symbols.add(symbol_name)


def get_loop_nest_level(loop_op: MOperation) -> int:
    depth = 0
    parent = loop_op.get_parent_op()

    while parent is not None and \
            parent.type is not OperationType.UNKNOWN:
        try:
            parent_type = parent.type
        except Exception:
            parent_type = None

        if parent_type in {OperationType.SCF_FOR, OperationType.AFFINE_FOR}:
            depth += 1

        parent = parent.get_parent_op()

    return depth


def count_nested_loops(loop_op: MOperation) -> int:
    """Count the number of loops contained within the given loop operation."""

    loop_kinds = {
        OperationType.SCF_FOR,
        OperationType.AFFINE_FOR,
        OperationType.SCF_WHILE,
    }

    def _walk(op: MOperation, is_root: bool = False) -> int:
        total = 0
        try:
            op_type = op.type
        except Exception:
            op_type = None

        if not is_root and op_type in loop_kinds:
            total += 1

        if getattr(op, 'has_regions', False):
            try:
                regions = op.get_regions()
            except Exception:
                regions = []
            for region in regions:
                for block in region.get_blocks():
                    for child in block.get_operations():
                        total += _walk(child)

        return total

    return _walk(loop_op, is_root=True)
