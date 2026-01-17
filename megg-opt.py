#!/usr/bin/env python3
"""
megg-opt: MLIR + egglog optimization tool
"""

import sys
from pathlib import Path

# Add python directory to path so we can import megg
sys.path.insert(0, str(Path(__file__).parent / "megg"))

# Add MLIR Python bindings to path
mlir_path = Path(__file__).parent / "build" / "python_packages" / "mlir_core"
if mlir_path.exists():
    sys.path.insert(0, str(mlir_path))

if __name__ == "__main__":
    # Use runpy to execute cli.py as __main__
    import runpy
    sys.argv[0] = str(Path(__file__).parent / "python" / "megg" / "cli.py")
    runpy.run_module("megg.cli", run_name="__main__")
