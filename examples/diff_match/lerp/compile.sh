#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core" \
pixi run python ./megg-opt.py --mode c-e2e \
  "$SCRIPT_DIR/test_lerp.c" \
  --custom-instructions "$SCRIPT_DIR/lerp.mlir" \
  --encoding-json "$SCRIPT_DIR/lerp.json" \
  -o "$SCRIPT_DIR/lerp.out" \
  --keep-intermediate
echo ""
echo "✓ Compilation complete!"
echo "  Executable: $SCRIPT_DIR/lerp.out"
echo "  Disassembly: $SCRIPT_DIR/lerp.asm"
