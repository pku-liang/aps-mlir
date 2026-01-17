#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core" \
pixi run python ./megg-opt.py --mode c-e2e \
  "$SCRIPT_DIR/test_v3ddist_vs.c" \
  --custom-instructions "$SCRIPT_DIR/v3ddist_vs.mlir" \
  --encoding-json "$SCRIPT_DIR/v3ddist_vs.json" \
  -o "$SCRIPT_DIR/v3ddist_vs.out" \
  --keep-intermediate 
echo ""
echo "✓ Compilation complete!"
echo "  Executable: $SCRIPT_DIR/v3ddist_vs.out"
echo "  Disassembly: $SCRIPT_DIR/v3ddist_vs.asm"
