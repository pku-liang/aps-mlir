#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core" \
pixi run python ./megg-opt.py --mode c-e2e \
  "$SCRIPT_DIR/test_q15_mulr.c" \
  --custom-instructions "$SCRIPT_DIR/q15_mulr.mlir" \
  --encoding-json "$SCRIPT_DIR/q15_mulr.json" \
  -o "$SCRIPT_DIR/q15_mulr.out" \
  --keep-intermediate
echo ""
echo "✓ Compilation complete!"
echo "  Executable: $SCRIPT_DIR/q15_mulr.out"
echo "  Disassembly: $SCRIPT_DIR/q15_mulr.asm"
