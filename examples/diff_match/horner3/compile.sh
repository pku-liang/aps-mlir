#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core" \
pixi run python ./megg-opt.py --mode c-e2e \
  "$SCRIPT_DIR/test_horner3.c" \
  --custom-instructions "$SCRIPT_DIR/horner3.mlir" \
  --encoding-json "$SCRIPT_DIR/horner3.json" \
  -o "$SCRIPT_DIR/horner3.out" \
  --keep-intermediate
echo ""
echo "✓ Compilation complete!"
echo "  Executable: $SCRIPT_DIR/horner3.out"
echo "  Disassembly: $SCRIPT_DIR/horner3.asm"
