#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core" \
pixi run python ./megg-opt.py --mode c-e2e \
  "$SCRIPT_DIR/test_vgemv3d.c" \
  --custom-instructions "$SCRIPT_DIR/vgemv3d.mlir" \
  --encoding-json "$SCRIPT_DIR/vgemv3d.json" \
  -o "$SCRIPT_DIR/vgemv3d.out" \
  --keep-intermediate 
echo ""
echo "✓ Compilation complete!"
echo "  Executable: $SCRIPT_DIR/vgemv3d.out"
echo "  Disassembly: $SCRIPT_DIR/vgemv3d.asm"
