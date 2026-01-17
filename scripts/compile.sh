#!/bin/bash
# compile.sh - Full compilation pipeline from CADL + test.c to executable
# Usage: compile.sh <cadl_file> <test_c_file> <output_executable> [--handson]
#
# This script performs the complete compilation:
# 1. CADL → C (cadl2c)
# 2. C → MLIR (cgeist)
# 3. CADL → encoding JSON
# 4. Full E2E compilation with Megg
#
# Intermediate files (.c, .mlir, .json) are placed next to the CADL file.
# The output executable is placed at the specified path (relative or absolute).
#
# Options:
#   --handson    Enable handson mode - saves phase snapshots for tutorial visualization

set -e

# Parse arguments
CADL_FILE=""
TEST_C_FILE=""
OUTPUT=""
HANDSON_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --handson)
            HANDSON_FLAG="--handson"
            shift
            ;;
        *)
            if [ -z "$CADL_FILE" ]; then
                CADL_FILE="$1"
            elif [ -z "$TEST_C_FILE" ]; then
                TEST_C_FILE="$1"
            elif [ -z "$OUTPUT" ]; then
                OUTPUT="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CADL_FILE" ] || [ -z "$TEST_C_FILE" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <cadl_file> <test_c_file> <output_executable> [--handson]"
    echo ""
    echo "Example:"
    echo "  $0 examples/diff_match/vgemv3d/vgemv3d.cadl examples/diff_match/vgemv3d/test_vgemv3d.c vgemv3d.riscv"
    echo ""
    echo "Options:"
    echo "  --handson    Enable handson mode for tutorial visualization"
    exit 1
fi

# Get absolute paths for input files
CADL_FILE=$(realpath "$CADL_FILE")
TEST_C_FILE=$(realpath "$TEST_C_FILE")

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract base name from CADL file
CADL_BASENAME=$(basename "$CADL_FILE" .cadl)

OUTPUT_DIR="$(dirname "${OUTPUT}")"
INTERMEDIATE_DIR="$OUTPUT_DIR/compile_logs"
mkdir -p "$INTERMEDIATE_DIR"

PATTERN_C="$INTERMEDIATE_DIR/${CADL_BASENAME}.c"
PATTERN_MLIR="$INTERMEDIATE_DIR/${CADL_BASENAME}.mlir"
ENCODING_JSON="$INTERMEDIATE_DIR/${CADL_BASENAME}.json"

echo "=== Megg Compilation Pipeline ==="
echo "  CADL:   $CADL_FILE"
echo "  Test C: $TEST_C_FILE"
echo "  Output: $OUTPUT"
echo ""

# Step 1: CADL → C
echo "[1/4] Converting CADL to C..."
pixi run -q python "$PROJECT_ROOT/aps-frontend" cadl2c "$CADL_FILE" -o "$PATTERN_C"
echo "  ✓ Created: $PATTERN_C"

# Step 2: C → MLIR (using cgeist)
echo "[2/4] Converting C to MLIR..."
pixi run -q cgeist -S -O3 "$PATTERN_C" -o "$PATTERN_MLIR"

# Strip module-level attributes (dlti, llvm.* etc.) that cause parsing issues
pixi run -q python "$PROJECT_ROOT/scripts/strip_mlir_attrs.py" "$PATTERN_MLIR"
echo "  ✓ Created: $PATTERN_MLIR"

# Step 3: CADL → encoding JSON
echo "[3/4] Extracting encoding JSON..."
pixi run -q python "$PROJECT_ROOT/aps-frontend" encoding "$CADL_FILE" -o "$ENCODING_JSON"
echo "  ✓ Created: $ENCODING_JSON"

# Step 4: Full E2E compilation
echo "[4/4] Running Megg E2E compilation..."
pixi run -q python "$PROJECT_ROOT/megg-opt.py" --mode c-e2e \
    "$TEST_C_FILE" \
    --custom-instructions "$PATTERN_MLIR" \
    --encoding-json "$ENCODING_JSON" \
    -o "$OUTPUT" \
    --keep-intermediate \
    $HANDSON_FLAG

# Move .asm, .stats.json, and .snapshots.json to INTERMEDIATE_DIR
OUTPUT_BASENAME=$(basename "$OUTPUT" .riscv)
OUTPUT_DIR=$(dirname "$OUTPUT")
if [ -f "$OUTPUT_DIR/${OUTPUT_BASENAME}.asm" ]; then
    mv "$OUTPUT_DIR/${OUTPUT_BASENAME}.asm" "$INTERMEDIATE_DIR/${CADL_BASENAME}.asm"
fi
if [ -f "$OUTPUT_DIR/${OUTPUT_BASENAME}.stats.json" ]; then
    mv "$OUTPUT_DIR/${OUTPUT_BASENAME}.stats.json" "$INTERMEDIATE_DIR/${CADL_BASENAME}.stats.json"
fi
if [ -f "$OUTPUT_DIR/${OUTPUT_BASENAME}.snapshots.json" ]; then
    mv "$OUTPUT_DIR/${OUTPUT_BASENAME}.snapshots.json" "$INTERMEDIATE_DIR/${CADL_BASENAME}.snapshots.json"
fi

echo ""
echo "=== Compilation Complete ==="
echo "  Executable:   $OUTPUT"
echo "  Pattern C:    $PATTERN_C"
echo "  Pattern MLIR: $PATTERN_MLIR"
echo "  Encoding:     $ENCODING_JSON"
[ -f "$INTERMEDIATE_DIR/${CADL_BASENAME}.asm" ] && echo "  Assembly:     $INTERMEDIATE_DIR/${CADL_BASENAME}.asm"
[ -f "$INTERMEDIATE_DIR/${CADL_BASENAME}.stats.json" ] && echo "  Stats:        $INTERMEDIATE_DIR/${CADL_BASENAME}.stats.json"
