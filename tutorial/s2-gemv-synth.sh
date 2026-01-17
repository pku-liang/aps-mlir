#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

cd $APS/tutorial

CADL_INPUT="${1:-$APS/tutorial/cadl/vgemv3d.cadl}"
NAME="$(basename "$CADL_INPUT" .cadl)"
OUT_DIR="$APS/tutorial/outputs/$NAME"

mkdir -p "$OUT_DIR"

pixi run mlir "$CADL_INPUT" "$OUT_DIR/0_${NAME}_preopt.mlir"

pixi run opt-debug "$OUT_DIR/0_${NAME}_preopt.mlir" /dev/null "$OUT_DIR/${NAME}.log"

extract_pass_dump() {
  local pass_pattern="$1"
  local infile="$2"
  local outfile="$3"

  awk -v pass="$pass_pattern" '
    /^\/\/ -----\/\/ IR Dump After /{
      if (inblock) { exit }
      if ($0 ~ pass) { inblock=1; next }
    }
    inblock { print }
  ' "$infile" > "$outfile"
}

LOG_FILE="${LOG_FILE:-$OUT_DIR/${NAME}.log}"

PASS_SPECS=(
  "memory-map|memory-map|$OUT_DIR/1_${NAME}_memory-map.mlir"
  "infer-affine-mem-access|infer-affine-mem-access|$OUT_DIR/2_${NAME}_affine.mlir"
  "hls-unroll|hls-unroll|$OUT_DIR/3_${NAME}_unroll.mlir"
  "array-partition|new-array-partition|$OUT_DIR/4_${NAME}_array-partition.mlir"
  "tor-timegraph|tor-time-graph|$OUT_DIR/5_${NAME}_scheduled.mlir"
  "aps-to-cmt2-gen|aps-to-cmt2-gen|$OUT_DIR/6_${NAME}_cmt2.mlir"
)

for spec in "${PASS_SPECS[@]}"; do
  IFS='|' read -r label pattern outfile <<< "$spec"
  extract_pass_dump "$pattern" "$LOG_FILE" "$outfile"
done
