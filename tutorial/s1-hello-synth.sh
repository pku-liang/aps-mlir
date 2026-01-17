#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

cd $APS/tutorial

CADL_INPUT="${1:-$APS/tutorial/cadl/hello.cadl}"
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
  "tor-timegraph|tor-time-graph|$OUT_DIR/1_${NAME}_scheduled.mlir"
)

for spec in "${PASS_SPECS[@]}"; do
  IFS='|' read -r label pattern outfile <<< "$spec"
  extract_pass_dump "$pattern" "$LOG_FILE" "$outfile"
done
