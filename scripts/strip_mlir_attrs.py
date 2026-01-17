#!/usr/bin/env python3
"""Strip module-level attributes from MLIR file."""
import re
import sys

if len(sys.argv) != 2:
    print("Usage: strip_mlir_attrs.py <mlir_file>")
    sys.exit(1)

mlir_path = sys.argv[1]

with open(mlir_path, 'r') as f:
    text = f.read()

# Remove module attributes (dlti, llvm.* etc.)
pattern = re.compile(r'^(\s*module)\s+attributes\s*\{.*?\}\s*\{', re.DOTALL | re.MULTILINE)
cleaned_text, count = pattern.subn(r'\1 {', text, count=1)

if count > 0:
    with open(mlir_path, 'w') as f:
        f.write(cleaned_text)
