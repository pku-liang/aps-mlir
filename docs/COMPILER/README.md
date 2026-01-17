# APS Compiler

The APS compiler enables automatic utilization of custom instructions (ISAXs) in application code using **E-graph** based equality saturation combined with **MLIR** infrastructure.

## The Pattern Matching Challenge

Traditional pattern matching is **syntactic** - it matches exact code patterns:

```c
// Pattern: MAC instruction
acc += a[i] * b[i];

// Semantically equivalent but won't match:
acc = acc + a[i] * b[i];
acc = a[i] * b[i] + acc;
acc += a[i<<2+j] * b[j];
```

**Problems**: Control flow variations, expression variations, too many manual variants.

## Our Solution: E-graph + MLIR

The APS compiler combines:
- **E-graphs**: Compactly represent all equivalent expressions
- **MLIR passes**: Handle control flow transformations
- **Skeleton-components matching**: Decompose patterns for flexible matching

## Compiler Pipeline

1. **Parse** application C code and ISAX CADL descriptions
2. **Convert** both to MLIR
3. **Build E-graph** from application MLIR
4. **Apply internal rewrites** (algebraic laws)
5. **Apply external rewrites** (MLIR passes guided by ISAX patterns)
6. **Match** ISAX components in expanded E-graph
7. **Extract** optimal instruction sequence
8. **Generate** RISC-V binary with custom instructions

## Quick Start

```bash
# Compile application with ISAX matching
pixi run compile tutorial/cadl/vgemv3d.cadl csrc/test_gemv.c outputs/test.riscv

# View matched instructions
cat outputs/compile_logs/test.asm | grep insn
```

**Output**:
```asm
101ac:   62b5752b    .insn    4, 0x62b5752b   # Custom instruction!
```

## Key Concepts

### E-graphs

An **E-graph** compactly represents equivalent expressions. After rewrite `m * 2 = m << 1`, both representations coexist in the same e-class.

**Key Operations**:
- **Rewrite**: Add equivalent e-node to e-class
- **Union**: Merge two e-classes (discovered equivalence)
- **Extract**: Select best representative from e-class

### Semantic Translation

CADL ISAXs are translated to high-level MLIR patterns for matching (hardware-specific details stripped, only computation pattern remains).

### Hybrid Rewriting

**Internal Rewrites** (algebraic):
- `a + b -> b + a` (commutativity)
- `(a + b) * c -> a*c + b*c` (distributivity)
- `a * 1 -> a` (identity)

**External Rewrites** (MLIR passes):
- Loop unroll, fusion, reordering
- Guided by ISAX pattern structure

### Skeleton-Components Matching

ISAXs are decomposed into:
- **Skeleton**: Control flow structure (loops, branches)
- **Components**: Pure computation patterns

Matching process:
1. Match skeleton structure (loop bounds, nesting)
2. For each component, search e-graph for matches
3. Validate dependencies and ordering
4. Insert custom instruction e-node

## Extraction

Select optimal program from e-graph using cost model:

| Operation | Cost |
|-----------|------|
| Literal | 1 |
| Add/Sub | 2 |
| Mul | 4 |
| Load/Store | 5 |
| Custom Instr | 15 (configurable) |

## Output Generation

Matched instructions become inline assembly:

```mlir
// After matching
%result = llvm.inline_asm has_side_effects
    ".insn r 0x2B, 0x7, 0x31, $0, $1, $2",
    "=r,r,r,~{memory}" %arg0, %arg1 : (i32, i32) -> i32
```

## File Locations

| Component | Location |
|-----------|----------|
| Compiler CLI | `megg-opt.py` |
| Core compiler | `python/megg/compiler.py` |
| E-graph layer | `python/megg/egraph/` |
| Rewrite rules | `python/megg/rewrites/` |
| LLVM backend | `python/megg/backend/` |

## Further Reading

- [egraph-architecture.md](egraph-architecture.md) - E-graph concepts and MLIR integration
- [pattern-matching.md](pattern-matching.md) - Detailed matching algorithm
