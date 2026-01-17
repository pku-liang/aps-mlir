# Pattern Matching

How the APS compiler matches application code against ISAX patterns using skeleton-components decomposition and e-graph search.

## Overview

Pattern matching follows three steps:

1. **Semantic Translation**: Convert ISAX CADL to matching patterns
2. **Skeleton-Components Decomposition**: Separate control flow from computation
3. **E-graph Search**: Find matches in the expanded application e-graph

## Semantic Translation

ISAX descriptions are translated to high-level MLIR computation patterns. Hardware-specific details (`_irf`, `_burst_read`) are stripped; only the computation pattern remains.

## Skeleton-Components Decomposition

### Motivation

Matching entire ISAXs directly is fragile because loop bounds may differ, algebraic variations create many forms, and control flow structure must match exactly.

### Solution

Split the ISAX into:
- **Skeleton**: Control flow structure (loops, conditions)
- **Components**: Pure computation patterns (expressions)

**Benefits**:
1. Components match via e-graph (handles algebraic variations)
2. Skeleton provides matching constraints
3. Each component can match independently

## Component Patterns

Components are expressed as egglog rewrite rules:

```lisp
; MAC pattern: acc' = acc + a * b
(rewrite
    (Term.add ?acc (Term.mul (Term.load ?arr1 ?idx1) (Term.load ?arr2 ?idx2)))
    (Term.custom_instr "mac" ?acc ?arr1 ?idx1 ?arr2 ?idx2))
```

**Pattern Variables**: `?acc` (accumulator), `?arr1`/`?arr2` (array bases), `?idx1`/`?idx2` (index expressions).

### Index Pattern Matching

Index expressions match structurally. After internal rewrites, `matrix[i*4+j]` and `matrix[i<<2+j]` are in the same e-class and can match.

## Matching Process

**Step 1**: Build application e-graph, apply algebraic rules and external rewrites.

**Step 2**: Decompose ISAX into skeleton and component rules.

**Step 3**: Search skeleton - find regions matching skeleton structure.

**Step 4**: Match components - apply component rewrite rules within skeleton match.

**Step 5**: Validate and extract - ensure match respects dependencies.

## Guided External Rewrites

The compiler compares application and ISAX loop structures:

| App Structure | ISAX Structure | Transformation |
|---------------|----------------|----------------|
| `for(0,8,1)` | `for(0,4,1)` unroll=2 | `--affine-loop-unroll=2` |
| `for(0,16,1)` | `for(0,4,4)` | `--affine-loop-tile=4` |
| Two separate loops | Single fused loop | `--affine-loop-fusion` |

## Handling Partial Matches

When full ISAX doesn't match:
- **Splitting ISAXs**: Separate load/compute/store, match individually
- **Multiple Small Matches**: Tile larger operations to match smaller ISAX

## Cost-Based Selection

When multiple matches possible:
```python
cost = instr_cost(match.isax) + sum(operand_cost(op) for op in match.operands)
```

Select match with lowest total cost.

## Debugging

```bash
# Verbose output
pixi run compile --verbose app.c isax.cadl out.riscv

# Visualization
pixi run compile --dump-egraph=before.svg app.c isax.cadl out.riscv
```

**Statistics**:
```
Patterns attempted: 5
Patterns matched: 2
Custom instructions inserted: 2
```

## Common Issues

**No Match Found**: Loop bounds don't match, missing algebraic rule, data type mismatch.

**Wrong Match**: Pattern too general, missing dependency check.

**Performance Regression**: Extra data movement, partial match overhead.

## Further Reading

- [egraph-architecture.md](egraph-architecture.md) - E-graph internals
- [README.md](README.md) - Compiler overview
