# E-graph Architecture

How the APS compiler combines E-graphs with MLIR for pattern matching across equivalent program representations.

## What is an E-graph?

An **E-graph** (equality graph) compactly represents a large set of equivalent expressions:

- **E-nodes**: Functional operators or constants (e.g., `+`, `*`, `5`)
- **E-classes**: Sets of e-nodes that are semantically equivalent

Instead of choosing one representation, an e-graph stores **all equivalent representations** simultaneously.

## E-graph Operations

### Rewrite (e-matching + union)

**E-matching**: Find patterns in the e-graph.
**Union**: Merge equivalent e-classes.

Example: Rule `x * 2 -> x << 1` adds `<<` e-node to same e-class as `*` e-node.

### Extraction

Select the **best** (lowest cost) representative from each e-class based on a cost function.

## Egglog

APS uses **egglog**, a system that combines e-graphs with Datalog:

```lisp
; Type definitions
(datatype Expr (Num i64) (Var String) (Add Expr Expr) (Mul Expr Expr))

; Rewrite rules
(rewrite (Add x y) (Add y x))      ; Commutativity
(rewrite (Mul x (Num 1)) x)        ; Identity
```

## MLIR Integration

### Challenge: Control Flow

E-graphs naturally handle dataflow (expressions), but MLIR has basic blocks, regions, and side effects.

### Solution: Root-based Construction

Represent control flow as special e-nodes:

```mlir
scf.for %i = 0 to 4 step 1 {
    %v = memref.load %arr[%i]
    %sum = arith.addi %acc, %v
    scf.yield %sum
}
```

**E-graph representation**:
```
for(0, 4, 1, yield(add(arg(%acc), load(%arr, arg(%i)))))
```

### Term System

```python
class Term(Enum):
    Add, Sub, Mul, Div        # Arithmetic
    Load, Store               # Memory
    For, While, If, Yield     # Control flow
    Arg, Const, CustomInstr   # Special
```

## Func-to-Terms Conversion

1. Walk MLIR operations
2. Map SSA values to Term nodes
3. Track block arguments
4. Handle regions recursively
5. Collect roots (returns, yields)

**Mappings**: `ssa_to_term`, `ssa_to_id`, `roots`

## Internal Rewrite Rules

### Algebraic Laws

```lisp
; Commutativity
(rewrite (Term.add ?x ?y) (Term.add ?y ?x))

; Associativity
(rewrite (Term.add ?x (Term.add ?y ?z)) (Term.add (Term.add ?x ?y) ?z))

; Distributivity
(rewrite (Term.add (Term.mul ?a ?c) (Term.mul ?b ?c)) (Term.mul (Term.add ?a ?b) ?c))

; Identity
(rewrite (Term.add ?x (Term.const 0)) ?x)
(rewrite (Term.mul ?x (Term.const 1)) ?x)

; Shift-multiply equivalence
(rewrite (Term.mul ?x (Term.const 2)) (Term.shl ?x (Term.const 1)))
```

### Constant Folding

```lisp
(rule ((= x (Term.add (Term.const ?a) (Term.const ?b))))
      ((union x (Term.const (+ ?a ?b)))))
```

## External Rewrites

MLIR passes expand the e-graph with control flow variations.

**Guidance Strategy**:
1. Compare application and ISAX loop structures
2. Determine required transformations
3. Apply MLIR passes
4. Union results into e-graph

Example: If ISAX expects unroll=2, apply `--affine-loop-unroll=2`.

## Extraction Algorithm

```python
def extract(egraph, cost_fn):
    best = {}
    for eclass_id in topological_order(egraph):
        for enode in egraph.eclasses[eclass_id].nodes:
            cost = cost_fn.node_cost(enode.op)
            for child_id in enode.children:
                cost += best[child_id].cost
            if cost < best_cost:
                best[eclass_id] = ExpressionNode(enode.op, children, cost)
    return best
```

### Cost Function

| Operation | Cost |
|-----------|------|
| const, arg | 1.0 |
| add, sub, shl, shr | 2.0 |
| mul | 4.0 |
| div, rem | 8.0 |
| load, store | 5.0 |
| for, if | 10.0 |
| while | 15.0 |
| custom_instr | 15.0 (configurable) |

## Debugging

```python
# Visualization
compiler.visualize_egraph("output.svg")

# Statistics
stats = egraph.get_statistics()
print(f"E-classes: {stats['num_eclasses']}, E-nodes: {stats['num_enodes']}")
```

## Best Practices

1. Keep terms simple (complex nested structures are hard to match)
2. Use typed rules (separate rules for i32, f32, etc.)
3. Limit saturation (set iteration/time limits)
4. Order phases: Internal -> External -> Matching
5. Validate matches (check dependencies after matching)

## Further Reading

- [pattern-matching.md](pattern-matching.md) - How patterns are matched
- [Egglog Paper](https://arxiv.org/abs/2304.04332) - Academic foundation
