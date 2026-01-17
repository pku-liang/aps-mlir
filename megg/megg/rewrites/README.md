# Megg Internal Rewrites - Mathematical Rewrite Rules

This module provides egglog-based internal mathematical rewrite rules for the Megg compiler optimization pipeline.

## Overview

`internal_rewrites.py` implements two categories of rewrite rules:

1. **Basic Mathematical Laws** (`basic_math_laws`) - Safe and practical algebraic simplification rules
2. **Constant Folding** (`constant_folding_laws`) - Compile-time constant expression evaluation

## Quick Start

### Basic Usage

```python
import egglog
from egglog import EGraph
from megg.rewrites.internal_rewrites import register_internal_rewrites
from megg.egraph.term import Term, LitTerm

# Create e-graph
egraph = EGraph()

# Register basic mathematical laws (recommended default)
register_internal_rewrites(egraph)

# Or enable constant folding as well (optional)
register_internal_rewrites(egraph, include_constant_folding=True)
```

### Complete Example

```python
import egglog
from egglog import EGraph
from megg.rewrites.internal_rewrites import register_internal_rewrites
from megg.egraph.term import Term, LitTerm

# Setup environment
import os
tmp_dir = os.path.join(os.getcwd(), "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir

# Create e-graph
egraph = EGraph(save_egglog_string=True)

# Register rewrite rules
register_internal_rewrites(egraph, include_constant_folding=False)

# Create expression: x + (-x)
x = egraph.let("x", Term.arg(egglog.i64(0), egglog.String("i32")))
neg_x = egraph.let("neg_x", Term.neg(x, egglog.String("i32")))
expr = egraph.let("expr", Term.add(x, neg_x, egglog.String("i32")))

# Create expected result: 0
zero = egraph.let("zero", Term.lit(LitTerm.int(egglog.i64(0)), egglog.String("i32")))

# Apply rewrite rules
from megg.rewrites.internal_rewrites import basic_math_laws
math_laws = basic_math_laws()
egraph.run(math_laws.saturate())

# Verify x + (-x) = 0
result = egraph.check_bool(egglog.eq(expr).to(zero))
print(f"x + (-x) = 0? {result}")  # Output: True
```

## API Reference

### Main Functions

#### `register_internal_rewrites(egraph, include_constant_folding=False)`

Register all internal mathematical rewrite rules with the given e-graph.

**Parameters:**
- `egraph: EGraph` - The e-graph to register rules with
- `include_constant_folding: bool` - Whether to include constant folding rules (default False)

**Examples:**
```python
# Basic mathematical laws only
register_internal_rewrites(egraph)

# Include constant folding
register_internal_rewrites(egraph, include_constant_folding=True)
```

#### `basic_math_laws() -> egglog.Ruleset`

Return the basic mathematical laws ruleset.

**Example:**
```python
from megg.rewrites.internal_rewrites import basic_math_laws

egraph = EGraph()
math_laws = basic_math_laws()
egraph.run(math_laws.saturate())
```

#### `constant_folding_laws() -> egglog.Ruleset`

Return the constant folding ruleset.

**Example:**
```python
from megg.rewrites.internal_rewrites import constant_folding_laws

egraph = EGraph()
# First register egglog functions...
cf_laws = constant_folding_laws()
egraph.run(cf_laws.saturate())
```

## Supported Rewrite Rules

### Basic Mathematical Laws

#### Distributivity/Factoring (inspired by example.py)
```
(a*c)+(c*b) ⟷ (a+b)*c
(c*a)+(b*c) ⟷ (a+b)*c
(a*c)+(b*c) ⟷ (a+b)*c
(c*a)+(c*b) ⟷ c*(a+b)
```

#### Identity Rules
```
x + 0 = x        # Additive identity
x * 1 = x        # Multiplicative identity
x * 0 = 0        # Zero element
```

#### Negation and Subtraction
```
-(-x) = x        # Double negation
x - y ⟷ x + (-y) # Subtraction conversion
x + (-x) = 0     # Additive inverse
(-x) + x = 0     # Additive inverse (commutative)
x - x = 0        # Self subtraction
```

### Constant Folding Rules

Constant folding rules evaluate constant expressions at compile time:

```
2 + 3 → 5        # Integer addition
5 - 2 → 3        # Integer subtraction
3 * 4 → 12       # Integer multiplication
-(5) → -5        # Integer negation
```

## Type Support

All rules support the following types:
- `i32` - 32-bit integers
- `f32` - 32-bit floating point
- `f64` - 64-bit floating point

## Integration with Megg Compiler

### Integration with transform.py

```python
from megg.rewrites.internal_rewrites import register_internal_rewrites
from megg.egraph.transform import MeggTransform

def optimize_with_math_laws(mlir_module):
    transformer = MeggTransform()
    egraph = transformer.to_egraph(mlir_module)

    # Apply mathematical rewrite rules
    register_internal_rewrites(egraph)

    # Continue with other optimizations...
    result = transformer.from_egraph(egraph)
    return result
```

### Integration with CLI (megg-opt)

```python
# In cli.py
from megg.rewrites.internal_rewrites import register_internal_rewrites

def run_optimization(input_mlir, enable_math_laws=True):
    egraph = create_egraph_from_mlir(input_mlir)

    if enable_math_laws:
        register_internal_rewrites(egraph)

    # Run other optimizations...
    return generate_output_mlir(egraph)
```

## Performance Considerations

### Recommended Settings

- **Use basic mathematical laws by default**: Safe and proven, won't crash the solver
- **Use constant folding cautiously**: Useful but may increase solver complexity

### Avoided Patterns

Based on comments in example.py, the following patterns are intentionally avoided because they "extremely expand the design space and crash the solver":

```python
# Don't add these rules (will crash the solver)
# rewrite(Term.add(Term.add(a, b, ty), c, ty)).to(Term.add(a, Term.add(b, c, ty), ty))  # Associativity
# rewrite(Term.add(a, b, ty)).to(Term.add(b, a, ty))  # Commutativity
```

## Troubleshooting

### Common Issues

1. **Rules not applied**
   ```python
   # Make sure to run rules after creating terms
   egraph.let("expr", some_expression)
   math_laws = basic_math_laws()
   egraph.run(math_laws.saturate())  # Apply rules here
   ```

2. **Type mismatch**
   ```python
   # Ensure type strings match
   Term.add(x, y, egglog.String("i32"))  # Correct
   # Term.add(x, y, "i32")  # Wrong
   ```

3. **Constant folding failure**
   ```python
   # Make sure to register egglog functions first
   egraph.register(extract_int)
   egraph.register(add_i64)
   # Then register rules...
   ```

### Debugging

Enable debug output to see e-graph state:

```python
egraph = EGraph(save_egglog_string=True)
# ... operations ...
print(egraph.as_egglog_string)
```

## Testing

Run the built-in test:

```bash
cd python
PYTHONPATH=. python ./megg/rewrites/internal_rewrites.py
```

Expected output:
```
Registered basic mathematical laws
Basic law test: x + (-x) = 0? True
E-graph has 46 lines
```

## Contributing

### Adding New Rules

1. Add rules to the appropriate function
2. Add rules for each supported type
3. Add test cases
4. Ensure no solver performance issues

### Code Style

Follow existing patterns:
```python
# Apply rules for each type
for ty in [i32_ty, f32_ty, f64_ty]:
    rules.append(
        rewrite(pattern).to(replacement)
    )
```

## License

Follows the Megg project license.