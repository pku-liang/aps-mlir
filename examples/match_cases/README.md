# Custom Instruction Matching Test Cases

This directory contains test cases for custom instruction pattern matching with skeleton-based control flow matching.

## Test Case Categories

### Basic Cases (01-07)
Exact structural matches between pattern and code.

### Partial Match Cases (08-11)
**NEW**: Test partial matching where code contains extra operations but the core pattern is still detectable.

---

## Test Case Descriptions

### 01-simple-add
**Type**: Simple computation pattern
- **Pattern**: `c = a + b`
- **Code**: Identical
- **Tests**: Basic egglog rewrite matching

### 02-bitwise
**Type**: Bitwise operation pattern
- **Pattern**: Bitwise operations (and, or, xor)
- **Code**: Identical
- **Tests**: Arithmetic operation matching

### 05-loop-if
**Type**: Control flow pattern
- **Pattern**: Loop with if-then-else inside
- **Code**: Identical
- **Tests**: Nested control flow skeleton matching

### 07-if-nested-loop
**Type**: Complex control flow
- **Pattern**: If-then-else with loops in both branches
- **Code**: Identical
- **Tests**: Multi-level skeleton matching with operand constraints

---

## New Test Cases (Partial Matching)

### 08-partial-match-with-extra-code ⭐
**Type**: Partial match with surrounding code

**Pattern** (08.mlir):
```mlir
func.func @simple_conditional(%cond: i1, %a: i32, %b: i32) -> i32 {
  %result = scf.if %cond -> (i32) {
    %then_result = arith.addi %a, %b : i32
    scf.yield %then_result : i32
  } else {
    %else_result = arith.subi %a, %b : i32
    scf.yield %else_result : i32
  }
  return %result : i32
}
```

**Code** (08_.mlir):
```mlir
func.func @test_partial(%cond: i1, %a: i32, %b: i32) -> i32 {
  // ❌ Extra: Pre-processing
  %c1 = arith.constant 1 : i32
  %a_plus_1 = arith.addi %a, %c1 : i32

  // ✅ MATCHES: Core if-then-else pattern
  %result = scf.if %cond -> (i32) {
    %then_result = arith.addi %a, %b : i32
    scf.yield %then_result : i32
  } else {
    %else_result = arith.subi %a, %b : i32
    scf.yield %else_result : i32
  }

  // ❌ Extra: Post-processing
  %final = arith.addi %result, %a_plus_1 : i32
  return %final : i32
}
```

**Expected**: Should match the if-then-else pattern despite extra operations.

**Tests**:
- Skeleton matching ignores surrounding operations
- Component instructions match within control flow
- Operand constraints verified for matched portion

---

### 09-loop-with-surrounding-code ⭐
**Type**: Loop pattern with pre/post processing

**Pattern** (09.mlir):
```mlir
func.func @accumulate_loop(%n: index, %init: i32, %step: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %next = arith.addi %acc, %step : i32
    scf.yield %next : i32
  }
  return %result : i32
}
```

**Code** (09_.mlir):
```mlir
func.func @complex_compute(%n: index, %init: i32, %step: i32) -> i32 {
  // ❌ Extra: Scale the step
  %c2 = arith.constant 2 : i32
  %scaled_step = arith.muli %step, %c2 : i32

  // ❌ Extra: Adjust init
  %c10 = arith.constant 10 : i32
  %adjusted_init = arith.addi %init, %c10 : i32

  // ✅ MATCHES: Core loop pattern (with adjusted values)
  %loop_result = scf.for %i = %c0 to %n step %c1
      iter_args(%acc = %adjusted_init) -> (i32) {
    %next = arith.addi %acc, %scaled_step : i32
    scf.yield %next : i32
  }

  // ❌ Extra: Post-process
  %final = arith.subi %loop_result, %init : i32
  return %final : i32
}
```

**Expected**: Should match the loop pattern.

**Challenge**: Loop uses `%adjusted_init` and `%scaled_step` instead of original `%init` and `%step`.

**Tests**:
- Pattern matches despite different variable names
- E-graph unions make equivalent terms match
- Skeleton structure matches regardless of operand identity

---

### 10-multiple-patterns-in-code ⭐
**Type**: Multiple occurrences of same pattern

**Pattern** (10.mlir):
```mlir
func.func @while_pattern(%init: i32, %limit: i32, %step: i32) -> i32 {
  %result = scf.while (%arg = %init) : (i32) -> i32 {
    %cond = arith.cmpi slt, %arg, %limit : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %step : i32
    scf.yield %next : i32
  }
  return %result : i32
}
```

**Code** (10_.mlir):
```mlir
func.func @double_while(%init: i32, %limit: i32, %step: i32) -> i32 {
  // ✅ FIRST MATCH: While loop
  %result1 = scf.while (%arg = %init) : (i32) -> i32 {
    %cond = arith.cmpi slt, %arg, %limit : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %step : i32
    scf.yield %next : i32
  }

  // Intermediate operation
  %c1 = arith.constant 1 : i32
  %intermediate = arith.addi %result1, %c1 : i32

  // ✅ SECOND MATCH: Another while loop
  %result2 = scf.while (%arg = %intermediate) : (i32) -> i32 {
    %cond = arith.cmpi slt, %arg, %limit : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %step : i32
    scf.yield %next : i32
  }

  return %result2 : i32
}
```

**Expected**: Should match **BOTH** while loops.

**Tests**:
- Multiple instances of same pattern detected
- Each match creates separate custom instruction
- No interference between matches

---

### 11-nested-if-variant ⭐
**Type**: Nested control flow with variations

**Pattern** (11.mlir):
```mlir
func.func @nested_if_pattern(%outer_cond: i1, %inner_cond: i1,
                             %a: i32, %b: i32) -> i32 {
  %result = scf.if %outer_cond -> (i32) {
    %inner_result = scf.if %inner_cond -> (i32) {
      %sum = arith.addi %a, %b : i32
      scf.yield %sum : i32
    } else {
      %diff = arith.subi %a, %b : i32
      scf.yield %diff : i32
    }
    scf.yield %inner_result : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }
  return %result : i32
}
```

**Code** (11_.mlir):
```mlir
func.func @complex_nested(%cond1: i1, %cond2: i1, %x: i32, %y: i32) -> i32 {
  // ❌ Extra: Pre-scaling
  %c5 = arith.constant 5 : i32
  %x_scaled = arith.muli %x, %c5 : i32

  // ✅ MATCHES: Outer if
  %result = scf.if %cond1 -> (i32) {
    // ❌ Extra: intermediate computation
    %c2 = arith.constant 2 : i32
    %temp = arith.addi %x, %c2 : i32

    // ✅ MATCHES: Inner if (nested)
    %inner = scf.if %cond2 -> (i32) {
      %sum = arith.addi %x, %y : i32
      scf.yield %sum : i32
    } else {
      %diff = arith.subi %x, %y : i32
      scf.yield %diff : i32
    }

    // ❌ Extra: adjust result
    %adjusted = arith.addi %inner, %temp : i32
    scf.yield %adjusted : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }

  // ❌ Extra: Post-processing
  %final = arith.addi %result, %x_scaled : i32
  return %final : i32
}
```

**Expected**: Should match the nested if pattern.

**Tests**:
- Nested skeleton matching works with extra operations
- Inner pattern matched recursively
- Variable name differences handled by operand constraints

---

## Testing Commands

### Run Individual Test Case
```bash
./megg-opt tests/match/match_cases/08-partial-match-with-extra-code/08_.mlir \
  --custom-instructions tests/match/match_cases/08-partial-match-with-extra-code/08.mlir
```

### Run All New Test Cases
```bash
for case in 08 09 10 11; do
  echo "=== Test Case $case ==="
  ./megg-opt tests/match/match_cases/${case}-*/$(ls tests/match/match_cases/${case}-*/ | grep '_\.mlir') \
    --custom-instructions tests/match/match_cases/${case}-*/$(ls tests/match/match_cases/${case}-*/ | grep -v '_\.mlir' | grep '\.mlir')
done
```

### Expected Behavior

For **partial match** cases:
1. ✅ Pattern skeleton should be found despite extra code
2. ✅ Component instructions should match within control flow
3. ✅ Operand constraints should verify correctly
4. ✅ Custom instruction created with correct operands
5. ✅ Extra operations should be preserved (not replaced)

For **multiple occurrence** cases:
1. ✅ Each occurrence should be detected separately
2. ✅ Multiple custom instructions created (one per match)

---

## Key Testing Aspects

### 1. Robustness to Extra Code
- Pattern matching should work even with unrelated operations
- Skeleton structure is what matters, not exact code identity

### 2. Variable Name Independence
- Patterns match based on structure and semantics
- Variable names can differ (`%a` vs `%x`, `%init` vs `%adjusted_init`)

### 3. E-graph Equivalence
- E-graph unions allow matching of equivalent expressions
- `%step` and `%scaled_step` may be in same eclass after rewrites

### 4. Nested Pattern Handling
- Recursive skeleton matching for nested control flow
- Each nesting level verified independently

### 5. Multiple Matches
- Same pattern can match multiple times in one function
- Each match should be independent

---

## Success Criteria

A test case **passes** if:

1. **Skeleton Match**: Control flow structure detected
2. **Component Match**: Leaf operations found and matched
3. **Constraint Verification**: Operand equality constraints satisfied
4. **Custom Instruction**: Created with correct operands and type
5. **Code Preservation**: Unmatched operations preserved in output
6. **MLIR Validity**: Generated MLIR is syntactically correct

---

## Debugging Tips

### If Skeleton Doesn't Match
- Check skeleton construction in pattern (verify control flow structure)
- Verify control flow index has the right operation types
- Check if `container_type` mapping is correct

### If Components Don't Match
- Verify component instructions are created during rewrite phase
- Check if `leaf_patterns` contains expected components
- Ensure component operands are extracted correctly

### If Constraints Fail
- Check `operand_constraints` list in skeleton
- Verify variable names extracted correctly (`_get_var_name`)
- Look at `var_bindings` vs `arg_name_to_eclass` in debug log

### If Multiple Matches Expected But Not Found
- Check if skeleton matcher returns all matches
- Verify each match has unique `eclass_id`
- Ensure loop over matches processes all

---

## Future Test Ideas

1. **Algebraic Equivalence**: Pattern `a+b`, Code `b+a` (tests commutativity)
2. **Constant Folding**: Pattern with variables, Code with constants
3. **Loop Unrolling**: Pattern with loop, Code with unrolled version
4. **Branch Elimination**: Pattern with if, Code with simplified branch
5. **Data Flow Variants**: Same control flow, different data dependencies

---

**Last Updated**: 2025-10-11
**Status**: Test cases 08-11 ready for validation
