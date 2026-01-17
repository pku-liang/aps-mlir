# SDC Scheduling Algorithm

The APS scheduling infrastructure uses **System of Difference Constraints (SDC)** based modulo scheduling for high-level synthesis.

## Overview

The scheduler assigns cycle times to MLIR operations while respecting:
- Data dependencies (RAW, WAR, WAW)
- Resource constraints (multipliers, memory ports)
- Timing constraints (clock period)
- Loop pipelining requirements

## SDC Fundamentals

A System of Difference Constraints consists of inequalities:
```
x_j - x_i >= c_ij
```

Where `x_i`, `x_j` are schedule times for operations, and `c_ij` is the minimum separation.

**Key Insight**: This is equivalent to finding **longest paths** in a constraint graph.

### Constraint Types

**Data Dependency**: `t_dest - t_src >= latency(src)`

**Loop-Carried Dependency** (with initiation interval II): `t_dest - t_src >= latency(src) - distance * II`

**Resource Constraint**: `t_op2 - t_op1 >= 1` (if both need same resource)

## Loop Pipelining

**Initiation Interval (II)** = Number of cycles between starting consecutive loop iterations.

**Minimum II (MII)** = max(recMII, resMII)
- **recMII**: Determined by recurrence (feedback loops)
- **resMII**: Determined by resource constraints

### Resource MII Calculation

```
resMII = max over all resources R: ceil(total_usage(R) / available(R))
```

**Memory Port MII**:
- RAM_1P: `II >= num_accesses(M)`
- RAM_T2P: `II >= ceil(num_accesses(M) / 2)`

## Scheduling Algorithm

1. Calculate `MII = max(recMII, resMII)`
2. Try target_II if specified (if `target_II >= MII`)
3. If failed, binary search for minimum feasible II
4. Return achieved_II

### scheduleWithII()

1. Allocate SDC variables for each operation
2. Add dependency constraints
3. Solve for ASAP times (Bellman-Ford)
4. Minimize register lifetimes (LP solver)
5. Resolve resource constraints (iterative placement)

## SDC Solver

### Initial Solution (Bellman-Ford)

**Complexity**: O(V x E) where V = operations, E = dependencies

### Incremental Constraint Addition

Adding constraint `x - y >= c` uses Dijkstra from x with modified weights. Only updates affected variables, preserving previous solution.

## Register Lifetime Minimization

**LP Objective**: `minimize Sum(bitwidth[i] x l[i])`

**Constraints**: Lifetime must cover all uses.

**Intuition**: Shorter lifetimes = fewer registers = smaller hardware.

## Operation Chaining

Operations can execute in the same cycle if combinational delay allows. DFS through combinational paths adds constraints when cumulative delay exceeds clock period.

## Memory Port Scheduling

Each memory gets a dedicated resource for scheduling:
- RAM_1P: Max 1 access/cycle
- RAM_T2P: Max 2 accesses/cycle

### Burst Transfer Scheduling

Burst operations must execute consecutively with constraints `t[i] - t[i-1] = 1`.

## Scheduled Output

After scheduling, each operation has timing attributes:

```mlir
%0 = aps.readrf %arg0 {st = 0, end = 1}    // Cycles 0-1
%1 = aps.readrf %arg1 {st = 0, end = 1}    // Cycles 0-1 (parallel)
%2 = tor.addi %0 %1 {st = 1, end = 2}      // Cycles 1-2
aps.writerf %arg2, %2 {st = 1, end = 2}    // Cycles 1-2
```

## Performance Characteristics

| Phase | Complexity |
|-------|------------|
| CDFG construction | O(ops) |
| Dependency analysis | O(ops^2) |
| SDC initialization | O(deps x ops) |
| LP solving | Polynomial average |
| Resource resolution | O(ops x II x resources) |

**Overall**: O(ops^2 x log(II)) typical case

**Scalability**:
- Small loops (< 100 ops): < 1s
- Medium loops (100-1000 ops): 1-10s
- Large loops (> 1000 ops): May require partitioning

## Further Reading

- [hls-passes.md](hls-passes.md) - Optimization passes before scheduling
- [cadl-language.md](cadl-language.md) - CADL directives affecting scheduling
