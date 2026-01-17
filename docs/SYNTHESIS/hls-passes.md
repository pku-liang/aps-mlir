# HLS Optimization Passes

MLIR transformation passes for the APS synthesis flow.

## Pass Pipeline

| Pass | Purpose |
|------|---------|
| `memory-map` | Assign addresses to scratchpad memories |
| `raise-to-affine` | Convert SCF loops to Affine dialect |
| `hls-unroll` | Apply loop unrolling directives |
| `array-partition` | Partition arrays for parallel access |
| `infer-affine-mem` | Infer affine memory patterns |

## Memory Map Pass

**Location**: `lib/TOR/MemoryMapPass.cpp`

Creates a memory map tracking: memory name, bank assignments, base address, size, and partition info.

**Input**:
```mlir
memref.global @matrix : memref<16xi32> {
  partition_dim = 0, partition_factor = 4, partition_cyclic = true
}
memref.global @vec : memref<4xi32>
```

**Output**:
```mlir
aps.memorymap {
  aps.mem_entry "matrix" : banks([@matrix_0, @matrix_1, @matrix_2, @matrix_3]),
      base(0), size(64), count(4), cyclic(1)
  aps.mem_entry "vec" : banks([@vec]), base(64), size(16), count(1), cyclic(0)
  aps.mem_finish
}
```

## Raise to Affine Pass

**Location**: `lib/TOR/RaiseToAffinePass.cpp`

Converts SCF loops to Affine dialect for analysis (loop bounds, dependence analysis, polyhedral optimization).

**Input (SCF)**:
```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %idx = arith.muli %i, %c4 : index
  %addr = arith.addi %idx, %j : index
  %v = memref.load %matrix[%addr] : memref<16xi32>
}
```

**Output (Affine)**:
```mlir
affine.for %i = 0 to 4 {
  %v = affine.load %matrix[%i * 4 + %j] : memref<16xi32>
}
```

**Requirements**: Constant bounds, constant step, affine index expressions.

## HLS Unroll Pass

**Location**: `lib/TOR/HlsUnrollPass.cpp`

Applies loop unrolling based on `[[unroll(N)]]` directives.

**Full Unroll**: When unroll factor equals trip count.

**Partial Unroll**: When unroll factor < trip count, creates outer loop with replicated body.

**Benefits**: Exposes parallelism, reduces loop overhead, enables operation chaining.

## Array Partition Pass

**Location**: `lib/TOR/ArrayPartitionPass.cpp`

Distributes array elements across multiple memory banks for parallel access.

**Cyclic Partitioning**: `bank = element_index % num_banks`
- Good for stride-1 access patterns

**Block Partitioning**: `bank = element_index / (array_size / num_banks)`
- Good for tiled access patterns

**Example** (cyclic, factor=4, 16 elements):
- `matrix_0`: Elements 0, 4, 8, 12
- `matrix_1`: Elements 1, 5, 9, 13
- `matrix_2`: Elements 2, 6, 10, 14
- `matrix_3`: Elements 3, 7, 11, 15

## Pass Options

```bash
# Memory Map
--memory-map-base=0x1000     # Starting address
--memory-map-alignment=64    # Bank alignment

# HLS Unroll
--unroll-full-threshold=16   # Max trip count for full unroll

# Array Partition
--partition-threshold=8      # Min array size for partitioning
--partition-max-factor=16    # Max partition factor
```

## Debugging

```bash
# View intermediate IR after each pass
pixi run mlir-opt input.mlir --memory-map -o step1.mlir
pixi run mlir-opt step1.mlir --raise-to-affine -o step2.mlir
pixi run mlir-opt step2.mlir --hls-unroll -o step3.mlir
pixi run mlir-opt step3.mlir --array-partition -o step4.mlir

# Dump analysis info
pixi run mlir-opt input.mlir --memory-map --dump-memory-map
pixi run mlir-opt input.mlir --array-partition --dump-partition-info
```

## Common Issues

**Non-Affine Index**: Loop index contains non-affine expressions. Solution: Use only constants and loop IVs.

**Partition Factor Mismatch**: Unroll factor doesn't match partition factor. Solution: Match them for optimal parallelism.

**Memory Bank Conflict**: Multiple accesses to same bank in one cycle. Solution: Adjust partition factor or access pattern.

## Further Reading

- [scheduling.md](scheduling.md) - SDC scheduling algorithm
- [cadl-language.md](cadl-language.md) - CADL directive syntax
