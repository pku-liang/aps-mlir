# VCOVMAT3D_VS Optimization: Fused Accumulation

## Problem

The original **VCOVMAT3D.VV** instruction computes a single outer product between two 3D points. For point cloud registration (ICP algorithm), we need covariance over 16 point pairs, requiring:

- **16 separate instruction calls**
- **16 DMA startup penalties** (each ~10-15 cycles)
- **Total: ~240 cycles** for batch processing

This creates a bottleneck since DMA startup time dominates computation time.

## Solution: VCOVMAT3D_VS with Accumulation Fusion

The new **VCOVMAT3D.VS** (Vector-Scalar mode) fuses covariance computation with accumulation:

- **Processes all 16 points in ONE call**
- **3 burst DMA reads** (X[16], Y[16], Z[16])
- **Single centroid input** (3 scalar values)
- **Accumulated covariance matrix output**

### Key Innovation: Accumulator Loop

Instead of computing one outer product per call:
```
for 16 points:
    call VCOVMAT3D  // 16 × 15 cycles = 240 cycles
```

We compute accumulated sum in hardware:
```
VCOVMAT3D_VS:
    read X[16], Y[16], Z[16]  // 3 DMA reads
    for i in 0..15:           // Hardware loop (unrolled 4x)
        accumulate (pᵢ - centroid)(pᵢ - centroid)ᵀ
    write result              // 1 DMA write
```

## Performance Analysis

### Original VCOVMAT3D.VV (16 calls)
| Operation | Cycles | Count | Total |
|-----------|--------|-------|-------|
| DMA read (2 points) | 10 | 16 | 160 |
| Compute outer product | 5 | 16 | 80 |
| DMA write (6 values) | 5 | 16 | 80 |
| **Total** | | | **~240** |

### Optimized VCOVMAT3D.VS (1 call)
| Operation | Cycles | Count | Total |
|-----------|--------|-------|-------|
| DMA read X[16] | 10 | 1 | 10 |
| DMA read Y[16] | 10 | 1 | 10 |
| DMA read Z[16] | 10 | 1 | 10 |
| Read centroid (3 scalars) | 3 | 1 | 3 |
| Loop: 16 iterations × (3 subs + 6 MACs) | ~2 | 4 | 8 |
| DMA write result | 5 | 1 | 5 |
| **Total** | | | **~46** |

### Speedup: **5.2x faster!**

## Memory Layout

### Input: Structure-of-Arrays (SOA) Format
```
Point Cloud (192 bytes total):
┌─────────────────────────────────────────────┐
│ X[0]  X[1]  X[2]  ... X[15]  │ 64 bytes   │
├─────────────────────────────────────────────┤
│ Y[0]  Y[1]  Y[2]  ... Y[15]  │ 64 bytes   │
├─────────────────────────────────────────────┤
│ Z[0]  Z[1]  Z[2]  ... Z[15]  │ 64 bytes   │
└─────────────────────────────────────────────┘
```

### Centroid Input (12 bytes)
```
┌────────────────────────┐
│ cx (4B) │ cy (4B) │ cz (4B) │
└────────────────────────┘
```

### Output: Covariance Matrix (32 bytes)
```
┌─────────────────────────────────────────────────────────┐
│ c00 │ c01 │ c02 │ c11 │ c12 │ c22 │ count │ reserved │
│  σ_xx│ σ_xy│ σ_xz│ σ_yy│ σ_yz│ σ_zz│   16  │    0     │
└─────────────────────────────────────────────────────────┘
```

## Hardware Implementation Details

### Array Partitioning
- **X, Y, Z arrays**: 16-way cyclic partitioning → 16 parallel reads
- **Accumulator**: 8-way complete partitioning → 6 parallel MACs
- **Loop unrolling**: 4x factor → 4 iterations process 16 points

### Pipeline
1. **Cycle 0-10**: Burst read X[16]
2. **Cycle 10-20**: Burst read Y[16] (overlapped)
3. **Cycle 20-30**: Burst read Z[16] (overlapped)
4. **Cycle 30-33**: Read centroid (3 scalars)
5. **Cycle 34-41**: Compute loop (4 unrolled iterations)
   - Each iteration: 4 points × (3 subtracts + 6 MACs) = ~2 cycles
6. **Cycle 42-46**: Burst write result

Total: **46 cycles** (vs 240 cycles original)

## Integration with ICP Algorithm

The ICP registration pipeline now uses:

1. **V3DDIST.VV** - Find correspondences (16 pairwise distances)
2. **VCOVMAT3D.VS** - Estimate transformation (accumulated covariance) ← **NEW**
3. **VGEMV3D** - Apply transformation (16 3D matrix-vector products)
4. **VFPSMAX** - Check convergence (find max error)

### Updated Algorithm Flow
```python
# Initialization
centroid_src = compute_centroid(source_cloud)  # CPU
centroid_tgt = compute_centroid(target_cloud)  # CPU

for iteration in range(MAX_ITER):
    # Find correspondences (unchanged)
    distances = V3DDIST.VV(source, target)

    # Compute covariance (5.2x faster!)
    cov = VCOVMAT3D.VS(centered_source, centroid_src)  # Single call!

    # Derive transformation from covariance
    rotation, translation = svd_decompose(cov)  # CPU

    # Apply transformation (unchanged)
    source = VGEMV3D(rotation, source) + translation

    # Check convergence (unchanged)
    max_error = VFPSMAX(distances)
    if max_error < threshold:
        break
```

## Benefits Summary

1. ✅ **5.2x speedup** in covariance computation
2. ✅ **Reduced DMA overhead** from 16 startups to 3
3. ✅ **Same opcode** (0x2B) as original VCOVMAT3D
4. ✅ **Fits in 4 ISAx slots** (V3DDIST.VV, VCOVMAT3D.VS, VGEMV3D, VFPSMAX)
5. ✅ **Backward compatible** - VS mode distinguishable by funct7
6. ✅ **Natural batching** - processes exactly 16 points (typical for ICP)
7. ✅ **Metadata output** - count field enables validation

## Usage Example

```c
// Prepare data in SOA format
int32_t points_x[16] = {...};
int32_t points_y[16] = {...};
int32_t points_z[16] = {...};

// Compute centroid (CPU)
int32_t centroid[3] = {
    sum(points_x) / 16,
    sum(points_y) / 16,
    sum(points_z) / 16
};

// Call custom instruction
CovarianceMatrix cov;
vcovmat3d_vs(
    (uint32_t)points_x,    // rs1: base address (X, Y, Z arrays)
    (uint32_t)centroid,    // rs2: centroid address
    (uint32_t)&cov         // rd: output address
);

// Result contains accumulated covariance matrix
printf("σ_xx = %d\n", cov.c00);
printf("σ_xy = %d\n", cov.c01);
// ... etc
```

## Test Coverage

The test suite (`test_vcovmat3d_vs.c`) includes:

1. **Points centered at origin** - Validates symmetry
2. **Cube vertices** - Tests diagonal covariance (no correlation)
3. **Linear correlation** (y = x) - Tests strong positive correlation
4. **Identical points** - Tests zero covariance (edge case)
5. **Planar points** - Tests rank-deficient covariance

All tests compare hardware output against software reference implementation.

## Conclusion

The **VCOVMAT3D.VS** instruction demonstrates effective fusion of computation with accumulation, achieving **5.2x speedup** by eliminating redundant DMA operations. This optimization is critical for real-time point cloud processing applications like SLAM and 3D reconstruction.

The key insight: **batch processing with on-chip accumulation** beats repeated single-operation calls when DMA startup dominates execution time.
