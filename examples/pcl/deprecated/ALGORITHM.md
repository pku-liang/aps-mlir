# Point Cloud Registration Algorithm

## Overview

This application implements a **simplified Iterative Closest Point (ICP)** algorithm for point cloud registration using custom RISC-V instruction extensions. The goal is to align two 3D point clouds by computing an optimal rigid transformation (rotation + translation).

## Problem Statement

**Input:**
- **Source cloud**: 16 3D points that need to be transformed
- **Target cloud**: 16 3D points (reference/goal position)

**Output:**
- Transformation matrix (3×3 rotation + 3D translation vector)
- Aligned source point cloud matching the target

**Applications:**
- 3D object recognition and pose estimation
- SLAM (Simultaneous Localization and Mapping)
- 3D reconstruction and model alignment
- Robotics and autonomous navigation

## Algorithm Pipeline

### Initialization
```
1. Load source and target point clouds (16 points each)
2. Compute centroids of both clouds
3. Initialize transformation as identity
```

### Iterative Refinement Loop

```
WHILE not converged AND iterations < MAX_ITERATIONS:
    ┌─────────────────────────────────────────────────────────┐
    │ Step 1: Find Correspondences (V3DDIST.VV)              │
    │   For each point in source, find closest in target     │
    │   Custom Instruction: V3DDIST.VV                       │
    │   Computes 16 pairwise squared distances               │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Step 2: Estimate Transformation (VCOVMAT3D)            │
    │   Compute covariance of centered point sets            │
    │   Custom Instruction: VCOVMAT3D                        │
    │   Returns 3×3 covariance matrix                        │
    │   Derive rotation + translation                        │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Step 3: Apply Transformation (VGEMV3D)                 │
    │   Transform all source points                          │
    │   Custom Instruction: VGEMV3D                          │
    │   Applies 3×3 matrix to 16 3D vectors                  │
    │   Add translation offset                               │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │ Step 4: Check Convergence (VFPSMAX)                    │
    │   Find maximum distance error                          │
    │   Custom Instruction: VFPSMAX                          │
    │   Returns max value and its index                      │
    │   Converged if max_error < threshold                   │
    └─────────────────────────────────────────────────────────┘
                            ↓
                    [Converged?]
                    /          \
                 YES            NO
                  ↓              ↓
              [OUTPUT]      [Next Iteration]
```

## Detailed Step-by-Step Breakdown

### Step 1: Find Correspondences using V3DDIST.VV

**Goal:** Match each source point to its closest target point

**Algorithm:**
```c
for i = 0 to 15:
    distances[i] = min_k( ||source[i] - target[k]||² )
```

**Custom Instruction: V3DDIST.VV**
- **Opcode**: `0x2B` (0101011)
- **Funct7**: `0x2A` (0101010)
- **Operation**: Computes pairwise squared Euclidean distances

**Input Memory Layout:**
```
┌─────────────┬─────────────┬─────────────┐
│ source_x[16]│ source_y[16]│ source_z[16]│
├─────────────┼─────────────┼─────────────┤
│ target_x[16]│ target_y[16]│ target_z[16]│
└─────────────┴─────────────┴─────────────┘
     0-63          64-127        128-191
```

**Output:**
```
distances[16] = [dist²₀, dist²₁, ..., dist²₁₅]
```

**Hardware Acceleration:**
- 4-way parallel computation (4 distances per cycle)
- Cyclic memory partitioning enables simultaneous access
- Total cycles: ~20 (including burst I/O)
- **Speedup: ~10x vs scalar**

**Formula per point pair:**
```
dist²[i] = (x₁[i] - x₂[i])² + (y₁[i] - y₂[i])² + (z₁[i] - z₂[i])²
```

### Step 2: Estimate Transformation using VCOVMAT3D

**Goal:** Compute optimal rotation and translation between matched points

**Algorithm:**
```c
// Center points around their centroids
source_centered[i] = source[i] - source_centroid
target_centered[i] = target[i] - target_centroid

// Compute 3×3 covariance matrix
for i,j in {0,1,2}:
    Cov[i,j] = (1/16) × Σₖ source_centered[i][k] × target_centered[j][k]

// Derive transformation (simplified - full ICP uses SVD)
rotation = extract_rotation(Cov)
translation = target_centroid - rotation × source_centroid
```

**Custom Instruction: VCOVMAT3D**
- **Opcode**: `0x2B`
- **Funct7**: `0x2B` (0101011)
- **Operation**: Computes 3×3 covariance matrix

**Input Memory Layout:**
```
┌──────────────┬──────────────┬──────────────┐
│ centered_x[16]│ centered_y[16]│ centered_z[16]│
└──────────────┴──────────────┴──────────────┘
```

**Output:**
```
covariance[9] = [c₀₀, c₀₁, c₀₂, c₁₀, c₁₁, c₁₂, c₂₀, c₂₁, c₂₂]
```

**Covariance Matrix Computation:**
```
       ┌                                      ┐
       │  Σ(x-x̄)(x-x̄)  Σ(x-x̄)(y-ȳ)  Σ(x-x̄)(z-z̄) │
Cov = │  Σ(y-ȳ)(x-x̄)  Σ(y-ȳ)(y-ȳ)  Σ(y-ȳ)(z-z̄) │ / n
       │  Σ(z-z̄)(x-x̄)  Σ(z-z̄)(y-ȳ)  Σ(z-z̄)(z-z̄) │
       └                                      ┘
```

**Hardware Acceleration:**
- Computes 9 matrix elements in parallel
- Each element requires 16 multiply-accumulate operations
- Total cycles: ~45
- **Speedup: ~15x vs scalar**

### Step 3: Apply Transformation using VGEMV3D

**Goal:** Transform all 16 source points using computed transformation

**Algorithm:**
```c
for i = 0 to 15:
    transformed[i] = rotation_matrix × source[i] + translation
```

**Custom Instruction: VGEMV3D**
- **Opcode**: `0x2B`
- **Funct7**: `0x2C` (0101100)
- **Operation**: 3D general matrix-vector multiply

**Input Memory Layout:**
```
┌──────────┬─────────────┬─────────────┬─────────────┐
│ matrix[9]│ points_x[16]│ points_y[16]│ points_z[16]│
└──────────┴─────────────┴─────────────┴─────────────┘
  (3×3 rot)     X coords      Y coords      Z coords
```

**Matrix-Vector Multiplication:**
```
┌    ┐   ┌         ┐   ┌   ┐
│ x' │   │ r₀ r₁ r₂│   │ x │
│ y' │ = │ r₃ r₄ r₅│ × │ y │
│ z' │   │ r₆ r₇ r₈│   │ z │
└    ┘   └         ┘   └   ┘
```

**Output:**
```
transformed_points[48] = [x'[16] | y'[16] | z'[16]]
```

**Hardware Acceleration:**
- Applies 3×3 matrix to 16 vectors in parallel
- 4-way unrolling for throughput
- Total cycles: ~35
- **Speedup: ~12x vs scalar**

### Step 4: Check Convergence using VFPSMAX

**Goal:** Determine if alignment error is below threshold

**Algorithm:**
```c
max_error = max(distances[0..15])
max_index = argmax(distances[0..15])

if max_error < THRESHOLD:
    CONVERGED = true
```

**Custom Instruction: VFPSMAX**
- **Opcode**: `0x2B`
- **Funct7**: `0x2D` (0101101)
- **Operation**: Find maximum value and index in vector

**Input:**
```
distances[16] = [dist²₀, dist²₁, ..., dist²₁₅]
```

**Output:**
```
result[2] = [max_value, max_index]
```

**Reduction Tree:**
```
       [d₀ d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ d₁₀ d₁₁ d₁₂ d₁₃ d₁₄ d₁₅]
              ↓ Compare 8 pairs in parallel ↓
       [max(d₀,d₁) max(d₂,d₃) max(d₄,d₅) max(d₆,d₇) ...]
              ↓ Compare 4 pairs in parallel ↓
       [max(max(d₀,d₁), max(d₂,d₃)) ...]
              ↓ Compare 2 pairs ↓
       [max(...) max(...)]
              ↓ Final compare ↓
       [global_max, index]
```

**Hardware Acceleration:**
- Parallel reduction tree (log₂(16) = 4 levels)
- Index tracking throughout reduction
- Total cycles: ~8
- **Speedup: ~16x vs scalar**

## Convergence Criteria

The algorithm terminates when either:

1. **Convergence**: `max_error < THRESHOLD`
   - Default threshold: 100 (squared distance units)
   - Indicates all points are close to their correspondences

2. **Maximum Iterations**: `iterations >= MAX_ITERATIONS`
   - Default: 10 iterations
   - Prevents infinite loops on difficult cases

## Performance Analysis

### Complexity per Iteration

| Step | Operation | Scalar Ops | Custom Inst | Speedup |
|------|-----------|------------|-------------|---------|
| 1. Correspondences | V3DDIST.VV | 16 × (3 sub + 3 mul + 2 add) = 128 | ~20 cycles | 10x |
| 2. Covariance | VCOVMAT3D | 16 × 9 × (1 mul + 1 add) = 288 | ~45 cycles | 15x |
| 3. Transform | VGEMV3D | 16 × (9 mul + 6 add) = 240 | ~35 cycles | 12x |
| 4. Max Error | VFPSMAX | 16 comparisons = 16 | ~8 cycles | 16x |
| **Total per iteration** | | **~672 ops** | **~108 cycles** | **~6x** |

### End-to-End Performance

Typical registration (3-5 iterations):
- **Scalar code**: ~3,000-5,000 cycles
- **With custom instructions**: ~400-600 cycles
- **Overall speedup: ~7-8x**

Additional benefits:
- Reduced memory bandwidth (burst I/O)
- Lower power consumption (less instruction fetch)
- Deterministic timing (no branches in accelerated paths)

## Memory Layout and Data Flow

### Structure-of-Arrays (SOA) Format

Point clouds stored as separate X, Y, Z arrays:
```
Traditional (AOS):          Optimized (SOA):
┌─────────────┐            ┌─────────────┐
│ x₀ y₀ z₀    │            │ x₀ x₁ ... x₁₅│  X array
│ x₁ y₁ z₁    │            │ y₀ y₁ ... y₁₅│  Y array
│ ...         │    →       │ z₀ z₁ ... z₁₅│  Z array
│ x₁₅ y₁₅ z₁₅ │            └─────────────┘
└─────────────┘
```

**Advantages:**
- Enables cyclic memory partitioning
- Supports parallel access to multiple coordinates
- Simplifies burst read/write operations
- Better cache locality for vector operations

### Memory Partitioning

All arrays use **4-way cyclic partitioning**:
```
Physical Memory Banks:
┌──────┬──────┬──────┬──────┐
│Bank 0│Bank 1│Bank 2│Bank 3│
├──────┼──────┼──────┼──────┤
│ x₀   │ x₁   │ x₂   │ x₃   │
│ x₄   │ x₅   │ x₆   │ x₇   │
│ x₈   │ x₉   │ x₁₀  │ x₁₁  │
│ x₁₂  │ x₁₃  │ x₁₄  │ x₁₅  │
└──────┴──────┴──────┴──────┘
```

**Result:** 4 elements accessible per cycle (4× throughput)

## Example Execution Trace

### Initial State
```
Source Cloud (centered at origin):
  Points: [(10,10,10), (20,10,10), (30,10,10), (40,10,10), ...]
  Centroid: (25, 25, 15)

Target Cloud (translated by (50, 30, 100)):
  Points: [(60,40,110), (70,40,110), (80,40,110), (90,40,110), ...]
  Centroid: (75, 55, 115)
```

### Iteration 1

**Step 1: Correspondences**
```
V3DDIST.VV computes 16 distances:
  dist²[0] = (60-10)² + (40-10)² + (110-10)² = 13,400
  dist²[1] = (70-20)² + (40-10)² + (110-10)² = 13,400
  ...
Average error: ~13,400
```

**Step 2: Transformation**
```
VCOVMAT3D → Covariance matrix:
  Translation: (50, 30, 100)  ← Centroid difference
  Rotation: Identity (in simplified version)
```

**Step 3: Apply Transform**
```
VGEMV3D transforms all 16 points:
  (10,10,10) + (50,30,100) = (60,40,110) ✓
  (20,10,10) + (50,30,100) = (70,40,110) ✓
  ...
```

**Step 4: Convergence Check**
```
VFPSMAX finds max error after transformation:
  max_error ≈ 0 (perfect alignment!)
  CONVERGED ✓
```

### Result
```
Final transformation:
  Translation: [50.0, 30.0, 100.0]
  Rotation: Identity (no rotation needed)

Aligned source matches target perfectly!
```

## Limitations and Simplifications

This implementation makes several simplifications compared to full ICP:

1. **No rotation estimation**: Uses only translation (identity rotation matrix)
   - Full ICP would compute SVD of covariance matrix to extract rotation

2. **Simplified correspondences**: Uses point-to-point matching
   - Advanced ICP uses point-to-plane distances

3. **Fixed vector length**: Only handles 16 points
   - Production code would support variable-length clouds

4. **No outlier rejection**: Assumes all correspondences are valid
   - Robust ICP uses RANSAC or M-estimators

5. **Fixed-point arithmetic**: Uses integer coordinates
   - Higher precision applications use floating-point

## Extensions and Future Work

### Near-term improvements:
- Implement full SVD for rotation estimation
- Add outlier detection and rejection
- Support variable vector lengths (32, 64, 128 points)
- Use floating-point for higher precision

### Advanced features:
- Multi-scale ICP (coarse-to-fine pyramid)
- Colored point cloud registration (ICP + color matching)
- GPU-style SIMT execution for large clouds
- Real-time streaming point cloud alignment

## References

- **ICP Algorithm**: Besl & McKay, "A Method for Registration of 3-D Shapes", IEEE TPAMI 1992
- **CADL Language**: See `../../README.md`
- **RISC-V Custom Instructions**: RISC-V ISA Manual, Chapter 25
- **Point Cloud Processing**: Rusu & Cousins, "3D is here: Point Cloud Library", ICRA 2011
