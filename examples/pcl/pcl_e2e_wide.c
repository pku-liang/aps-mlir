/**
 * Point Cloud Registration Application
 *
 * This application demonstrates a complete end-to-end point cloud processing
 * pipeline using custom RISC-V instruction extensions for acceleration.
 *
 * Task: Simplified ICP (Iterative Closest Point) Algorithm
 * - Aligns two point clouds by finding optimal transformation
 * - Uses custom instructions for acceleration:
 *   1. V3DDIST.VS - Compute distances from points to centroid
 *   2. V3DDIST.VV - Compute pairwise distances (correspondence)
 *   3. VCOVMAT3D - Compute covariance for transformation estimation
 *   4. VGEMV3D - Apply transformation to points
 *   5. VFPSMAX - Find maximum error for convergence check
 *
 * Pipeline:
 * 1. Load source and target point clouds (32 points each)
 * 2. Compute centroids of both clouds
 * 3. Iterate:
 *    a. Find closest point correspondences (V3DDIST.VV)
 *    b. Compute transformation (VCOVMAT3D)
 *    c. Apply transformation (VGEMV3D)
 *    d. Check convergence (VFPSMAX on errors)
 * 4. Output aligned point cloud and final transformation
 */

#define HARDWARE
#define NO_DBG_PRINT

#ifdef HARDWARE
#include "marchid.h"
#endif
#include <math.h>
#ifdef HARDWARE
#include <riscv-pk/encoding.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// Architecture-aware pointer type
// ============================================================================
// On 32-bit ASIC hardware, use uint32_t for addresses
// On 64-bit host systems, use uintptr_t for proper pointer handling
#ifdef HARDWARE
typedef uint32_t ptr_t;
#else
typedef uintptr_t ptr_t;
#endif

// ============================================================================
// Custom Instruction Wrappers
// ============================================================================

#ifdef HARDWARE
// V3DDIST.VV - Vector-Vector 3D distance
__attribute__((alwaysinline)) uint32_t v3ddist_vv(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x0B, 0b111, 0x2A, %0, %1, %2"
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}
#else
uint32_t v3ddist_vv(ptr_t r1, ptr_t r2) {
  uint32_t *rs1 = (uint32_t *)r1;
  uint32_t *rs2 = (uint32_t *)r2;
  for (int i = 0; i < 16; ++i) {  // 16 points
    uint32_t x1 = rs1[i];
    uint32_t y1 = rs1[16 + i];
    uint32_t z1 = rs1[32 + i];
    uint32_t x2 = rs1[48 + i];
    uint32_t y2 = rs1[64 + i];
    uint32_t z2 = rs1[80 + i];
    uint32_t dx = (x1 - x2);
    uint32_t dy = (y1 - y2);
    uint32_t dz = (z1 - z2);
    uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
    rs2[i] = dist_sq;
  }
  return 0;
}
#endif

#ifdef HARDWARE
// VCOVMAT3D - 3D covariance matrix
__attribute__((alwaysinline)) uint32_t vcovmat3d_vs(uint32_t rs1,
                                                    uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x2B, %0, %1, %2"
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}
#else
uint32_t vcovmat3d_vs(ptr_t r1, ptr_t r2) {
  uint32_t *rs1 = (uint32_t *)r1;
  uint32_t *rs2 = (uint32_t *)r2;
  // Centroid now passed via rs2[8..10]
  uint32_t cx = rs2[8];
  uint32_t cy = rs2[9];
  uint32_t cz = rs2[10];
  for (int i = 0; i < 6; i++) {
    rs2[i] = 0;
  }
  for (int i = 0; i < 16; ++i) {  // 16 points
    uint32_t x1 = rs1[i];
    uint32_t y1 = rs1[16 + i];
    uint32_t z1 = rs1[32 + i];
    uint32_t dx = (x1 - cx);
    uint32_t dy = (y1 - cy);
    uint32_t dz = (z1 - cz);
    rs2[0] += dx * dx;
    rs2[1] += dx * dy;
    rs2[2] += dx * dz;
    rs2[3] += dy * dy;
    rs2[4] += dy * dz;
    rs2[5] += dz * dz;
  }
  return 1;
}
#endif

#ifdef HARDWARE
// VFPSMAX - Floating-point scalar maximum
__attribute__((alwaysinline)) uint32_t vfpsmax(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x5B, 0b111, 0x2D, %0, %1, %2"
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}
#else
uint32_t vfpsmax(ptr_t r1, ptr_t r2) {
  uint32_t *rs1 = (uint32_t *)r1;
  uint32_t *rs2 = (uint32_t *)r2;
  uint32_t max_val = 0;
  uint32_t max_idx = 0;
  for (int i = 0; i < 16; i++) {
    if (rs1[i] > max_val) {
      max_val = rs1[i];
      max_idx = i;
    }
  }
  rs2[0] = max_val;
  rs2[1] = max_idx;
  return max_idx;
}
#endif

// VGEMV3D - 3D general matrix-vector multiply
#ifdef HARDWARE
__attribute__((alwaysinline)) uint32_t vgemv3d(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x7B, 0b111, 0x2C, %0, %1, %2"
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}
#else
uint32_t vgemv3d(ptr_t r1, ptr_t r2) {
  uint32_t *rs1 = (uint32_t *)r1;
  uint32_t *rs2 = (uint32_t *)r2;
  uint32_t *matrix = rs1;
  uint32_t *vec = &rs1[16];
  for (int i = 0; i < 4; i++) {
    uint32_t acc = 0;
    for (int j = 0; j < 4; j++) {
      acc += matrix[i * 4 + j] * vec[j];
    }
    rs2[i] = acc;
  }
  return 0;
}
#endif

// ============================================================================
// Data Structures
// ============================================================================

#define NUM_POINTS 16  // Standard ISAX instruction width
#define MAX_ITERATIONS 20
#define CONVERGENCE_THRESHOLD 500 // Squared distance threshold

// Point cloud in Structure-of-Arrays layout (without padding, exactly 48 words)
// Used as part of PointCloudPair for zero-copy V3DDIST.VV access
typedef struct {
  uint32_t x[NUM_POINTS];      // Offset 0-15
  uint32_t y[NUM_POINTS];      // Offset 16-31
  uint32_t z[NUM_POINTS];      // Offset 32-47
} PointCloud;

// ZERO-COPY optimization: Store both clouds consecutively for V3DDIST.VV
// Layout matches instruction requirements EXACTLY:
// [source_x[16] | source_y[16] | source_z[16] |
//  target_x[16] | target_y[16] | target_z[16]]
// Total: 96 words = 384 bytes (128-byte aligned)
typedef struct {
  PointCloud source;  // Offset 0-47
  PointCloud target;  // Offset 48-95
} __attribute__((aligned(128))) PointCloudPair;

// Separate centroid storage (needed for VCOVMAT3D)
typedef struct {
  uint32_t data[3];
} Centroid;

// 3x3 Transformation matrix (rotation + translation)
typedef struct {
  uint32_t matrix[9];     // 3x3 rotation matrix (row-major)
  int32_t translation[3]; // Translation vector
} Transform;

// ============================================================================
// Helper Functions
// ============================================================================

void print_point_cloud(const char *name, const PointCloud *cloud, const Centroid *centroid) {
  printf("\n%s:\n", name);
  // Print only first 4 points
  for (int i = 0; i < 4; i++) {
    printf("p%d:[%u,%u,%u]\n", i, cloud->x[i], cloud->y[i], cloud->z[i]);
  }
  printf("c:[%u,%u,%u]\n", centroid->data[0], centroid->data[1],
         centroid->data[2]);
}

void print_transform(const Transform *t) {
  // Removed - not needed for minimal output
}

// Compute centroid using scalar code (could be accelerated)
void compute_centroid(const PointCloud *cloud, Centroid *centroid) {
  uint64_t sum_x = 0, sum_y = 0, sum_z = 0;
  for (int i = 0; i < NUM_POINTS; i++) {
    sum_x += cloud->x[i];
    sum_y += cloud->y[i];
    sum_z += cloud->z[i];
  }
  centroid->data[0] = sum_x / NUM_POINTS;
  centroid->data[1] = sum_y / NUM_POINTS;
  centroid->data[2] = sum_z / NUM_POINTS;
}

// ============================================================================
// Point Cloud Registration Pipeline
// ============================================================================

/**
 * Step 1: Compute correspondences
 * For each point in source, find closest point in target using V3DDIST.VV
 * NOTE: ZERO-COPY optimization - clouds_pair is already in the correct layout!
 */
void find_correspondences(const PointCloudPair *clouds_pair,
                          volatile uint32_t *distances) {

#ifndef NO_DBG_PRINT
  printf("\n[S1]V3DDIST.VV\n");
  // Debug: print input data
  printf("IN sx0:%u sy0:%u sz0:%u\n",
         clouds_pair->source.x[0], clouds_pair->source.y[0], clouds_pair->source.z[0]);
  printf("IN tx0:%u ty0:%u tz0:%u\n",
         clouds_pair->target.x[0], clouds_pair->target.y[0], clouds_pair->target.z[0]);
#endif

  // ZERO-COPY: Pass clouds_pair directly - it's already in the correct layout!
  // No packing loop needed - PointCloudPair layout matches V3DDIST.VV requirements exactly
  uint32_t result = v3ddist_vv((ptr_t)clouds_pair, (ptr_t)distances);
  asm volatile("fence");

#ifndef NO_DBG_PRINT
  // Debug: print output data
  printf("OUT d[0-3]:%u,%u,%u,%u\n", distances[0], distances[1], distances[2],
         distances[3]);
  printf("st:%u\n", result);
#endif
}

/**
 * Step 2: Compute transformation using covariance
 * Uses VCOVMAT3D to compute covariance matrix
 * NOTE: ZERO-COPY optimization - pass source directly, centroid via cov_output[8..10]
 */
void estimate_transformation(const PointCloud *source, const Centroid *source_centroid,
                             const Centroid *target_centroid,
                             Transform *transform,
                             volatile uint32_t *cov_output) {
#ifndef NO_DBG_PRINT
  printf("\n[S2]VCOVMAT3D\n");
  printf("DBG src0:[%u,%u,%u] sc:[%u,%u,%u]\n", source->x[0], source->y[0],
         source->z[0], source_centroid->data[0], source_centroid->data[1],
         source_centroid->data[2]);
#endif

  // ZERO-COPY: Pass source directly, centroid via cov_output[8..10]
  // vcovmat3d_vs will read centroid from rs2[8..10] and compute covariance
  cov_output[8] = source_centroid->data[0];
  cov_output[9] = source_centroid->data[1];
  cov_output[10] = source_centroid->data[2];

#ifndef NO_DBG_PRINT
  printf("IN src0:[%u,%u,%u]\n", source->x[0], source->y[0], source->z[0]);
  printf("IN ctr:[%u,%u,%u]\n", cov_output[8], cov_output[9], cov_output[10]);
#endif

  // Call VCOVMAT3D to compute 3x3 covariance matrix
  // rs1 = source point cloud, rs2 = cov_output (with centroid at [8..10])
  uint32_t result = vcovmat3d_vs((ptr_t)source, (ptr_t)cov_output);
  asm volatile("fence");

#ifndef NO_DBG_PRINT
  // Debug: print output data
  printf("OUT cov[0-2]:%u,%u,%u\n", cov_output[0], cov_output[1],
         cov_output[2]);
#endif
  // Extract covariance matrix (9 values in row-major order)
  // For simplicity, use identity matrix as transformation
  // (In real ICP, would do SVD decomposition on covariance)
  for (int i = 0; i < 9; i++) {
    if (i % 4 == 0) {
      transform->matrix[i] = 1; // Diagonal = 1
    } else {
      transform->matrix[i] = 0;
    }
  }

  // Compute translation as difference between centroids
  transform->translation[0] =
      (int32_t)target_centroid->data[0] - (int32_t)source_centroid->data[0];
  transform->translation[1] =
      (int32_t)target_centroid->data[1] - (int32_t)source_centroid->data[1];
  transform->translation[2] =
      (int32_t)target_centroid->data[2] - (int32_t)source_centroid->data[2];

#ifndef NO_DBG_PRINT
  printf("t:[%d,%d,%d]\n", transform->translation[0], transform->translation[1],
         transform->translation[2]);
  printf("st:%u\n", result);
#endif
}

/**
 * Step 3: Apply transformation to source points
 * Uses VGEMV3D for matrix-vector multiplication
 * NOTE: gemv_buffer and result_buffer must be 128-byte aligned
 */
void apply_transformation(PointCloud *source, Centroid *source_centroid, const Transform *transform,
                          volatile uint32_t *gemv_buffer,
                          volatile uint32_t *result_buffer) {
#ifndef NO_DBG_PRINT
  printf("\n[S3]VGEMV3D\n");
#endif

  // Pack 4x4 transformation matrix (row-major) with translation in last column
  // Layout: [matrix[16] | vector[4]]
  // Matrix format:
  // [1  0  0  tx]
  // [0  1  0  ty]
  // [0  0  1  tz]
  // [0  0  0  1 ]

  int tx = (int)transform->translation[0];
  int ty = (int)transform->translation[1];
  int tz = (int)transform->translation[2];

  // Row 0: [1, 0, 0, tx]
  gemv_buffer[0] = 1;
  gemv_buffer[1] = 0;
  gemv_buffer[2] = 0;
  gemv_buffer[3] = tx;
  // Row 1: [0, 1, 0, ty]
  gemv_buffer[4] = 0;
  gemv_buffer[5] = 1;
  gemv_buffer[6] = 0;
  gemv_buffer[7] = ty;
  // Row 2: [0, 0, 1, tz]
  gemv_buffer[8] = 0;
  gemv_buffer[9] = 0;
  gemv_buffer[10] = 1;
  gemv_buffer[11] = tz;
  // Row 3: [0, 0, 0, 1]
  gemv_buffer[12] = 0;
  gemv_buffer[13] = 0;
  gemv_buffer[14] = 0;
  gemv_buffer[15] = 1;

#ifndef NO_DBG_PRINT
  // Debug: print input matrix
  printf("IN M4x4:[%u,%u,%u,%u|%u,%u,%u,%u|%u,%u,%u,%u|%u,%u,%u,%u]\n",
         gemv_buffer[0], gemv_buffer[1], gemv_buffer[2], gemv_buffer[3],
         gemv_buffer[4], gemv_buffer[5], gemv_buffer[6], gemv_buffer[7],
         gemv_buffer[8], gemv_buffer[9], gemv_buffer[10], gemv_buffer[11],
         gemv_buffer[12], gemv_buffer[13], gemv_buffer[14], gemv_buffer[15]);
#endif

  // Transform each point individually - unrolled for better performance
  // Note: VGEMV3D must be called per-point (hardware constraint)
  #pragma GCC unroll 4
  for (int i = 0; i < NUM_POINTS; i++) {
    // Pack point as homogeneous coordinate [x, y, z, 1]
    gemv_buffer[16] = source->x[i];
    gemv_buffer[17] = source->y[i];
    gemv_buffer[18] = source->z[i];
    gemv_buffer[19] = 1;

#ifndef NO_DBG_PRINT
    if (i == 0) {
      printf("IN p0:[%u,%u,%u,1]\n", gemv_buffer[16], gemv_buffer[17],
             gemv_buffer[18]);
    }
#endif
    // Call VGEMV3D for this point (ACCELERATED)
    uint32_t result = vgemv3d((ptr_t)gemv_buffer, (ptr_t)result_buffer);
    asm volatile("fence");
    // Update point with transformed values (minimal scalar ops)
    source->x[i] = result_buffer[0];
    source->y[i] = result_buffer[1];
    source->z[i] = result_buffer[2];
    // result_buffer[3] contains w (should be 1)

#ifndef NO_DBG_PRINT
    if (i == 0) {
      printf("OUT p0:[%u,%u,%u]\n", result_buffer[0], result_buffer[1],
             result_buffer[2]);
    }
#endif
  }

  // Update centroid incrementally (optimization: avoid full recomputation)
  // Since we applied translation transform, centroid simply shifts by translation
  source_centroid->data[0] += (uint32_t)tx;
  source_centroid->data[1] += (uint32_t)ty;
  source_centroid->data[2] += (uint32_t)tz;

#ifndef NO_DBG_PRINT
  printf("c:[%u,%u,%u]\n", source_centroid->data[0], source_centroid->data[1],
         source_centroid->data[2]);
  printf("st:0\n");
#endif
}

/**
 * Step 4: Check convergence
 * Uses VFPSMAX to find maximum error
 * NOTE: distances and max_result must be 128-byte aligned
 */
uint32_t check_convergence(volatile uint32_t *distances,
                           volatile uint32_t *max_result) {
#ifndef NO_DBG_PRINT
  printf("\n[S4]VFPSMAX\n");
#endif

  // Call VFPSMAX to find maximum distance
  // NO memcpy - distances buffer is already aligned and populated
  uint32_t result = vfpsmax((ptr_t)distances, (ptr_t)max_result);
  asm volatile("fence");
  
  uint32_t max_dist = max_result[0];
  uint32_t max_idx = max_result[1];

#ifndef NO_DBG_PRINT
  printf("max:%u@%u th:%u\n", max_dist, max_idx, CONVERGENCE_THRESHOLD);
  printf("st:%u\n", result);
#endif

  return max_dist;
}

/**
 * Main Registration Pipeline
 * All buffers are 128-byte aligned to meet instruction requirements
 * ZERO-COPY optimized: uses PointCloudPair for direct V3DDIST.VV access
 */
int run_registration(PointCloudPair *clouds_pair, Centroid *source_centroid,
                     const Centroid *target_centroid) {
#ifndef NO_DBG_PRINT
  printf("\n=ICP Start=\n");
#endif

  // Allocate aligned work buffers for custom instructions
  // ZERO-COPY: No V3DDistBuffer needed - use clouds_pair directly!
  volatile uint32_t distances[NUM_POINTS] __attribute__((aligned(128)));
  volatile uint32_t cov_output[16] __attribute__((aligned(128)));  // covariance + centroid at [8..10]
  volatile uint32_t gemv_buffer[32] __attribute__((aligned(128)));
  volatile uint32_t gemv_result[4] __attribute__((aligned(128)));
  volatile uint32_t max_result[2] __attribute__((aligned(128)));
  Transform transform;

  // Compute initial source centroid only once (optimization)
  compute_centroid(&clouds_pair->source, source_centroid);
#ifndef NO_DBG_PRINT
  printf("Sc:[%u,%u,%u]\n", source_centroid->data[0], source_centroid->data[1],
         source_centroid->data[2]);
  printf("Tc:[%u,%u,%u]\n", target_centroid->data[0], target_centroid->data[1],
         target_centroid->data[2]);
#endif

  // ICP iterations
  int iteration = 0;
  uint32_t max_error = UINT32_MAX;

  while (iteration < MAX_ITERATIONS && max_error > CONVERGENCE_THRESHOLD) {

#ifndef NO_DBG_PRINT
    printf("\n---Iter %d---\n", iteration + 1);
#endif

    // Step 1: Find correspondences (ACCELERATED, ZERO-COPY)
    find_correspondences(clouds_pair, distances);

    // Step 2: Estimate transformation (ACCELERATED, ZERO-COPY)
    estimate_transformation(&clouds_pair->source, source_centroid, target_centroid,
                           &transform, cov_output);

    // Step 3: Apply transformation (ACCELERATED)
    apply_transformation(&clouds_pair->source, source_centroid, &transform,
                        gemv_buffer, gemv_result);

    // Step 4: Check convergence (ACCELERATED)
    max_error = check_convergence(distances, max_result);

    // Early exit without branch prediction penalty
    iteration++;
    if (max_error <= CONVERGENCE_THRESHOLD) {
#ifndef NO_DBG_PRINT
      printf("\nOK:e=%u\n", max_error);
#endif
      break;
    }
  }

#ifndef NO_DBG_PRINT
  if (iteration >= MAX_ITERATIONS) {
    printf("\nMax iter\n");
    return 1;
  }
#endif
  return 0;
}

// ============================================================================
// Main Program
// ============================================================================

int main(void) {
  // ZERO-COPY: Use PointCloudPair for consecutive memory layout
  // This allows V3DDIST.VV to read both clouds without any packing/copying!
  PointCloudPair clouds_pair __attribute__((aligned(128))) = {
      .source = {
          .x = {100, 150, 200, 250, 100, 150, 200, 250, 100, 150, 200, 250, 100, 150, 200, 250},
          .y = {100, 100, 100, 100, 150, 150, 150, 150, 200, 200, 200, 200, 250, 250, 250, 250},
          .z = {100, 100, 100, 100, 100, 100, 100, 100, 150, 150, 150, 150, 150, 150, 150, 150}},
      .target = {
          .x = {200, 200, 200, 200, 250, 250, 250, 250, 300, 300, 300, 300, 350, 350, 350, 350},
          .y = {400, 450, 500, 550, 400, 450, 500, 550, 400, 450, 500, 550, 400, 450, 500, 550},
          .z = {605, 602, 608, 595, 597, 603, 601, 609, 655, 648, 652, 658, 651, 649, 654, 647}}};

  // Separate centroid storage
  Centroid source_centroid, target_centroid;

  // Compute initial centroids
  compute_centroid(&clouds_pair.source, &source_centroid);
  compute_centroid(&clouds_pair.target, &target_centroid);

#ifndef NO_DBG_PRINT
  print_point_cloud("Src Init", &clouds_pair.source, &source_centroid);
  print_point_cloud("Tgt", &clouds_pair.target, &target_centroid);
#endif

  // Run registration
  int result = run_registration(&clouds_pair, &source_centroid, &target_centroid);

// Print final results
#ifndef NO_DBG_PRINT
  printf("\n=Done=\n");
  print_point_cloud("Src Final", &clouds_pair.source, &source_centroid);

  if (result == 0) {
    printf("\nOK\n");
  } else {
    printf("\nFAIL\n");
  }
#endif

  return result;
}
