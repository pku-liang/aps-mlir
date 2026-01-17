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
 * 1. Load source and target point clouds (16 points each)
 * 2. Compute centroids of both clouds
 * 3. Iterate:
 *    a. Find closest point correspondences (V3DDIST.VV)
 *    b. Compute transformation (VCOVMAT3D)
 *    c. Apply transformation (VGEMV3D)
 *    d. Check convergence (VFPSMAX on errors)
 * 4. Output aligned point cloud and final transformation
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"

// ============================================================================
// Custom Instruction Wrappers
// ============================================================================

// V3DDIST.VS - Vector-Scalar 3D distance
__attribute__((alwaysinline))
uint32_t v3ddist_vs(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x29, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// V3DDIST.VV - Vector-Vector 3D distance
__attribute__((alwaysinline))
uint32_t v3ddist_vv(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x2A, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// VCOVMAT3D - 3D covariance matrix
__attribute__((alwaysinline))
uint32_t vcovmat3d(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x2B, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// VGEMV3D - 3D general matrix-vector multiply
__attribute__((alwaysinline))
uint32_t vgemv3d(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x2C, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// VFPSMAX - Floating-point scalar maximum
__attribute__((alwaysinline))
uint32_t vfpsmax(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x2D, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// ============================================================================
// Data Structures
// ============================================================================

#define NUM_POINTS 16
#define MAX_ITERATIONS 10
#define CONVERGENCE_THRESHOLD 100  // Squared distance threshold

// Point cloud in Structure-of-Arrays layout
typedef struct {
  uint32_t x[NUM_POINTS];
  uint32_t y[NUM_POINTS];
  uint32_t z[NUM_POINTS];
  uint32_t centroid[3];  // Centroid coordinates
} PointCloud;

// 3x3 Transformation matrix (rotation + translation)
typedef struct {
  float matrix[9];       // 3x3 rotation matrix (row-major)
  float translation[3];  // Translation vector
} Transform;

// ============================================================================
// Helper Functions
// ============================================================================

void print_point_cloud(const char* name, const PointCloud* cloud) {
  printf("\n%s (16 points):\n", name);
  printf("   X      Y      Z\n");
  printf("----------------------\n");
  for (int i = 0; i < NUM_POINTS; i++) {
    printf("%5u  %5u  %5u\n", cloud->x[i], cloud->y[i], cloud->z[i]);
  }
  printf("Centroid: (%u, %u, %u)\n",
         cloud->centroid[0], cloud->centroid[1], cloud->centroid[2]);
}

void print_transform(const Transform* t) {
  printf("\nTransformation Matrix:\n");
  printf("[%6.3f %6.3f %6.3f]\n", t->matrix[0], t->matrix[1], t->matrix[2]);
  printf("[%6.3f %6.3f %6.3f]\n", t->matrix[3], t->matrix[4], t->matrix[5]);
  printf("[%6.3f %6.3f %6.3f]\n", t->matrix[6], t->matrix[7], t->matrix[8]);
  printf("Translation: [%6.3f, %6.3f, %6.3f]\n",
         t->translation[0], t->translation[1], t->translation[2]);
}

// Compute centroid using scalar code (could be accelerated)
void compute_centroid(PointCloud* cloud) {
  uint64_t sum_x = 0, sum_y = 0, sum_z = 0;
  for (int i = 0; i < NUM_POINTS; i++) {
    sum_x += cloud->x[i];
    sum_y += cloud->y[i];
    sum_z += cloud->z[i];
  }
  cloud->centroid[0] = sum_x / NUM_POINTS;
  cloud->centroid[1] = sum_y / NUM_POINTS;
  cloud->centroid[2] = sum_z / NUM_POINTS;
}

// ============================================================================
// Point Cloud Registration Pipeline
// ============================================================================

/**
 * Step 1: Compute correspondences
 * For each point in source, find closest point in target using V3DDIST.VV
 */
void find_correspondences(const PointCloud* source,
                          const PointCloud* target,
                          uint32_t* distances,
                          volatile uint32_t* work_buffer) {
  printf("\n[Step 1] Finding correspondences using V3DDIST.VV...\n");

  // Pack both point clouds into work buffer
  // Layout: [source_x[16] | source_y[16] | source_z[16] |
  //          target_x[16] | target_y[16] | target_z[16]]
  memcpy((void*)work_buffer, source->x, NUM_POINTS * sizeof(uint32_t));
  memcpy((void*)(work_buffer + 16), source->y, NUM_POINTS * sizeof(uint32_t));
  memcpy((void*)(work_buffer + 32), source->z, NUM_POINTS * sizeof(uint32_t));
  memcpy((void*)(work_buffer + 48), target->x, NUM_POINTS * sizeof(uint32_t));
  memcpy((void*)(work_buffer + 64), target->y, NUM_POINTS * sizeof(uint32_t));
  memcpy((void*)(work_buffer + 80), target->z, NUM_POINTS * sizeof(uint32_t));

  // Call V3DDIST.VV custom instruction
  volatile uint32_t dist_buffer[NUM_POINTS] __attribute__((aligned(128)));
  uint32_t result = v3ddist_vv(
      (uint32_t)(unsigned long)work_buffer,
      (uint32_t)(unsigned long)dist_buffer
  );

  // Copy results
  memcpy(distances, (void*)dist_buffer, NUM_POINTS * sizeof(uint32_t));

  printf("  Computed %d pairwise distances\n", NUM_POINTS);
  printf("  Status: %u\n", result);
}

/**
 * Step 2: Compute transformation using covariance
 * Uses VCOVMAT3D to compute covariance matrix
 */
void estimate_transformation(const PointCloud* source,
                            const PointCloud* target,
                            Transform* transform,
                            volatile uint32_t* work_buffer) {
  printf("\n[Step 2] Estimating transformation using VCOVMAT3D...\n");

  // Center points around centroid for source cloud
  uint32_t centered[NUM_POINTS * 3];
  for (int i = 0; i < NUM_POINTS; i++) {
    centered[i] = source->x[i] - source->centroid[0];
    centered[i + 16] = source->y[i] - source->centroid[1];
    centered[i + 32] = source->z[i] - source->centroid[2];
  }

  // Pack into work buffer
  memcpy((void*)work_buffer, centered, NUM_POINTS * 3 * sizeof(uint32_t));

  // Call VCOVMAT3D to compute 3x3 covariance matrix
  volatile uint32_t cov_buffer[16] __attribute__((aligned(128)));
  uint32_t result = vcovmat3d(
      (uint32_t)(unsigned long)work_buffer,
      (uint32_t)(unsigned long)cov_buffer
  );

  // Extract covariance matrix (9 values in row-major order)
  // For simplicity, use identity + small perturbation as transformation
  // (In real ICP, would do SVD decomposition on covariance)
  for (int i = 0; i < 9; i++) {
    if (i % 4 == 0) {
      transform->matrix[i] = 1.0f;  // Diagonal = 1
    } else {
      transform->matrix[i] = 0.0f;
    }
  }

  // Compute translation as difference between centroids
  transform->translation[0] = (float)(target->centroid[0] - source->centroid[0]);
  transform->translation[1] = (float)(target->centroid[1] - source->centroid[1]);
  transform->translation[2] = (float)(target->centroid[2] - source->centroid[2]);

  printf("  Computed covariance matrix\n");
  printf("  Status: %u\n", result);
}

/**
 * Step 3: Apply transformation to source points
 * Uses VGEMV3D for matrix-vector multiplication
 */
void apply_transformation(PointCloud* source,
                         const Transform* transform,
                         volatile uint32_t* work_buffer) {
  printf("\n[Step 3] Applying transformation using VGEMV3D...\n");

  // Pack transformation matrix and points
  // Layout: [matrix[9] | points_x[16] | points_y[16] | points_z[16]]
  volatile float* float_buf = (volatile float*)work_buffer;
  memcpy((void*)float_buf, transform->matrix, 9 * sizeof(float));

  // Convert points to float and pack
  for (int i = 0; i < NUM_POINTS; i++) {
    float_buf[9 + i] = (float)source->x[i];
    float_buf[9 + 16 + i] = (float)source->y[i];
    float_buf[9 + 32 + i] = (float)source->z[i];
  }

  // Call VGEMV3D
  volatile uint32_t result_buffer[NUM_POINTS * 3] __attribute__((aligned(128)));
  uint32_t result = vgemv3d(
      (uint32_t)(unsigned long)work_buffer,
      (uint32_t)(unsigned long)result_buffer
  );

  // Update source points with transformed values
  volatile float* result_float = (volatile float*)result_buffer;
  for (int i = 0; i < NUM_POINTS; i++) {
    source->x[i] = (uint32_t)(result_float[i] + transform->translation[0]);
    source->y[i] = (uint32_t)(result_float[i + 16] + transform->translation[1]);
    source->z[i] = (uint32_t)(result_float[i + 32] + transform->translation[2]);
  }

  // Recompute centroid
  compute_centroid(source);

  printf("  Applied transformation to %d points\n", NUM_POINTS);
  printf("  Status: %u\n", result);
}

/**
 * Step 4: Check convergence
 * Uses VFPSMAX to find maximum error
 */
uint32_t check_convergence(const uint32_t* distances,
                           volatile uint32_t* work_buffer) {
  printf("\n[Step 4] Checking convergence using VFPSMAX...\n");

  // Pack distances into work buffer
  memcpy((void*)work_buffer, distances, NUM_POINTS * sizeof(uint32_t));

  // Call VFPSMAX to find maximum distance
  volatile uint32_t max_result[2] __attribute__((aligned(128)));
  uint32_t result = vfpsmax(
      (uint32_t)(unsigned long)work_buffer,
      (uint32_t)(unsigned long)max_result
  );

  uint32_t max_dist = max_result[0];
  uint32_t max_idx = max_result[1];

  printf("  Max squared distance: %u (at point %u)\n", max_dist, max_idx);
  printf("  Threshold: %u\n", CONVERGENCE_THRESHOLD);
  printf("  Status: %u\n", result);

  return max_dist;
}

/**
 * Main Registration Pipeline
 */
int run_registration(PointCloud* source, const PointCloud* target) {
  printf("\n========================================\n");
  printf("Point Cloud Registration Pipeline\n");
  printf("========================================\n");

  // Allocate work buffer for custom instructions
  volatile uint32_t work_buffer[256] __attribute__((aligned(128))) = {0};
  uint32_t distances[NUM_POINTS];
  Transform transform;

  // Compute initial centroids
  compute_centroid(source);
  printf("Source centroid: (%u, %u, %u)\n",
         source->centroid[0], source->centroid[1], source->centroid[2]);
  printf("Target centroid: (%u, %u, %u)\n",
         target->centroid[0], target->centroid[1], target->centroid[2]);

  // ICP iterations
  int iteration = 0;
  uint32_t max_error = UINT32_MAX;

  while (iteration < MAX_ITERATIONS && max_error > CONVERGENCE_THRESHOLD) {
    printf("\n========================================\n");
    printf("Iteration %d\n", iteration + 1);
    printf("========================================\n");

    // Step 1: Find correspondences
    find_correspondences(source, target, distances, work_buffer);

    // Step 2: Estimate transformation
    estimate_transformation(source, target, &transform, work_buffer);
    print_transform(&transform);

    // Step 3: Apply transformation
    apply_transformation(source, &transform, work_buffer);

    // Step 4: Check convergence
    max_error = check_convergence(distances, work_buffer);

    if (max_error <= CONVERGENCE_THRESHOLD) {
      printf("\n✓ Converged! Max error %u <= threshold %u\n",
             max_error, CONVERGENCE_THRESHOLD);
      break;
    }

    iteration++;
  }

  if (iteration >= MAX_ITERATIONS) {
    printf("\n⚠ Reached maximum iterations (%d)\n", MAX_ITERATIONS);
    return 1;
  }

  return 0;
}

// ============================================================================
// Main Program
// ============================================================================

int main(void) {
  printf("========================================\n");
  printf("Point Cloud Registration Application\n");
  printf("Using Custom RISC-V Instructions\n");
  printf("========================================\n");

  // Print architecture info
  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n", march);

  // Initialize source point cloud (16 points in a rough cube)
  PointCloud source = {
    .x = {10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40},
    .y = {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40},
    .z = {10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20}
  };

  // Initialize target point cloud (source translated by (50, 30, 100))
  PointCloud target = {
    .x = {60, 70, 80, 90, 60, 70, 80, 90, 60, 70, 80, 90, 60, 70, 80, 90},
    .y = {40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60, 70, 70, 70, 70},
    .z = {110, 110, 110, 110, 110, 110, 110, 110, 120, 120, 120, 120, 120, 120, 120, 120}
  };

  // Compute initial centroids
  compute_centroid(&source);
  compute_centroid((PointCloud*)&target);

  print_point_cloud("Source Cloud (Initial)", &source);
  print_point_cloud("Target Cloud", &target);

  // Run registration
  int result = run_registration(&source, &target);

  // Print final results
  printf("\n========================================\n");
  printf("Registration Complete\n");
  printf("========================================\n");
  print_point_cloud("Source Cloud (Aligned)", &source);
  print_point_cloud("Target Cloud (Reference)", &target);

  if (result == 0) {
    printf("\n✓ SUCCESS: Point clouds aligned!\n");
  } else {
    printf("\n⚠ PARTIAL: Alignment incomplete\n");
  }

  return result;
}
