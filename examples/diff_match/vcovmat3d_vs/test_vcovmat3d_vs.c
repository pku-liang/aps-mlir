#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>

// __attribute__((always_inline))
// uint32_t vcovmat3d_vs(uint32_t rs1, uint32_t rs2) {
//   uint32_t rd = 0;
//   asm volatile(".insn r 0x2B, 0b111, 0x2B, %0, %1, %2"
//                : "=r"(rd) : "r"(rs1), "r"(rs2));
//   return rd;
// }
#pragma megg optimize
uint8_t vcovmat3d_vs(int32_t *rs1, int32_t *rs2) {
    uint8_t rd_result = 0;
    int32_t * points_addr = rs1;
    int32_t * out_addr = rs2;
    int32_t cx = rs1[48];
    int32_t cy = rs1[49];
    int32_t cz = rs1[50];
    rs2[0] = 0;
    rs2[1] = 0;
    rs2[2] = 0;
    rs2[3] = 0;
    rs2[4] = 0;
    rs2[5] = 0;
    rs2[6] = 16;
    rs2[7] = 0;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 16; ++i) {
        int32_t x = rs1[i];
        int32_t y = rs1[16 + i];
        int32_t z = rs1[32 + i];
        int32_t dx = (x - cx);
        int32_t dy = (y - cy);
        int32_t dz = (z - cz);
        rs2[0] = (rs2[0] + (dx * dx));
        rs2[1] = (rs2[1] + (dx * dy));
        rs2[2] = (rs2[2] + (x * dz - cx * dz));
        rs2[3] = (rs2[3] + (dy * dy));
        rs2[4] = (rs2[4] + (dy * dz));
        rs2[5] = (rs2[5] + z * z + cz * cz - (z * cz << 1));
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 1;
    return rd_result;
}


// Input: 16 3D points (X[16] | Y[16] | Z[16]) + centroid (cx, cy, cz)
volatile int32_t input_data[51] __attribute__((aligned(128))) = {0};

// Output: covariance matrix (c00, c01, c02, c11, c12, c22, count, reserved)
volatile int32_t output_data[8] __attribute__((aligned(128))) = {0};

void print_covariance(const int32_t* cov) {
  printf("Covariance matrix:\n");
  printf("  [%10d %10d %10d]\n", cov[0], cov[1], cov[2]);
  printf("  [%10d %10d %10d]\n", cov[1], cov[3], cov[4]);
  printf("  [%10d %10d %10d]\n", cov[2], cov[4], cov[5]);
}

int main(void) {
  printf("VCOVMAT3D.VS Test\n");
  printf("========================================\n");

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Setup 16 points forming a cube (0,0,0) to (10,10,10)
  int32_t points[16][3] = {
    {0, 0, 0}, {10, 0, 0}, {0, 10, 0}, {10, 10, 0},
    {0, 0, 10}, {10, 0, 10}, {0, 10, 10}, {10, 10, 10},
    {0, 0, 0}, {10, 0, 0}, {0, 10, 0}, {10, 10, 0},
    {0, 0, 10}, {10, 0, 10}, {0, 10, 10}, {10, 10, 10}
  };

  // Pack into SOA format
  for (int i = 0; i < 16; i++) {
    input_data[i] = points[i][0];      // X
    input_data[16 + i] = points[i][1]; // Y
    input_data[32 + i] = points[i][2]; // Z
  }

  // Centroid: (5, 5, 5)
  input_data[48] = 5;
  input_data[49] = 5;
  input_data[50] = 5;

  printf("Point cloud: 16 cube vertices\n");
  printf("Centroid: (%d, %d, %d)\n\n", input_data[48], input_data[49], input_data[50]);

  // Call custom instruction
  printf("Computing covariance...\n");
  vcovmat3d_vs((uint32_t*)input_data,
               (uint32_t*)output_data);

  print_covariance((const int32_t*)output_data);

  printf("\n========================================\n");
  printf("Test Complete!\n");
  printf("========================================\n");

  return 0;
}
