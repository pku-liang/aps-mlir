#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
uint8_t v3ddist_vs(uint32_t *rs1, uint32_t *rs2) {
    uint8_t rd_result = 0;
    uint32_t * addr = rs1;
    uint32_t * out_addr = rs2;
    uint32_t vl = 16;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t ref_x = rs1[48];
    uint32_t ref_y = rs1[49];
    uint32_t ref_z = rs1[50];
    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x = rs1[i];
        uint32_t y = rs1[16 + i];
        uint32_t z = rs1[32 + i];
        uint32_t dx = (x - ref_x);
        uint32_t dy = (y - ref_y);
        uint32_t dz = (z - ref_z);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        rs2[i] = dist_sq;
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
volatile uint32_t input_data[64] __attribute__((aligned(128))) = {
    // points_x[16] - X coordinates
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,

    // points_y[16] - Y coordinates
    5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,

    // points_z[16] - Z coordinates
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
    1500, 1600,

    // Reference point (at offset 48, 49, 50 in array = byte offset 192, 196,
    // 200)
    50, 50, 500, // ref_x=50, ref_y=50, ref_z=500

    // Padding
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Output buffer for distances
volatile uint32_t output_data[16] __attribute__((aligned(128))) = {0};

int main(void) {
  // printf("V3DDIST.VS Test - Vector-Scalar 3D Distance Squared\n");
  // printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  // printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char *march = get_march(marchid);
  // printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  // for (int i = 0; i < 10; i++) {
  asm volatile("fence" ::: "memory");
    result = v3ddist_vs((uint32_t*)input_data,
               (uint32_t*)output_data);
  asm volatile("fence" ::: "memory");
  // }

  // printf("Returned rd value: %u\n", result);
  // printf("\nReference point: (%u, %u, %u)\n", input_data[48], input_data[49],
  //        input_data[50]);

  // printf("\nComputed distances squared:\n");
  // for (int i = 0; i < 16; i++) {
  //   uint32_t x = input_data[i];
  //   uint32_t y = input_data[16 + i];
  //   uint32_t z = input_data[32 + i];

  //   printf("Point %2d: (%3u, %3u, %4u) -> dist² = %10u\n", i, x, y, z,
  //          output_data[i]);
  // }

  // // Verify first few results manually
  // printf("\nVerification (manual calculation):\n");
  // for (int i = 0; i < 3; i++) {
  //   uint32_t x = input_data[i];
  //   uint32_t y = input_data[16 + i];
  //   uint32_t z = input_data[32 + i];
  //   uint32_t ref_x = input_data[48];
  //   uint32_t ref_y = input_data[50];
  //   uint32_t ref_z = input_data[52];

  //   int32_t dx = (int32_t)(x - ref_x);
  //   int32_t dy = (int32_t)(y - ref_y);
  //   int32_t dz = (int32_t)(z - ref_z);
  //   uint32_t expected = (uint32_t)(dx * dx + dy * dy + dz * dz);

  //   printf("Point %d: expected=%u, got=%u %s\n", i, expected, output_data[i],
  //          expected == output_data[i] ? "✓" : "✗");
  // }

  return 0;
}
