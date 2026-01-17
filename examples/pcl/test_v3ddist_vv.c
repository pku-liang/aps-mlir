#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>

__attribute__((alwaysinline))
uint32_t v3ddist_vv(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(
  ".insn r 0x0B, 0b111, 0x28, %0, %1, %2"  // opcode=0x2B (0101011), funct7=0x28 (0101000)
  : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data: 2 sets of 16 3D points in Structure-of-Arrays layout
// Memory layout:
// - points1_x[16] at offset 0
// - points1_y[16] at offset 64 (16*4 bytes)
// - points1_z[16] at offset 128 (32*4 bytes)
// - points2_x[16] at offset 192 (48*4 bytes)
// - points2_y[16] at offset 256 (64*4 bytes)
// - points2_z[16] at offset 320 (80*4 bytes)
volatile uint32_t input_data[96] __attribute__((aligned(128))) = {

  // points2_x[16]
  15, 25, 35, 45, 55, 65, 75, 85,
  95, 105, 115, 125, 135, 145, 155, 165,

  // points2_y[16]
  10, 20, 30, 40, 50, 60, 70, 80,
  90, 100, 110, 120, 130, 140, 150, 160,

  // points2_z[16]
  110, 210, 310, 410, 510, 610, 710, 810,
  910, 1010, 1110, 1210, 1310, 1410, 1510, 1610,
  
  // points1_x[16]
  10, 20, 30, 40, 50, 60, 70, 80,
  90, 100, 110, 120, 130, 140, 150, 160,

  // points1_y[16]
  5, 15, 25, 35, 45, 55, 65, 75,
  85, 95, 105, 115, 125, 135, 145, 155,

  // points1_z[16]
  100, 200, 300, 400, 500, 600, 700, 800,
  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
};

// Output buffer for distances
volatile uint32_t output_data[16] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("V3DDIST.VV Test - Vector-Vector 3D Distance Squared\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = v3ddist_vv((uint32_t)(unsigned long)input_data,
               (uint32_t)(unsigned long)output_data);
  }
  printf("Returned rd value: %u\n", result);

  printf("\nComputed pairwise distances squared:\n");
  for (int i = 0; i < 16; i++) {
    uint32_t x1 = input_data[i];
    uint32_t y1 = input_data[16 + i];
    uint32_t z1 = input_data[32 + i];
    uint32_t x2 = input_data[48 + i];
    uint32_t y2 = input_data[64 + i];
    uint32_t z2 = input_data[80 + i];

    printf("Point %2d: (%3u, %3u, %4u) <-> (%3u, %3u, %4u) = %10u\n",
           i, x1, y1, z1, x2, y2, z2, output_data[i]);
  }

  // Verify first few results manually
  printf("\nVerification (manual calculation):\n");
  for (int i = 0; i < 3; i++) {
    uint32_t x1 = input_data[i];
    uint32_t y1 = input_data[16 + i];
    uint32_t z1 = input_data[32 + i];
    uint32_t x2 = input_data[48 + i];
    uint32_t y2 = input_data[64 + i];
    uint32_t z2 = input_data[80 + i];

    int32_t dx = (int32_t)(x1 - x2);
    int32_t dy = (int32_t)(y1 - y2);
    int32_t dz = (int32_t)(z1 - z2);
    uint32_t expected = (uint32_t)(dx*dx + dy*dy + dz*dz);

    printf("Point %d: expected=%u, got=%u %s\n",
           i, expected, output_data[i],
           expected == output_data[i] ? "✓" : "✗");
  }

  return 0;
}
