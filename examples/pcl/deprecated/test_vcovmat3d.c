#include "marchid.h"
#include <riscv-pk/encoding.h>
#include <stdint.h>
#include <stdio.h>

__attribute__((alwaysinline)) uint32_t vcovmat3d_vv(uint32_t rs1,
                                                    uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x30, %0, %1, %2" // opcode=0x2B (0101011),
                                                       // funct7=0x30 (0110000)
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data: 2 3D points stored contiguously
// Layout: [p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, extra1, extra2]
// Using signed integers (i32) for this operation
volatile int32_t input_data[8] __attribute__((aligned(128))) = {
    100, 200, 300, // Point 1: (100, 200, 300)
    50,  80,  150, // Point 2 (centroid): (50, 80, 150)
    123, 456       // Extra values that should be passed through
};

// Output buffer for covariance matrix (6 upper-triangle values + 2 passthrough)
// [C00, C01, C02, C11, C12, C22, extra1, extra2]
volatile int32_t output_data[8] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("VCOVMAT3D Test - 3D Covariance Matrix / Outer Product\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char *march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vcovmat3d_vv((uint32_t)(unsigned long)input_data,
                 (uint32_t)(unsigned long)output_data);
  }

  printf("Returned rd value: %u\n", result);

  printf("\nInput points:\n");
  printf("Point 1: (%d, %d, %d)\n", input_data[0], input_data[1],
         input_data[2]);
  printf("Point 2: (%d, %d, %d)\n", input_data[3], input_data[4],
         input_data[5]);

  int32_t dx = input_data[0] - input_data[3];
  int32_t dy = input_data[1] - input_data[4];
  int32_t dz = input_data[2] - input_data[5];
  printf("\nDifference vector: (%d, %d, %d)\n", dx, dy, dz);

  printf("\nComputed covariance matrix (upper triangle):\n");
  printf("  C00 (σ_xx) = %10d\n", output_data[0]);
  printf("  C01 (σ_xy) = %10d\n", output_data[1]);
  printf("  C02 (σ_xz) = %10d\n", output_data[2]);
  printf("  C11 (σ_yy) = %10d\n", output_data[3]);
  printf("  C12 (σ_yz) = %10d\n", output_data[4]);
  printf("  C22 (σ_zz) = %10d\n", output_data[5]);
  printf("  Extra values: %d, %d\n", output_data[6], output_data[7]);

  printf("\nFull symmetric matrix:\n");
  printf("  [%10d %10d %10d]\n", output_data[0], output_data[1],
         output_data[2]);
  printf("  [%10d %10d %10d]\n", output_data[1], output_data[3],
         output_data[4]);
  printf("  [%10d %10d %10d]\n", output_data[2], output_data[4],
         output_data[5]);

  // Verify results manually
  printf("\nVerification (manual calculation):\n");
  int32_t expected[6];
  expected[0] = dx * dx;
  expected[1] = dx * dy;
  expected[2] = dx * dz;
  expected[3] = dy * dy;
  expected[4] = dy * dz;
  expected[5] = dz * dz;

  const char *names[] = {"C00", "C01", "C02", "C11", "C12", "C22"};
  int all_correct = 1;
  for (int i = 0; i < 6; i++) {
    int match = (expected[i] == output_data[i]);
    printf("%s: expected=%d, got=%d %s\n", names[i], expected[i],
           output_data[i], match ? "✓" : "✗");
    if (!match)
      all_correct = 0;
  }

  printf("\nExtra values passthrough: expected=(123, 456), got=(%d, %d) %s\n",
         output_data[6], output_data[7],
         (output_data[6] == 123 && output_data[7] == 456) ? "✓" : "✗");

  printf("\nOverall: %s\n", all_correct ? "PASS" : "FAIL");

  return 0;
}
