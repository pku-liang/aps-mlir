#include "marchid.h"
#include <riscv-pk/encoding.h>
#include <stdint.h>
#include <stdio.h>

__attribute__((alwaysinline)) uint32_t vfpsmax_v(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x5B, 0b111, 0x2A, %0, %1, %2" // opcode=0x2B (0101011),
                                                       // funct7=0x2A (0101010)
               : "=r"(rd)
               : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data: 16 distance values (u32)
// This is used in Farthest Point Sampling to find the point with maximum
// distance
volatile uint32_t input_data[16] __attribute__((aligned(128))) = {
    100,  500,  250, 750, // Values 0-3
    1200, 300,  900, 450, // Values 4-7
    600,  1500, 200, 850, // Values 8-11
    400,  950,  350, 1100 // Values 12-15
};

// Output buffer: [max_value, max_index]
volatile uint32_t output_data[2] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("VFPSMAX Test - Vector Max Reduction with Index Tracking\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char *march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vfpsmax_v((uint32_t)(unsigned long)input_data,
              (uint32_t)(unsigned long)output_data);
  }

  printf("Returned rd value (max index): %u\n", result);

  printf("\nInput distance values:\n");
  for (int i = 0; i < 16; i++) {
    printf("  [%2d] = %4u%s\n", i, input_data[i],
           (i == result || i == output_data[1]) ? " <-- MAX" : "");
  }

  printf("\nResults:\n");
  printf("  Max value: %u\n", output_data[0]);
  printf("  Max index: %u\n", output_data[1]);
  printf("  Returned in rd: %u\n", result);

  // Verify results manually
  printf("\nVerification (manual max search):\n");
  uint32_t expected_max = 0;
  uint32_t expected_idx = 0;
  for (int i = 0; i < 16; i++) {
    if (input_data[i] > expected_max) {
      expected_max = input_data[i];
      expected_idx = i;
    }
  }

  printf("Expected max value: %u\n", expected_max);
  printf("Expected max index: %u\n", expected_idx);

  int value_match = (expected_max == output_data[0]);
  int index_match = (expected_idx == output_data[1]);
  int rd_match = (expected_idx == result);

  printf("\nValue match: %s (expected=%u, got=%u)\n", value_match ? "✓" : "✗",
         expected_max, output_data[0]);
  printf("Index match: %s (expected=%u, got=%u)\n", index_match ? "✓" : "✗",
         expected_idx, output_data[1]);
  printf("RD match: %s (expected=%u, got=%u)\n", rd_match ? "✓" : "✗",
         expected_idx, result);

  printf("\nOverall: %s\n",
         (value_match && index_match && rd_match) ? "PASS" : "FAIL");

  return 0;
}
