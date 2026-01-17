#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
uint32_t vfpsmax_v(uint32_t *rs1, uint32_t *rs2) {
    uint32_t indices[16];
    uint32_t reduction_idxs_in[16];
    uint32_t reduction_idxs_out[16];
    uint32_t reduction_vals_out[16];

    uint32_t rd_result = 0;
    uint32_t * addr = rs1;
    uint32_t vl = 16;
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 16; ++i) {
        indices[i] = i;
        reduction_idxs_in[i] = i;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 8; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = rs1[idx1];
        uint32_t val2 = rs1[idx2];
        uint8_t is_sentinel1 = ((val1 == 0) ? 1 : 0);
        uint8_t is_sentinel2 = ((val2 == 0) ? 1 : 0);
        uint32_t max_val = (is_sentinel1 ? val2 : (is_sentinel2 ? val1 : ((val1 > val2) ? val1 : val2)));
        uint32_t reduction_idxs_in_idx2 = reduction_idxs_in[idx2];
        uint32_t reduction_idxs_in_idx1 = reduction_idxs_in[idx1];
        uint32_t max_idx = (is_sentinel1 ? reduction_idxs_in_idx2 : (is_sentinel2 ? reduction_idxs_in_idx1 : ((val1 > val2) ? reduction_idxs_in_idx1 : reduction_idxs_in_idx2)));
        reduction_vals_out[i] = max_val;
        reduction_idxs_out[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 4; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = reduction_vals_out[idx1];
        uint32_t val2 = reduction_vals_out[idx2];
        uint32_t max_val = ((val1 > val2) ? val1 : val2);
        uint32_t reduction_idxs_out_idx1 = reduction_idxs_out[idx1];
        uint32_t reduction_idxs_out_idx2 = reduction_idxs_out[idx2];
        uint32_t max_idx = ((val1 > val2) ? reduction_idxs_out_idx1 : reduction_idxs_out_idx2);
        rs1[i] = max_val;
        reduction_idxs_in[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 2; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = rs1[idx1];
        uint32_t val2 = rs1[idx2];
        uint32_t max_val = ((val1 > val2) ? val1 : val2);
        uint32_t reduction_idxs_in_idx1 = reduction_idxs_in[idx1];
        uint32_t reduction_idxs_in_idx2 = reduction_idxs_in[idx2];
        uint32_t max_idx = ((val1 > val2) ? reduction_idxs_in_idx1 : reduction_idxs_in_idx2);
        reduction_vals_out[i] = max_val;
        reduction_idxs_out[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    uint32_t val1 = reduction_vals_out[0];
    uint32_t val2 = reduction_vals_out[1];
    uint32_t final_max = ((val1 > val2) ? val1 : val2);
    uint32_t reduction_idxs_out_0 = reduction_idxs_out[0];
    uint32_t reduction_idxs_out_1 = reduction_idxs_out[1];
    uint32_t final_idx = ((val1 > val2) ? reduction_idxs_out_0 : reduction_idxs_out_1);
    rd_result = final_idx;
    uint32_t * out_addr = rs2;
    rs2[0] = final_max;
    rs2[1] = final_idx;
    return rd_result;
}


// Input: 16 unsigned integers
volatile uint32_t input_data[16] __attribute__((aligned(128))) = {
    42, 17, 99, 3, 87, 54, 23, 76,
    91, 8, 65, 31, 48, 72, 15, 84
};

// Output: maximum value
volatile uint32_t output_data[1] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("VFPSMAX Test - Vector Floating-Point Signed Maximum\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char *march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vfpsmax_v((uint32_t *)input_data, (uint32_t *)output_data);
  }

  printf("Result: %u\n", result);
  printf("Maximum value: %u (expected: 99)\n", output_data[0]);

  return 0;
}
