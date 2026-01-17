// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t gemm_4x4(int32_t *rs1, int32_t *rs2) {
    int32_t acc = 0;

    uint8_t rd_result = 0;
    int32_t * addr_a = rs1;
    int32_t * out_addr = rs2;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 4; ++i) {
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            acc = 0;
            uint32_t k;
            for (k = 0; k < 4; ++k) {
                uint32_t a_idx = ((i * 4) + k);
                int32_t a_val = rs1[a_idx];
                uint32_t b_idx = ((k * 4) + j);
                int32_t b_val = rs1[16 + b_idx];
                acc = (acc + (a_val * b_val));
                uint32_t k_ = (k + 1);
            }
            uint32_t c_idx = ((i * 4) + j);
            rs2[c_idx] = acc;
            uint32_t j_ = (j + 1);
        }
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
