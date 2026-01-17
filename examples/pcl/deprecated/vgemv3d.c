// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t vgemv3d_vv(int32_t *rs1, int32_t *rs2) {
    int32_t acc = 0;

    uint8_t rd_result = 0;
    int32_t * addr = rs1;
    int32_t * out_addr = rs2;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 4; ++i) {
        acc = 0;
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            acc = (acc + (rs1[((i * 4) + j)] * rs1[16 + j]));
            uint32_t j_ = (j + 1);
        }
        rs2[i] = acc;
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
