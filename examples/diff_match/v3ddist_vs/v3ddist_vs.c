// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

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
    // tiled by factor of 8
    for (uint32_t ib = 0; ib < 16; ib += 8) {
        uint32_t i_end = ib + 8;
        for (i = ib; i < i_end; ++i) {
            uint32_t x = rs1[i];
            uint32_t y = rs1[16 + i];
            uint32_t z = rs1[32 + i];
            uint32_t dx = (x - ref_x);
            uint32_t dy = (y - ref_y);
            uint32_t dz_2 = z * z + (ref_z * ref_z) - (z * ref_z << 1);
            uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz_2));
            rs2[i] = dist_sq;
            uint32_t i_ = (i + 1);
        }
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
