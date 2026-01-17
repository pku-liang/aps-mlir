// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t v3ddist_vv(uint32_t *rs1, uint32_t *rs2) {
    uint32_t points2_z[16];

    uint8_t rd_result = 0;
    uint32_t * addr1 = rs1;
    uint32_t * out_addr = rs2;
    uint32_t vl = 16;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read eliminated (fallback)
    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x1 = rs1[i];
        uint32_t y1 = rs1[16 + i];
        uint32_t z1 = rs1[32 + i];
        uint32_t x2 = rs1[48 + i];
        uint32_t y2 = rs1[64 + i];
        uint32_t z2 = points2_z[i];
        uint32_t dx = (x1 - x2);
        uint32_t dy = (y1 - y2);
        uint32_t dz = (z1 - z2);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        rs2[i] = dist_sq;
        uint32_t i_ = (i + 1);
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
