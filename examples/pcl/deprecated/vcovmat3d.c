// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t vcovmat3d_vv(int32_t *rs1, int32_t *rs2) {
    uint8_t rd_result = 0;
    int32_t * addr = rs1;
    int32_t * out_addr = rs2;
    // burst_read lowered via register-backed scratchpad
    int32_t x = rs1[0];
    int32_t y = rs1[1];
    int32_t z = rs1[2];
    int32_t cx = rs1[3];
    int32_t cy = rs1[4];
    int32_t cz = rs1[5];
    int32_t dx = (x - cx);
    int32_t dy = (y - cy);
    int32_t dz = (z - cz);
    rs2[0] = (dx * dx);
    rs2[1] = (dx * dy);
    rs2[2] = (dx * dz);
    rs2[3] = (dy * dy);
    rs2[4] = (dy * dz);
    rs2[5] = (dz * dz);
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
