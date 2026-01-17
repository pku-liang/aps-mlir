// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t v3ddist_vv(uint32_t *rs1, uint32_t *rs2) {
    uint8_t rd_result = 0;
    uint32_t * addr1 = rs1;
    uint32_t * out_addr = rs2;
    uint32_t vl = 16;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    // for (i = 0; i < vl; ++i) {
    //     uint32_t x1 = rs1[i];
    //     uint32_t y1 = rs1[16 + i];
    //     uint32_t z1 = rs1[32 + i];
    //     uint32_t x2 = rs1[48 + i];
    //     uint32_t y2 = rs1[64 + i];
    //     uint32_t z2 = rs1[80 + i];
    //     uint32_t dx = (x1 - x2);
    //     uint32_t dy = (y1 - y2);
    //     uint32_t dz = (z1 - z2);
    //     uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
    //     rs2[i] = dist_sq;
    //     uint32_t i_ = (i + 1);
    // }
    // version 1 - tile by 4

    for (uint32_t ib = 0; ib < 16; ib += 4) {   // 外层 Tile：每次处理 4 个元素
        uint32_t i_end = ib + 4;               // Tile 结束位置

        for (uint32_t i = ib; i < i_end; ++i) {
            // 读取坐标 (x1, y1, z1)
            int32_t x1 = rs1[i];
            int32_t y1 = rs1[16 + i];
            int32_t z1 = rs1[32 + i];

            // 读取坐标 (x2, y2, z2)
            int32_t x2 = rs1[48 + i];
            int32_t y2 = rs1[64 + i];
            int32_t z2 = rs1[80 + i];

            // 计算 (dx^2 + dy^2 + dz^2)
            int32_t dx = x1 - x2;
            int32_t dx2 = dx * dx;

            int32_t dy = y1 - y2;
            int32_t dy2 = dy * dy;

            int32_t dz = z1 - z2;
            int32_t dz2 = dz * dz;

            int32_t dist2 = dx2 + dy2 + dz2;

            // 存入 rs2[i]
            rs2[i] = dist2;
        }
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
