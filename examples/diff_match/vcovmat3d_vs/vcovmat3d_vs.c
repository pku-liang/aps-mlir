// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source
#include <stdint.h>

uint8_t vcovmat3d_vs(int32_t *rs1, int32_t *rs2) {
  uint8_t rd_result = 0;
  int32_t *points_addr = rs1;
  int32_t *out_addr = rs2;
  int32_t cx = rs1[48];
  int32_t cy = rs1[49];
  int32_t cz = rs1[50];
  rs2[0] = 0;
  rs2[1] = 0;
  rs2[2] = 0;
  rs2[3] = 0;
  rs2[4] = 0;
  rs2[5] = 0;
  rs2[6] = 16;
  rs2[7] = 0;

  uint32_t i;
  // for (i = 0; i < 16; ++i) {
  //   int32_t x = rs1[i];
  //   int32_t y = rs1[16 + i];
  //   int32_t z = rs1[32 + i];
  //   int32_t dx = (x - cx);
  //   int32_t dy = (y - cy);
  //   int32_t dz = (z - cz);
  //   rs2[0] = (rs2[0] + (dx * dx));
  //   rs2[1] = (rs2[1] + (dx * dy));
  //   rs2[2] = (rs2[2] + (dx * dz));
  //   rs2[3] = (rs2[3] + (dy * dy));
  //   rs2[4] = (rs2[4] + (dy * dz));
  //   rs2[5] = (rs2[5] + (dz * dz));
  //   uint32_t i_ = (i + 1);
  // }
  // tile by factor of 4
    for (uint32_t ib = 0; ib < 16; ib+=4) {
        uint32_t i_end = ib + 4;
        for (i = ib; i < i_end; ++i) {
          int32_t x = rs1[i];
          int32_t y = rs1[16 + i];
          int32_t z = rs1[32 + i];
          int32_t dx = (x - cx);
          int32_t dy = (y - cy);
          int32_t dz = (z - cz);
          rs2[0] = (rs2[0] + (dx * dx));
          rs2[1] = (rs2[1] + (dx * dy));
          rs2[2] = (rs2[2] + (dx * dz));
          rs2[3] = (rs2[3] + (dy * dy));
          rs2[4] = (rs2[4] + (dy * dz));
          rs2[5] = (rs2[5] + (dz * dz));
          uint32_t i_ = (i + 1);
        }
    }
  // burst_write lowered via register-backed scratchpad
  rd_result = 1;
  return rd_result;
}