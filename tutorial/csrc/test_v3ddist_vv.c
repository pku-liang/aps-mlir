#include <stdint.h>
#include <stdio.h>

#pragma megg optimize
uint8_t v3ddist_vv(uint32_t *rs1, uint32_t *rs2) {
    uint8_t rd_result = 0;
    uint32_t vl = 16;
    for (uint32_t i = 0; i < vl; ++i) {
        uint32_t x1 = rs1[i];
        uint32_t y1 = rs1[16 + i];
        uint32_t z1 = rs1[32 + i];
        uint32_t x2 = rs1[48 + i];
        uint32_t y2 = rs1[64 + i];
        uint32_t z2 = rs1[80 + i];
        uint32_t dx_2 = (x1 * x1 + x2 * x2 - 2 * x1 * x2);
        uint32_t dy = (y1 - y2);
        uint32_t dz = (z1 - z2);
        uint32_t dist_sq = ((dx_2 + (dy * dy)) + (dz * dz));
        rs2[i] = dist_sq;
    }
    rd_result = 0;
    return rd_result;
}

volatile uint32_t input_data[96] __attribute__((aligned(128))) = {
  // points1_x[16]
  10, 20, 30, 40, 50, 60, 70, 80,
  90, 100, 110, 120, 130, 140, 150, 160,

  // points1_y[16]
  5, 15, 25, 35, 45, 55, 65, 75,
  85, 95, 105, 115, 125, 135, 145, 155,

  // points1_z[16]
  100, 200, 300, 400, 500, 600, 700, 800,
  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,

  // points2_x[16]
  15, 25, 35, 45, 55, 65, 75, 85,
  95, 105, 115, 125, 135, 145, 155, 165,

  // points2_y[16]
  10, 20, 30, 40, 50, 60, 70, 80,
  90, 100, 110, 120, 130, 140, 150, 160,

  // points2_z[16]
  110, 210, 310, 410, 510, 610, 710, 810,
  910, 1010, 1110, 1210, 1310, 1410, 1510, 1610,
};

// Output buffer for distances
volatile uint32_t output_data[16] __attribute__((aligned(128))) = {0};

volatile uint32_t result = 0;

__always_inline uint32_t read_cycle(void) {
  uint32_t cycle;
  asm volatile ("fence");
  asm volatile ("rdcycle %0" : "=r" (cycle));
  asm volatile ("fence");
  return cycle;
}

int main(void) {
  for (int i = 0; i < 10; i++) {
    uint32_t start_cycle = read_cycle();
    result = v3ddist_vv((uint32_t*)input_data,
               (uint32_t*)output_data);
    uint32_t end_cycle = read_cycle();
    if (i == 9) {
      printf("Used %lu cycles\n", end_cycle - start_cycle);
      printf("Computed results:\n");
      for (int i = 0; i < 16; i++) {
        uint32_t x1 = input_data[i];
        uint32_t y1 = input_data[16 + i];
        uint32_t z1 = input_data[32 + i];
        uint32_t x2 = input_data[48 + i];
        uint32_t y2 = input_data[64 + i];
        uint32_t z2 = input_data[80 + i];

        printf("[%2d]: (%3lu, %3lu, %4lu) <-> (%3lu, %3lu, %4lu) = %4lu\n",
              i + 1, x1, y1, z1, x2, y2, z2, output_data[i]);
      }
    }
  }

  return 0;
}
