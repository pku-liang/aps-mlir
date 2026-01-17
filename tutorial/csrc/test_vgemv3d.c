#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
uint8_t vgemv3d_vv(int32_t *array1, int32_t *array2) {
    int32_t acc = 0;
    uint8_t rd_result = 0;
    int32_t * addr = array1;
    int32_t * out_addr = array2;
    uint32_t i;
    for (i = 0; i < 4; ++i) {
        acc = 0;
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            acc = ((array1[((i << 2) + j)] * array1[16 + j]) + acc);
            uint32_t j_ = (j + 1);
        }
        array2[i] = acc;
        uint32_t i_ = (i + 1);
    }
    rd_result = 0;
    return rd_result;
}
volatile int32_t input_data[20] __attribute__((aligned(128))) = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
    1, 1, 1, 1
};
volatile int32_t output_data[4] __attribute__((aligned(128))) = {0};

int main(void) {
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vgemv3d_vv((int32_t *)input_data, (int32_t *)output_data);
  }

  printf("Result: %u\n", result);
  printf("Output vector: [%d, %d, %d, %d]\n",
         output_data[0], output_data[1], output_data[2], output_data[3]);

  return 0;
}
