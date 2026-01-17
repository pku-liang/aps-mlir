#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cd $APS/tutorial/csrc/inference

mkdir -p $APS/tutorial/outputs
mkdir -p $APS/tutorial/outputs/inference_model

python3 bin2header.py stories15Mq.bin $APS/tutorial/outputs/inference_model/model_weights.h
python3 tokenizer2header.py tokenizer.bin $APS/tutorial/outputs/inference_model/tokenizer_data.h

riscv32-unknown-elf-gcc -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf -march=rv32imaf_zicsr_zifencei -mabi=ilp32f -mcmodel=medany -specs=htif_nano.specs -static -T htif.ld -I$MEGGINCLUDE -I$APS/tutorial/outputs/inference_model -o $APS/tutorial/outputs/inference.riscv inference.c tinyalloc/tinyalloc.c -lm

riscv32-unknown-elf-gcc -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf -march=rv32imaf_zicsr_zifencei -mabi=ilp32f -mcmodel=medany -specs=htif_nano.specs -static -T htif.ld -I$MEGGINCLUDE -I$APS/tutorial/outputs/inference_model -o $APS/tutorial/outputs/inference_native.riscv inference_native.c tinyalloc/tinyalloc.c -lm

cd -