#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

pixi run compile-native $APS/tutorial/csrc/test_v3ddist_vv.c $APS/tutorial/outputs/v3ddist_vv_native.riscv

pixi run riscv32-unknown-elf-objdump -j .text -D $APS/tutorial/outputs/v3ddist_vv_native.riscv | grep -v f1202573 | grep insn -C 5