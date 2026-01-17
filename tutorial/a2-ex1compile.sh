#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

pixi run compile $APS/tutorial/cadl/v3ddist_vv.cadl $APS/tutorial/csrc/test_v3ddist_vv.c $APS/tutorial/outputs/v3ddist_vv.riscv
