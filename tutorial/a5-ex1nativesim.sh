#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cd $APS_CHIPYARD/sims/verilator && make CONFIG=APSRocketConfig run-binary-debug BINARY=$APS/tutorial/outputs/v3ddist_vv_native.riscv LOADMEM=1 -j4

cd -