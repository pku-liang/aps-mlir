#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cd $APS_VLSI && make yosys CONFIG=APSRocketConfig

python3 $APS/tutorial/vlsi-report-simplify.py