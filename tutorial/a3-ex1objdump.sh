#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cat $APS/tutorial/outputs/compile_logs/v3ddist_vv.asm | grep -v f1202573 | grep insn -C 5