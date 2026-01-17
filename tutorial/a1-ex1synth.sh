#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

# Run CADL-to-MLIR parser
pixi run mlir $APS/tutorial/cadl/v3ddist_vv.cadl $APS/tutorial/outputs/v3ddist_vv.mlir

# Run APS synthesis tool
pixi run opt $APS/tutorial/outputs/v3ddist_vv.mlir $APS/tutorial/outputs/v3ddist_vv_cmt.mlir

# Run Cement's synthesis flow
pixi run sv $APS/tutorial/outputs/v3ddist_vv_cmt.mlir $APS/tutorial/outputs/v3ddist_vv.sv