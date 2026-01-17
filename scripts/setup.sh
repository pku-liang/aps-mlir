#!/usr/bin/env bash

set -e

# This script sets up the environment for the LLVM/MLIR/CIRCT project and its dependencies.

# check if circt is already installed
# ${which circt-opt} should be ${PWD}/install/bin/circt-opt
# ${circt-opt --version} should contain ${CIRCT_COMMIT}
# if so, return success
if [ -f "${PWD}/install/bin/circt-opt" ] && [[ "$(circt-opt --version)" == *"${CIRCT_COMMIT}"* ]]; then
    echo "circt is already installed"
    exit 0
fi

# - CIRCT_COMMIT: The commit to checkout
CIRCT_COMMIT=$1
if [ -z "$CIRCT_COMMIT" ]; then
    echo "Error: CIRCT_COMMIT is not set"
    exit 1
fi

# Submodule update
# be careful, don't update chipyard's submodule, it's fragile
# and must be updated with it's own script!!!
git submodule update --init --recursive circt/

# cd into the CIRCT repository, pushd is better for this
pushd circt

# Mkdir build 
mkdir -p build

# Cd into build
pushd build

# Workaround for protobuf version conflict:
# OR-Tools bundles protobuf 5.26, but pixi has protobuf 6.x
# Solution: Clear CMAKE_PREFIX_PATH to prevent CMake from finding pixi's protobuf
# We only keep the paths we explicitly need

# Save paths we need from pixi
PIXI_ZSTD_INCLUDE="$CONDA_PREFIX/include"
PIXI_ZSTD_LIB="$CONDA_PREFIX/lib/libzstd.so"
PIXI_PYTHON="$CONDA_PREFIX/bin/python3"

# Clear pixi's influence on CMake search paths
unset CMAKE_PREFIX_PATH
unset CPLUS_INCLUDE_PATH
unset C_INCLUDE_PATH

# Cmake - use explicit paths for everything we need
cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DOR_TOOLS_PATH=${OR_TOOLS_PATH} \
    -DCMAKE_PREFIX_PATH="${OR_TOOLS_PATH}" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCMAKE_INSTALL_PREFIX=../../install \
    -DLLVM_ENABLE_ZSTD=FORCE_ON \
    -DZSTD_INCLUDE_DIR="${PIXI_ZSTD_INCLUDE}" \
    -DZSTD_LIBRARY="${PIXI_ZSTD_LIB}" \
    -DPython3_EXECUTABLE="${PIXI_PYTHON}"

# Ninja
ninja

# Ninja check mlir
# ninja check-mlir

# Ninja check mlir python
# ninja check-mlir-python

# Ninja check circt
# ninja check-circt

# Ninja check circt integration
# ninja check-circt-integration

# Ninja install
# ninja install

# Echo success
echo "llvm/mlir/circt build/install success with python bindings"

# Cd back to the circt repository
popd


# Popd from the circt repository
popd

# Echo success
echo "circt $CIRCT_COMMIT setup success" > __setup.success