#!/usr/bin/env bash

# This script sets up the environment for the CIRCT project.
# WARNING: THIS SCRIPT FAILED, since CIRCT's python bindings require a unified build.

# Take arguments:
# - CIRCT_COMMIT: The commit to checkout

CIRCT_COMMIT=$1
if [ -z "$CIRCT_COMMIT" ]; then
    echo "Error: CIRCT_COMMIT is not set"
    exit 1
fi

# Clone the CIRCT repository
if [ ! -d "circt" ]; then
    git clone git@github.com:circt/circt.git
fi

# cd into the CIRCT repository, pushd is better for this
pushd circt

# Checkout the CIRCT commit
git checkout $CIRCT_COMMIT

# Submodule update
git submodule init

# Submodule update
git submodule update

# Mkdir llvm/build 
mkdir -p llvm/build

# Cd into llvm/build
pushd llvm/build

# Cmake
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# Ninja
ninja

# Ninja check mlir
ninja check-mlir

# Ninja check mlir python
ninja check-mlir-python

# Echo success
echo "llvm/mlir build success with python bindings"

# Cd back to the circt repository
popd

# Mkdir build
mkdir -p build

# Cd into build
pushd build

# Cmake
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON

# Ninja
ninja

# Ninja check circt
ninja check-circt

# Ninja check circt integration
ninja check-circt-integration

# Echo success
echo "circt build success"

# Cd back to the circt repository
popd

# Popd from the circt repository
popd

# Echo success
echo "circt setup success"