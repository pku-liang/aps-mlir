#!/usr/bin/env bash

set -e

# This script sets up the environment for the LLVM/MLIR/CIRCT project and its dependencies.

# ORTOOLS: downloads from github and unzip

UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version: $UBUNTU_VERSION"
if [ "$UBUNTU_VERSION" == "22.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-22.04_cpp_v9.10.4067.tar.gz"
elif [ "$UBUNTU_VERSION" == "20.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-20.04_cpp_v9.10.4067.tar.gz"
elif [ "$UBUNTU_VERSION" == "24.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-24.04_cpp_v9.10.4067.tar.gz"
else
    echo "Error: Ubuntu version $UBUNTU_VERSION is not supported"
    exit 1
fi

# Create install directory if it doesn't exist
mkdir -p ./install

# Download and extract or-tools, stripping the top-level directory
wget -qO- $DOWNLOAD_URL | tar -xz --strip-components=1 -C ./install

echo "ORTOOLS setup completed" > __setup-ortools.success