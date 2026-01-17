#!/bin/bash

# Build the APS image with pre-built .pixi using multi-stage build
# Usage: ./build.sh [APS_DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    # Default to parent directory of docker/
    APS_DIR="$(realpath "$SCRIPT_DIR/..")"
else
    APS_DIR="$(realpath "$1")"
fi

if [ ! -d "$APS_DIR" ]; then
    echo "Error: APS_DIR '$APS_DIR' does not exist"
    exit 1
fi

IMAGE_NAME="aps-mlir:latest"

echo "Building APS image (multi-stage build)..."
echo "  APS_DIR: $APS_DIR"
echo "  Image:   $IMAGE_NAME"

docker build \
    --progress=plain \
    -t "$IMAGE_NAME" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$APS_DIR"

echo "Image '$IMAGE_NAME' created successfully."
echo "This image contains a pre-built .pixi directory with correct container paths."
