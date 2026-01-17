#!/bin/bash

# Run the APS Docker container
# Usage: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="aps-mlir:latest"
CONTAINER_NAME="aps-mlir-tute"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Error: Image '$IMAGE_NAME' not found."
    echo "Please build it first with: $SCRIPT_DIR/build.sh"
    exit 1
fi

# Check if container with this name already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' already exists."
    echo "To remove it, run: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
    exit 1
fi

echo "Starting APS Docker container..."
echo "  Image: $IMAGE_NAME"

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"
