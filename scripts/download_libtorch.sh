#!/bin/bash
# scripts/download_libtorch.sh
set -e

LIBTORCH_VERSION="2.5.1"
CUDA_VERSION="cu124"
LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${CUDA_VERSION}.zip"

echo "Downloading LibTorch ${LIBTORCH_VERSION} (${CUDA_VERSION})..."

# Create third_party directory if it doesn't exist
mkdir -p ../webcam_capture/third_party
cd ../webcam_capture/third_party

# Remove existing libtorch if it exists
if [ -d "libtorch" ]; then
    echo "Removing existing LibTorch..."
    rm -rf libtorch
fi

# Download LibTorch
echo "Downloading from: $LIBTORCH_URL"
wget --progress=bar -O libtorch.zip "$LIBTORCH_URL"

# Extract
echo "Extracting LibTorch..."
unzip -q libtorch.zip
rm libtorch.zip

echo "LibTorch ${LIBTORCH_VERSION} (${CUDA_VERSION}) installed successfully to third_party/libtorch/"
echo "You can now build your project with CUDA support!"