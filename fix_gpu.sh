#!/bin/bash
# fix_gpu.sh: A script to fix GPU compatibility issues in a Poetry environment.
# This script checks your CUDA version, uninstalls conflicting NVIDIA packages,
# sets up the correct LD_LIBRARY_PATH, installs a CUDA-enabled PyTorch wheel (2.0.0+cu118),
# and verifies that PyTorch can detect your GPU.
#
# Note: Even though your system has CUDA 12.6, the official PyTorch wheels are built
# against CUDA 11.x (cu118). The NVIDIA drivers on your system should support this.
#
# Adjust the torch version if you prefer a different version.

set -e

echo "=== Starting GPU compatibility fix ==="

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure that the CUDA toolkit is installed."
    exit 1
fi

# Query the CUDA version from nvcc
echo "Querying CUDA version..."
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //;s/,.*//')
echo "Detected CUDA version: $CUDA_VERSION"

# Check that CUDA version is at least 11.0
if [[ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]]; then
    echo "Error: CUDA version is lower than 11.0. This may not be supported by current PyTorch GPU builds."
    exit 1
fi

# Uninstall conflicting NVIDIA packages (if installed)
echo "Uninstalling conflicting NVIDIA packages from the Poetry environment..."
poetry run pip uninstall -y nvidia-cublas-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 || true

# Ensure that /usr/lib/x86_64-linux-gnu (where cuDNN is located) is in LD_LIBRARY_PATH
if [[ ":$LD_LIBRARY_PATH:" != *":/usr/lib/x86_64-linux-gnu:"* ]]; then
    echo "Adding /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
fi

# Install the PyTorch GPU build using CUDA 11.8 wheels.
# Since torch==1.13.1+cu118 is not available on your system, we use torch==2.0.0+cu118.
echo "Installing PyTorch 2.0.0 with CUDA 11.8 support..."
poetry run pip install --upgrade torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# (Optional) Install torchvision and torchaudio if needed:
# poetry run pip install --upgrade torchvision==0.14.1+cu118 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection using a small Python snippet
echo "Verifying GPU availability with PyTorch..."
GPU_AVAILABLE=$(poetry run python -c "import torch; print(torch.cuda.is_available())" | tr -d '\n')

if [ "$GPU_AVAILABLE" = "True" ]; then
    echo "Success: GPU is available for PyTorch."
else
    echo "Warning: GPU is NOT available for PyTorch. Please check your NVIDIA drivers and CUDA installation."
fi

echo "=== GPU compatibility fix completed ==="
