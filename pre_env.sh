#!/bin/sh

TORCH_VERSION=1.8.1
CUDA_VERSION=cpu

export WITH_METIS=1
pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-geometric
