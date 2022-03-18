#!/bin/bash
cmake .. \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
    -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
    -Donnxruntime_DIR=$CONDA_PREFIX
