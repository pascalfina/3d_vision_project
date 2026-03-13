#!/usr/bin/env bash

set -euo pipefail

python - <<'PY'
import os
import torch

print("VIRTUAL_ENV=", os.environ.get("VIRTUAL_ENV", ""))
print("CUDA_HOME=", os.environ.get("CUDA_HOME", ""))
print("DATA_ROOT_DIR=", os.environ.get("DATA_ROOT_DIR", ""))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())

import xformers
import torch_geometric
import spconv.pytorch as spconv  # noqa: F401
import simple_knn._C  # noqa: F401
import diff_gaussian_rasterization  # noqa: F401
import src.trainval.train_reconstruction  # noqa: F401

print("Object-X environment check passed")
PY
