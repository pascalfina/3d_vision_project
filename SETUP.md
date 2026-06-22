# Setup

Tested on Linux with Python 3.9, CUDA 12.8, and an NVIDIA GPU.

## 1. Storage paths

Create the local path configuration and adjust the two roots if needed:

```bash
cp configs/workflows/local_paths.env.example configs/workflows/local_paths.env
```

The file must contain:

```bash
export OBJECTX_USER_ROOT="/work/scratch/${USER}"
export OBJECTX_TEAM_ROOT="/work/courses/3dv/team35/${USER}"
```

Both roots may be changed to any accessible storage locations. Load them for
the remaining setup commands:

```bash
source configs/workflows/local_paths.env
```

## 2. Main environment

The workflow wrapper expects the environment at `.venv_objx`:

```bash
python3.9 -m venv .venv_objx
source .venv_objx/bin/activate
pip install -U pip setuptools wheel
pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/requirements.runtime.txt
pip install spconv-cu121==2.3.8 gsplat==1.4.0
```

On a GPU node, install the required Object-X CUDA extensions and compile the
optional fast RoPE kernels. Rebuild them after changing GPU architecture:

```bash
source scripts/activate_objectx_env.sh
pip install --no-build-isolation -e dependencies/gaussian-splatting/submodules/simple-knn
pip install --no-build-isolation -e dependencies/gaussian-splatting/submodules/diff-gaussian-rasterization
(cd dependencies/must3r/dust3r/croco/models/curope && python setup.py build_ext --inplace --force)
(cd dependencies/pi3/pi3/models/curope && python setup.py build_ext --inplace --force)
# Fallback when the compiled Pi3X kernel is incompatible:
# export OBJECTX_PI3X_DISABLE_CUROPE=1
```

## 3. SAMObject environment

SAMObject runs in a separate environment:

```bash
python3.10 -m venv "$OBJECTX_TEAM_ROOT/.venv_sam2object"
source "$OBJECTX_TEAM_ROOT/.venv_sam2object/bin/activate"
pip install -U pip
pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile
pip install -e dependencies/SAM2Object/segtrack
```

## 4. Weights and data

Place the downloaded weights at these paths:

```text
models/must3r/MUSt3R_512.pth
models/sam2ckpt/sam2_hiera_base_plus.pt
pretrained/slat_pretrained.pth.tar
pretrained/u3dgs_pretrained_16_ot.pth.tar
${OBJECTX_TEAM_ROOT}/models/pi3/Pi3X.safetensors
```

Download instructions for MUSt3R, SAM2, and Pi3X are in their READMEs under
`dependencies/`. The two files under `pretrained/` are the Object-X
checkpoints and must be obtained separately.

The prepared dataset must be located at
`${OBJECTX_USER_ROOT}/objectx-data-baseline`. Then verify the installation:

```bash
source scripts/activate_objectx_env.sh
bash scripts/check_objectx_env.sh
```
