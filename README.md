<p align="center">
  <h2 align="center"> [NeurIPS 2025] Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations </h2>
    <p align="center">
    <a>Gaia Di Lorenzo</a><sup>1</sup>
    .
    <a>Federico Tombari</a><sup>3</sup>
    .
    <a>Marc Pollefeys</a><sup>1, 2</sup>
    .
    <a>Dániel Béla Baráth</a><sup>1, 3</sup>
    .
  </p>
  <p align="center">
    <sup>1</sup>ETH Zürich · <sup>2</sup>Microsoft · <sup>3</sup>Google
  </p>
</p>
<p align="center">
<a href="https://arxiv.org/abs/2506.04789"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://gaiadilorenzo.github.io/object-x'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<p align="center">
  <a href="">
    <img src="assets/teaser.png" width="70%">
  </a>
</p>

## Current Workflow Quick Start

For the current Scan3R/cluster workflow, prefer the short profile-based entrypoints instead of long ad-hoc commands.
Each profile defines one scene setup, and each action runs exactly one stage of the pipeline.

Important: this workflow does **not** replace the earlier Object-X dataset preparation. It builds on top of already prepared scene files and preprocessing artifacts such as `files/3RScan.json`, `files/objects.json`, `files/Features3D/`, `files/<mask_source>/obj_id_pkl/<scene>.pkl`, and `scenes/<scene>/sequence/`.

For a step-by-step explanation of the generated files and how the roots connect,
see [`docs/current_workflow.md`](docs/current_workflow.md).

List available profiles:

```bash
cd /work/scratch/$USER/object-x
bash scripts/workflows/run_scene_profile.sh --list-profiles
```

The current main profile is:

- `cabinet_legacy_sam2_hybrid`
- `cabinet_predready_v2_floorfix`
- `oven_predready_v1`

Run the full current `cabinet` SAM2-hybrid pipeline:

```bash
cd /work/scratch/$USER/object-x

bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid segment-inputs
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid voxelise
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid build-pred-ready
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid validate-pred-ready
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid compare-arrangement
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid slat
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid u3dgs
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid render
```

The older GT-mask profiles are still useful for comparison and ablations:

- `cabinet_predready_v2_floorfix`
- `oven_predready_v1`

What each command does:

- `segment-inputs`: runs the SAM2 keyframe selection, propagation, merge/fusion, and writes `files/sam2_projection/`, `objects_sam2.json`, and `scenes_sam2/`
- `voxelise`: runs the current "2.5" reconstruction path, i.e. builds object geometry from masks + depth + poses and writes `gs_annotations`
- `build-pred-ready`: rebuilds `objects.json` and `files/orig/data.pkl.gz` from the reconstructed objects
- `validate-pred-ready`: checks that the new scene root is internally consistent and dataset-loadable
- `compare-arrangement`: compares the rebuilt scene arrangement against the baseline GT arrangement
- `slat`: runs the structured latent encoding step
- `u3dgs`: runs the unstructured latent encode/decode and writes the joint output / render; workflow profiles enable the dense CPU decode fallback by default so large hybrid/TSDF scenes do not fail immediately on smaller GPUs
- `render`: creates the final MP4 + interactive HTML inspection bundle

Useful extras:

```bash
bash scripts/workflows/run_scene_profile.sh cabinet_legacy_sam2_hybrid u3dgs --dry-run
```

- `--dry-run` prints the exact underlying command without executing it

### SAM2 Hybrid Notes

The `cabinet_legacy_sam2_hybrid` profile is the current end-to-end SAM path:

- `segment-inputs` writes SAM2 masks to `files/sam2_projection/` inside the reconstruction root
- `voxelise` consumes those SAM2 masks and writes reconstructed object geometry to `files/gs_annotations/<scene>/`
- `build-pred-ready` and `validate-pred-ready` rebuild the downstream scene graph from those reconstructed objects
- `slat`, `u3dgs`, and `render` now also use `sam2_projection`

Important implementation detail:

- the pred-ready root stores the rebuilt scene graph and staged scene files
- the SAM2 mask pickles still live in the reconstruction root
- the workflow now forwards both automatically, so `slat`, `u3dgs`, and `render` read `sam2_projection` from the reconstruction root while keeping the pred-ready root as the main inference root

That means the current SAM2-hybrid workflow is:

```text
segment-inputs -> voxelise -> build-pred-ready -> validate/compare -> slat -> u3dgs -> render
```

### How To Add A New Scene Profile

Profiles live under:

- [`configs/workflows/scene_profiles`](configs/workflows/scene_profiles)

The two current examples are:

- [`cabinet_predready_v2_floorfix.json`](configs/workflows/scene_profiles/cabinet_predready_v2_floorfix.json)
- [`oven_predready_v1.json`](configs/workflows/scene_profiles/oven_predready_v1.json)

The safest workflow is:

1. Copy an existing profile that is closest to what you want.
2. Rename it to the new profile name.
3. Update the scene-specific paths and labels.
4. Check the config with `--dry-run` before launching a real run.

For example, to create a new profile from the current `cabinet` setup:

```bash
cp configs/workflows/scene_profiles/cabinet_predready_v2_floorfix.json \
   configs/workflows/scene_profiles/<new_profile_name>.json
```

If you want to understand what needs to change, compare:

- [`configs/workflows/scene_profiles/cabinet_predready_v2_floorfix.json`](configs/workflows/scene_profiles/cabinet_predready_v2_floorfix.json)
- [`configs/workflows/scene_profiles/oven_predready_v1.json`](configs/workflows/scene_profiles/oven_predready_v1.json)

The structure stays the same. You mainly replace scene-specific values.

#### Local Path Overrides

Profiles can use environment variables such as `${OBJECTX_BASELINE_ROOT}` and
`${OBJECTX_CABINET_RECON_ROOT}` instead of hardcoding every absolute path.

For a new account, copy the example file once:

```bash
cp configs/workflows/local_paths.env.example configs/workflows/local_paths.env
```

Then edit only:

```text
configs/workflows/local_paths.env
```

This file is ignored by git. The workflow runner loads it automatically before
expanding profile JSON values.

It covers the common roots for all workflow actions: `segment-inputs`, `must3r`,
`mast3r-sfm`, `fuse`, `voxelise`, `build-pred-ready`, validation, `slat`,
`u3dgs`, and `render`. The comments inside the file explain which variable
controls which dataset/output family.

#### What You Need Before Creating A New Profile

Before filling a new profile, make sure these files or directories exist for the new scene:

- a baseline root, usually `/work/scratch/pafina/objectx-data-baseline`
- the standard Object-X preprocessing artifacts in that root, especially `files/3RScan.json`, `files/objects.json`, `files/Features3D/`, and `scenes/<scene_id>/sequence/`
- a reconstruction root that contains `files/gs_annotations/<scene_id>/`
- a manifest JSON listing the selected objects for that scene, for example `debug/<scene_name>_fullscene_all_objects_manifest.json`
- a target pred-ready root path where the rebuilt scene metadata will be written
- a joint output path such as `vis/<scene_id>_joint.ply`

If one of these is missing, the profile may look correct but the run will still fail.

#### Field-By-Field Guide

These are the fields you normally need to update:

- `name`
  - short command-line name of the profile
  - example: `cabinet_predready_v2_floorfix` or `oven_predready_v1`

- `scene_id`
  - the exact Scan3R scene id
  - example: `e61b0e04-bada-2f31-82d6-72831a602ba7` for cabinet
  - example: `5341b7e3-8a66-2cdd-8709-66a2159f0017` for oven

- `split`
  - usually `val`
  - this controls which split file is written for the temporary staged root

- `mask_source`
  - which mask directory to read from under `files/<mask_source>/obj_id_pkl/<scene_id>.pkl`
  - current common choice: `gt_projection`
  - later this can be changed to a predicted mask source

- `roots.baseline`
  - baseline dataset root used for compatibility files and background data
  - this usually stays the same across scenes

- `roots.reconstruction`
  - root that contains the reconstructed objects for this scene
  - must contain `files/gs_annotations/<scene_id>/`
  - this is the root produced by the current 2.5 / voxelise path

- `roots.pred_ready`
  - output root for the rebuilt downstream scene metadata
  - this is where `build-pred-ready` writes the new `files/objects.json` and `files/orig/data/<scene_id>.pkl.gz`

- `artifacts.manifest`
  - manifest JSON for the selected objects of this scene
  - used mainly by the final render / inspection bundle

- `artifacts.joint_ply`
  - expected path of the decoded joint output
  - usually looks like `vis/<scene_id>_joint.ply`

- `voxelise.log`, `slat.log`, `u3dgs.log`
  - log files for the main stages
  - use scene-specific names so runs do not overwrite each other

- `validate_pred_ready.out_dir`
  - folder for validation outputs

- `compare_arrangement.out_dir`
  - folder for arrangement comparison outputs

- `render_bundle.label`
  - short label used in the final visualization folder name
  - this should match the run variant, for example `supportonly`, `hybrid`, or `liftedonly`

#### Recommended Way To Fill It In

Start from a working example and only change the scene-specific parts first:

- `name`
- `scene_id`
- `roots.reconstruction`
- `roots.pred_ready`
- `artifacts.manifest`
- `artifacts.joint_ply`
- the log paths
- the output directories
- `render_bundle.label`

Keep the rest unchanged until the first dry-run works.

#### Minimal Example

This is the kind of scene-specific block you usually replace:

```json
{
  "name": "my_scene_profile",
  "scene_id": "NEW_SCENE_ID",
  "split": "val",
  "mask_source": "gt_projection",
  "roots": {
    "baseline": "/work/scratch/pafina/objectx-data-baseline",
    "reconstruction": "/work/scratch/pafina/objectx-data-fullscene-my-scene-hybrid-gtmask",
    "pred_ready": "/work/scratch/pafina/objectx-data-fullscene-my-scene-predready-v1"
  },
  "artifacts": {
    "manifest": "/work/scratch/pafina/object-x/debug/my_scene_fullscene_all_objects_manifest.json",
    "joint_ply": "/work/scratch/pafina/object-x/vis/NEW_SCENE_ID_joint.ply"
  }
}
```

#### How To Check The New Profile

After editing the JSON, always test it with dry-runs first:

```bash
bash scripts/workflows/run_scene_profile.sh <new_profile_name> voxelise --dry-run
bash scripts/workflows/run_scene_profile.sh <new_profile_name> build-pred-ready --dry-run
bash scripts/workflows/run_scene_profile.sh <new_profile_name> u3dgs --dry-run
bash scripts/workflows/run_scene_profile.sh <new_profile_name> render --dry-run
```

If those look correct, you can launch the real pipeline step by step.

### How To Disable TSDF

The current profiles use the hybrid object reconstruction path in the `voxelise` stage:

```json
"voxelise": {
  "env": {
    "OBJECTX_VOXEL_OBJECT_SOURCE": "hybrid_masks",
    "OBJECTX_VOXEL_TSDF_FALLBACK_TO_LIFTED": "1"
  }
}
```

If you want to disable TSDF and use only the lifted mask+depth+pose geometry, change the profile to:

```json
"voxelise": {
  "env": {
    "OBJECTX_VOXEL_OBJECT_SOURCE": "lifted_masks"
  }
}
```

This means:
- `hybrid_masks`: lifted voxel geometry + TSDF fusion
- `tsdf_masks`: TSDF-only geometry
- `lifted_masks`: lifted geometry only, with no TSDF step
- `gt_mesh`: old GT mesh path

After changing the profile, you can run the same command as before:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> voxelise
```

### Step-by-Step Pipeline Overview

The current workflow is easiest to think about as one linear pipeline driven by a profile:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> <action>
```

In practice, this means:
- the `profile` chooses the scene, roots, manifests, logs, and output folders
- the `action` chooses which pipeline stage to run
- you can run the whole pipeline step by step without rebuilding unrelated stages

This workflow starts **after** the earlier scene preprocessing is already available. In other words, the pipeline below assumes that the scene already has its standard Object-X input files on disk, and then adds the new reconstruction, pred-ready metadata, SLAT/U3DGS, and rendering stages on top.

For the current `cabinet` setup, the actions are:

1. `voxelise`
   - wrapper: [`scripts/voxel_annotations/pipeline/voxelise_features_tmp.sh`](scripts/voxel_annotations/pipeline/voxelise_features_tmp.sh)
   - runner: [`scripts/voxel_annotations/pipeline/run_scanwise_voxelise_tmp.py`](scripts/voxel_annotations/pipeline/run_scanwise_voxelise_tmp.py)
   - core logic: [`preprocessing/voxel_anno/voxelise_features.py`](preprocessing/voxel_anno/voxelise_features.py)
   - input: `files/objects.json`, `files/3RScan.json`, `files/<mask_source>/obj_id_pkl/<scene>.pkl`, `scenes/<scene>/sequence/frame-<id>.color.jpg`, `scenes/<scene>/sequence/frame-<id>.depth.pgm`, `scenes/<scene>/sequence/frame-<id>.pose.txt`, and `scenes/<scene>/sequence/_info.txt`
   - what happens: for each object, the masked depth pixels are lifted into 3D in world space using the camera poses; these lifted points are voxelized and optionally fused with TSDF to form object-level geometry
   - output: reconstructed object geometry under `files/gs_annotations/<scene>/<obj>/`, mainly `voxel_output_dense.npz` and `mean_scale_dense.npz`

2. `build-pred-ready`
   - script: [`scripts/segmentation/pipeline/build_pred_ready_scene_root.py`](scripts/segmentation/pipeline/build_pred_ready_scene_root.py)
   - input: `files/gs_annotations/<scene>/<obj>/voxel_output_dense.npz`, `files/gs_annotations/<scene>/<obj>/mean_scale_dense.npz`, plus baseline compatibility files such as `files/objects.json`, `files/orig/data/<scene>.pkl.gz`, `files/3RScan.json`, `files/Features3D/`, and `scenes/<scene>/sequence.zip` or `scenes/<scene>/sequence/`
   - what happens: the script turns each reconstructed object back into 3D points, computes object centers and sizes, chooses a root object, and writes a new scene description for the downstream pipeline
   - output: a separate pred-ready root with rebuilt `files/objects.json`, `files/orig/data/<scene>.pkl.gz`, linked `files/gs_annotations/<scene>/`, and a small manifest in `files/pred_ready_scene_manifest.json`

3. `validate-pred-ready`
   - script: [`scripts/segmentation/validation/validate_pred_ready_scene_root.py`](scripts/segmentation/validation/validate_pred_ready_scene_root.py)
   - input: the generated `files/objects.json`, `files/orig/data/<scene>.pkl.gz`, `files/gs_annotations/<scene>/`, and the staged scene under `scenes/<scene>/sequence/`
   - what happens: the script checks that object ordering is consistent, expected keys exist, point sets have the right shape, and the dataset loader can read the new root without crashing
   - output: a validation report and optional lightweight debug exports in the configured validation directory

4. `compare-arrangement`
   - script: [`scripts/segmentation/validation/compare_scene_arrangement.py`](scripts/segmentation/validation/compare_scene_arrangement.py)
   - input: `files/orig/data/<scene>.pkl.gz` from the pred-ready root and the matching `files/orig/data/<scene>.pkl.gz` from the baseline GT root
   - what happens: the script compares common objects between both roots using their reconstructed centers and pairwise distances, so we can check whether the rebuilt scene layout is still close to the baseline arrangement
   - output: metrics, plots, and an arrangement overlay in the configured comparison directory

5. `slat`
   - wrapper: [`scripts/inference/pipeline/run_pipeline_slat_tmp.sh`](scripts/inference/pipeline/run_pipeline_slat_tmp.sh)
   - staging helper: [`scripts/inference/pipeline/run_pipeline_tmp.py`](scripts/inference/pipeline/run_pipeline_tmp.py)
   - model entrypoint: [`src/inference/structured_latent_inference.py`](src/inference/structured_latent_inference.py)
   - input: the staged pred-ready root, mainly `files/objects.json`, `files/orig/data/<scene>.pkl.gz`, `files/gs_annotations/<scene>/`, `files/Features3D/`, and `scenes/<scene>/sequence/`
   - what happens: the structured Object-X model encodes the scene objects into a compact latent representation that is used by the downstream reconstruction step
   - output: `files/gs_embeddings/<scene>_slat.npz`

6. `u3dgs`
   - wrapper: [`scripts/inference/pipeline/run_pipeline_u3dgs_tmp.sh`](scripts/inference/pipeline/run_pipeline_u3dgs_tmp.sh)
   - staging helper: [`scripts/inference/pipeline/run_pipeline_tmp.py`](scripts/inference/pipeline/run_pipeline_tmp.py)
   - model entrypoint: [`src/inference/unstructured_latent_inference.py`](src/inference/unstructured_latent_inference.py)
   - input: `files/gs_embeddings/<scene>_slat.npz` plus the same staged root files used by `slat`, especially `files/orig/data/<scene>.pkl.gz`, `files/gs_annotations/<scene>/`, and `scenes/<scene>/sequence/`
   - what happens: the unstructured model decodes the latent into a joint Gaussian/point-based reconstruction for the whole scene; workflow profiles enable a default CPU fallback for the dense voxel decode stage so the run can continue without hard voxel cuts when GPU memory is too small
   - output: `files/gs_embeddings/<scene>_ulat.npz`, `vis/<scene>_joint.ply`, and `vis/rendered/<scene>_orbit_rendered.mp4`

7. `render`
   - wrapper: [`scripts/segmentation/visualization/render_joint_depth_background_bundle.sh`](scripts/segmentation/visualization/render_joint_depth_background_bundle.sh)
   - renderer: [`scripts/segmentation/visualization/render_joint_depth_background.py`](scripts/segmentation/visualization/render_joint_depth_background.py)
   - input: `vis/<scene>_joint.ply`, the replacement root `files/gs_annotations/<scene>/`, the manifest JSON for the selected objects, and the baseline scene files `scenes/<scene>/sequence/frame-<id>.depth.pgm`, `frame-<id>.pose.txt`, and `_info.txt`
   - what happens: the script combines the decoded joint reconstruction with a depth-based background, then produces an inspection bundle that is easier to browse than the raw model output alone
   - output: MP4, interactive HTML, contact sheet, and `summary.json`

In short, the current pipeline is:

```text
profile -> voxelise -> build-pred-ready -> validate/compare -> slat -> u3dgs -> render
```

## 📃 Abstract

Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g., images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy.Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.

## ⏩ Code Release
- [ ] Add code and instructions for evaluation
- [ ] Add code and instructions for baseline evaluation
- [ ] Release checkpoints and metadata for 3RScan and ScanNet

## 🔨 Installation Guide

The code is tested with the following dependencies

- **Operating System**: Ubuntu
- **Architecture**: x86_64 GNU/Linux
- **Python Version**: 3.9.18
- **CUDA Version**: 12.4
- **NVIDIA Driver Version**: 550.144.03
- **GPU**: NVIDIA A100 PCIe 40GB
- **Total GPU Memory**: Above 40GB

### Setting Up the Virtual Environment

1. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   *(Ensure that your virtual environment is activated before proceeding with the installation.)*

2. **Install dependencies from `requirements.txt`**:
   ```bash
   pip install -r requirements.txt # add [--no-deps] if installation causes dependency issues
   pip install -r other_deps.txt
   ```

3. **Install dependencies separately**
    ```bash
   pip install dependencies/gaussian-splatting
   pip install git+https://github.com/nerfstudio-project/gsplat.git # Needed for evaluation
   pip install dependencies/2d-gaussian-splatting # Needed for baselines evaluation
   ```

### Checking Installation

After installing the dependencies, verify your environment:

- **Check Python Version**:
  ```bash
  python --version
  ```
  *(Should output `Python 3.9.18`)*

- **Check CUDA & GPU Availability**:
  ```bash
  nvidia-smi
  ```
  *(Ensure the GPU is detected and available for computation.)*


## 🖥️ Cluster Setup (ETH Student Cluster, Recommended)

This section documents the exact workflow that we used to get this repository running on the ETH student cluster. It is intentionally more detailed than the generic installation guide above, because the cluster setup had a few important pitfalls:

- `requirements.txt` is not reliable as-is on the student cluster.
- `/home` and `/work/scratch` have different quota bottlenecks.
- CUDA extension builds must happen on a GPU node with a recent enough CUDA toolchain.
- this private repository already vendors the formerly nested dependencies, so collaborators should **not** use a submodule-based setup anymore.

If you are using **this private repository**, prefer the instructions in this section over the generic upstream setup.

### 0. What Is Different in This Private Repository?

This repository differs from the original upstream checkout in two important ways:

1. Third-party dependencies are already **vendored into the main repository**.
   - You do **not** need `git submodule update --init --recursive`.
   - You should clone this repo as a normal Git repository.
2. Several compatibility fixes that were needed on the cluster are already included in the codebase.
   - safer attention backend fallbacks (`sdpa` / `naive`)
   - compatibility fixes for the Gaussian Splatting camera and renderer APIs
   - compatibility fixes for TRELLIS path resolution
   - cluster helper scripts such as [`scripts/activate_objectx_env.sh`](scripts/activate_objectx_env.sh) and the workflow entrypoints under [`scripts/workflows/`](scripts/workflows/)

### 1. Storage, Quotas, and Where to Put Things

Before you install anything, be aware of the cluster storage limits. The main lessons from our setup were:

- `/home` is small and easy to fill up with virtual environments and pip caches.
- `/work/scratch/$USER` is the correct place for the repository, data, and outputs.
- `/work/scratch/$USER` is limited by **both** total size **and** file count.

At the time of writing, the relevant limits we hit were:

- `/home`: approximately **20 GB**
- `/work/scratch/$USER`: approximately **100 GB**
- `/work/scratch/$USER`: approximately **100000 files**

Practical consequences:

- keep the repository under `/work/scratch/$USER`
- keep only **one** active virtual environment for this project
- disable the pip cache
- use `/tmp` for temporary build files
- avoid creating duplicate environments (`.venv`, `.venv_objx`, `~/venvs/...`) unless you really need them

Useful quota checks:

```bash
quota -s
df -h /home/$USER
du -sh /work/scratch/$USER 2>/dev/null
find /work/scratch/$USER -xdev | wc -l
```

Useful cleanup commands:

```bash
rm -rf ~/.cache/pip
rm -rf /tmp/pip-* /tmp/pip-install-* /tmp/pip-req-build-* /tmp/tmp*
```

### 2. Clone the Repository Correctly

Because this repository uses Git LFS for `assets/teaser.png`, install Git LFS first and pull the LFS objects after cloning:

```bash
cd /work/scratch/$USER
git lfs install
git clone git@github.com:pascalfina/3d_vision_project.git object-x
cd object-x
git lfs pull
```

Important notes:

- Do **not** use `--recurse-submodules`.
- If a push later fails with a message about a missing LFS object, fetch the missing objects from upstream first:

```bash
git lfs fetch upstream --all
git push origin main
```

### 3. Use the Login Node and GPU Nodes for Different Jobs

Use the **login node** for:

- cloning the repo
- downloading datasets and metadata
- light preprocessing
- editing config files

Use a **GPU node** for:

- building CUDA extensions
- DINOv2 feature extraction
- any training or inference

We used the following interactive GPU allocation:

```bash
srun -A 3dv --qos=3dv-team35 -p jobs -t 24:00:00 --pty bash --login
```

Once you are on the node, check that a GPU is visible:

```bash
hostname
nvidia-smi
```

### 4. Recommended Environment Layout

The environment we ended up using successfully is:

- repository: `/work/scratch/$USER/object-x`
- environment: `/work/scratch/$USER/object-x/.venv_objx`
- dataset root: `/work/scratch/$USER/objectx-data`
- model/output root: `/work/scratch/$USER`
- temporary build files: `/tmp/$USER-objectx`

Create the environment:

```bash
cd /work/scratch/$USER/object-x
python3.9 -m venv .venv_objx
source .venv_objx/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
```

Set the build/runtime defaults:

```bash
export TMPDIR=/tmp/$USER-objectx
mkdir -p "$TMPDIR"
export PIP_NO_CACHE_DIR=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
```

### 5. Why We Do Not Install `requirements.txt` Blindly

On the student cluster, `pip install -r requirements.txt` caused multiple dependency conflicts and quota problems. The biggest ones we hit were:

- Jupyter / `jsonschema` resolver conflicts
- old MKL / NumPy pins
- conflicting `attrs` versions
- large optional packages that were not required for the Object-X core path

For that reason, this repository includes:

- [`requirements.cluster.txt`](requirements.cluster.txt)
- [`requirements.runtime.txt`](requirements.runtime.txt)

The cluster/runtime files are the safer starting point for this setup.

### 6. Install the Core Python Stack

Install the base packages first:

```bash
cd /work/scratch/$USER/object-x
source .venv_objx/bin/activate

python -m pip install --no-cache-dir numpy==1.23.5 scipy==1.9.3 attrs==25.4.0
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install --no-cache-dir xformers==0.0.32.post2 torch-geometric spconv-cu121==2.3.8 cumm-cu121==0.7.11
python -m pip install --no-cache-dir pccm ccimport pybind11 fire portalocker lark termcolor tqdm requests aiohttp psutil pyparsing ipython ipdb
python -m pip install --no-cache-dir --no-deps -r requirements.runtime.txt
```

### 7. Build `flash-attn` for ETH Blackwell GPUs (`sm_120`)

On the ETH student cluster we ended up needing a **source build** of `flash-attn` on a GPU node to get a wheel that actually targets our GPU architecture. The important check is that the build log contains:

```text
-gencode arch=compute_120,code=sm_120
```

You can confirm the GPU capability on the node with:

```bash
python - <<'PY'
import torch
print("gpu:", torch.cuda.get_device_name(0))
major, minor = torch.cuda.get_device_capability(0)
print("capability:", (major, minor))
print("target:", f"sm_{major}{minor}")
PY
```

The build sequence that worked for us was:

```bash
cd /work/scratch/$USER/object-x
source .venv_objx/bin/activate

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.8

unset LD_LIBRARY_PATH
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"

export SRC_PARENT=/tmp/$USER-flashattn
export SRC_DIR=$SRC_PARENT/src
export BUILD_TMP=$SRC_PARENT/buildtmp

rm -rf "$SRC_PARENT"
mkdir -p "$BUILD_TMP"

git clone --branch v2.8.2 --depth 1 https://github.com/Dao-AILab/flash-attention.git "$SRC_DIR"
cd "$SRC_DIR"

export TMPDIR="$BUILD_TMP"
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export NVCC_THREADS=2
export FORCE_CUDA=1
export FLASH_ATTN_CUDA_ARCHS="120"
export TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"

python - <<'PY'
from pathlib import Path
import re

p = Path("setup.py")
s = p.read_text()
s2 = re.sub(
    r'os\\.getenv\\("FLASH_ATTN_CUDA_ARCHS",\\s*"[^"]*"\\)',
    'os.getenv("FLASH_ATTN_CUDA_ARCHS", "120")',
    s,
)
if s != s2:
    p.write_text(s2)
    print("patched setup.py default arch -> 120")
else:
    print("setup.py arch default already compatible")
PY

python -m pip uninstall -y flash-attn
python -m pip install -v --no-cache-dir --no-build-isolation . 2>&1 | tee /work/scratch/$USER/object-x/flash_attn_build.log
```

Then verify the build log:

```bash
rg -n "compute_120|sm_120" /work/scratch/$USER/object-x/flash_attn_build.log
```

### 8. Always Activate Through the Helper Script

After the environment exists, prefer:

```bash
cd /work/scratch/$USER/object-x
source scripts/activate_objectx_env.sh
```

This script already does the things that mattered for us on the cluster:

- loads `cuda/12.8`
- activates `.venv_objx` (or `.venv` as a fallback)
- unsets `LD_LIBRARY_PATH`
- sets `TMPDIR`
- sets `CUDA_HOME`
- adds the repository and Gaussian Splatting code to `PYTHONPATH`

Why this matters:

- keeping an old `LD_LIBRARY_PATH` around caused PyTorch CUDA loader failures for us

### 9. Attention Backends We Recommend on the Cluster

Once `flash-attn` is installed, the dense attention path should resolve to `flash_attn` automatically.

For the ETH Blackwell GPUs we tested on, the sparse `xformers` attention path still crashed with:

```text
CUDA error (.../xformers/third_party/flash-attention/hopper/flash_fwd_launch_template.h:188): invalid argument
```

The stable runtime compromise for us was therefore:

- dense attention: `flash_attn`
- sparse attention: `sdpa`

Use that by exporting:

```bash
export SPARSE_ATTN_BACKEND=sdpa
```

You can verify the active backends with:

```bash
python - <<'PY'
import src.modules.attention as a
import src.modules.sparse as s
print("dense backend:", a.BACKEND)
print("sparse backend:", s.ATTN)
PY
```

On our working cluster setup this prints:

```text
dense backend: flash_attn
sparse backend: sdpa
```

### 10. Build the CUDA Extensions on a GPU Node

Do this only on a GPU node:

```bash
cd /work/scratch/$USER/object-x
source scripts/activate_objectx_env.sh

python -m pip install --no-cache-dir --no-build-isolation dependencies/gaussian-splatting/submodules/simple-knn
python -m pip install --no-cache-dir --no-build-isolation dependencies/gaussian-splatting/submodules/diff-gaussian-rasterization
```

Then verify the full environment:

```bash
bash scripts/check_objectx_env.sh
python -c "import src.trainval.train_latent_autoencoder; print('latent train import ok')"
```

### 11. TRELLIS Checkpoint Download

The latent autoencoder depends on the TRELLIS checkpoint tree. Download it once to `SCRATCH`:

```bash
cd /work/scratch/$USER/object-x
source .venv_objx/bin/activate

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JeffreyXiang/TRELLIS-image-large",
    local_dir=f"/work/scratch/{os.environ['USER']}/TRELLIS-image-large",
    resume_download=True,
)
PY
```

The current code expects:

```bash
/work/scratch/$USER/TRELLIS-image-large
```

and the `SCRATCH` environment variable should point to `/work/scratch/$USER`.

### 12. Dataset Layout on the Cluster

For the real dataset layout, follow the dataset generation section below. The important cluster-specific recommendation is to keep the root under scratch:

```bash
mkdir -p /work/scratch/$USER/objectx-data/{files,scenes}
export DATA_ROOT_DIR=/work/scratch/$USER/objectx-data
```

We validated the pipeline first with a **tiny 3RScan smoke-test subset** before moving to the full dataset. This was useful because it let us test the entire training stack without waiting for a full data download.

For the full setup you still need:

1. **3RScan**
2. **3DSSG**
3. **Additional Meta Files**
4. optional ScanNet / SceneGraphFusion data if you work on the ScanNet path

### 13. Minimal 3RScan Smoke-Test Subset

We used the official 3RScan toolkit to bootstrap a tiny sample:

```bash
cd /work/scratch/$USER
git clone https://github.com/WaldJohannaU/3RScan.git 3RScan-toolkit
cd 3RScan-toolkit
bash setup.sh
```

This downloads:

- `3RScan.json`
- `objects.json`
- `relationships.json`
- one reference scan and one rescan

We then linked that into our `DATA_ROOT_DIR` and added the missing metadata files from the official Object-X "Additional Meta Files" folder.

### 14. Preprocessing Notes

The preprocessing pipeline is split across CPU-friendly and GPU-heavy steps.

Recommended split:

- login node / light shell:
  - metadata checks
  - symlink creation
  - light preprocessing
- GPU node:
  - DINOv2 feature generation
  - any CUDA builds

For the VLSG preprocessing, the environment that worked for us was:

```bash
cd /work/scratch/$USER/object-x
source .venv_objx/bin/activate

export DATA_ROOT_DIR=/work/scratch/$USER/objectx-data
export Data_ROOT_DIR="$DATA_ROOT_DIR"
export VLSG_SPACE=/work/scratch/$USER/object-x/dependencies/VLSG
export PYTHONPATH="$VLSG_SPACE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
```

Then run the preprocessing steps from the dataset section below.

### 15. Running the Project

We use small shell wrappers to make the cluster workflow less fragile.

Main files:

- [`scripts/activate_objectx_env.sh`](scripts/activate_objectx_env.sh)
- [`scripts/workflows/run_scene_profile.sh`](scripts/workflows/run_scene_profile.sh)
- [`scripts/check_objectx_env.sh`](scripts/check_objectx_env.sh)
- [`configs/objectx_runner.env`](configs/objectx_runner.env)
- [`configs/objectx_train_params.env`](configs/objectx_train_params.env)

Typical workflow:

```bash
cd /work/scratch/$USER/object-x
source scripts/activate_objectx_env.sh
export DATA_ROOT_DIR=/work/scratch/$USER/objectx-data
export SCRATCH=/work/scratch/$USER

bash scripts/check_objectx_env.sh
bash scripts/train_val/train_latent_autoencoder.sh
```

For Slurm batch mode:

```bash
sbatch scripts/slurm/objectx_job.sbatch
```

The helper config [`configs/objectx_train_params.env`](configs/objectx_train_params.env) lets you override common training settings such as the maximum number of epochs. For quick smoke tests, we used:

```bash
TRAIN_MAX_EPOCH="10"
```

### 16. Common Errors We Hit and How We Fixed Them

Below is a summary of the issues we actually encountered on the cluster.

| Symptom | Root cause | Fix |
|---|---|---|
| `Disk quota exceeded` under `/home/...` | environment and pip cache were stored in `/home` | move the repo and environment to `/work/scratch/$USER`, delete `~/.cache/pip`, keep only one environment |
| `Disk quota exceeded` under `/work/scratch/...` even though plenty of GB were free | `/work/scratch` also has a file-count quota | do not create duplicate environments, disable pip cache, use `/tmp` for temporary build files, monitor `find /work/scratch/$USER -xdev | wc -l` |
| `ResolutionImpossible` when installing `requirements.txt` | upstream requirements include conflicting Jupyter / MKL / legacy pins | use `requirements.runtime.txt` / `requirements.cluster.txt` and install the core stack explicitly |
| `ImportError: ... libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion` | wrong CUDA runtime was loaded through `LD_LIBRARY_PATH` | unset `LD_LIBRARY_PATH`; use `scripts/activate_objectx_env.sh` instead of ad-hoc environment variables |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | the system `nvcc` was too old for the GPU architecture we were using | build on a GPU node and load `cuda/12.8` |
| `flash-attn` builds, but the log only shows `sm_80` / `sm_90` | the build did not actually target the Blackwell GPU architecture | build from source on a GPU node, force `FLASH_ATTN_CUDA_ARCHS="120"`, and verify that the log contains `compute_120` / `sm_120` |
| `CUDA error (... xformers ... flash_fwd_launch_template.h:188): invalid argument` | sparse `xformers` attention was unstable on our `sm_120` cluster GPUs | keep dense `flash_attn`, export `SPARSE_ATTN_BACKEND=sdpa`, and only revisit sparse `xformers` after moving to a newer Torch/xFormers stack |
| `Directory 'dependencies/gaussian-splatting' is not installable` | the parent folder is not a Python package | install `simple-knn` and `diff-gaussian-rasterization` separately |
| `ModuleNotFoundError: diff_gaussian_rasterization` or `simple_knn` | CUDA extensions were not built yet | install the two subpackages explicitly on a GPU node |
| `No module named 'plotly'` / `No module named 'dash'` when importing Open3D | Open3D tried to load optional Plotly visualization modules | either install the optional Python packages or strip the Plotly/Dash visualization path if you only need core geometry functionality |
| `RuntimeError: No CUDA devices available` during DINOv2 feature generation | DINOv2 preprocessing was started on a login node | run the feature generation scripts on a GPU node |
| `FileNotFoundError` for `scannet8_relationships.txt`, `relationships.txt`, `scannet40_classes.txt`, etc. | the Additional Meta Files were missing from `DATA_ROOT_DIR/files` | download the official Object-X Additional Meta Files and place them under `files/` |
| `ValueError: num_samples=0` in the toy subset | the tiny smoke-test subset did not contain the full expected set of preprocessing outputs | use the repo with the current compatibility fixes and treat the toy subset as a pipeline smoke test, not a final training setup |
| `Git LFS upload failed` when pushing the private repo | an older LFS object from the imported history was missing locally | run `git lfs fetch upstream --all` and push again |

### 17. Recommended Order of Operations

If you want the shortest path to a working setup on the cluster, this is the order we recommend:

1. clone this private repo with Git LFS
2. create `.venv_objx` on `/work/scratch/$USER`
3. install the core Python stack
4. build the two Gaussian Splatting CUDA extensions on a GPU node
5. run `bash scripts/check_objectx_env.sh`
6. download TRELLIS to `/work/scratch/$USER/TRELLIS-image-large`
7. prepare `DATA_ROOT_DIR`
8. run the required preprocessing
9. start with a short smoke-test run (`TRAIN_MAX_EPOCH="10"`)
10. only then move to longer jobs and the full dataset


## 🪑 Dataset Generation

### 1. Downloading the Datasets
This section outlines the required datasets and their organization within the root directory.

#### **1.1 3RScan, 3DSSG, and ScanNet**
1. **3RScan**: Download from [here](https://github.com/WaldJohannaU/3RScan) and move all files to `\<root_dir>/scenes/`.
2. **3DSSG**: Download from [here](https://3dssg.github.io/) and place all files in `\<root_dir>/files/`.
3. **ScanNet**: Download from [here](http://www.scan-net.org) and move the scenes to `\<root_dir>/scenes/`.
4. **Additional Meta Files**: Download from [this link](https://drive.google.com/drive/folders/1pdZsvAqsVjTkRbNuR3xMDMlkDnf-yyS6?usp=share_link) and move them to `\<root_dir>/files/`.

After this step, the directory structure should look as follows:

```
├── <root_dir>
│   ├── files                 <- Meta files and annotations
│   │   ├── <meta_files_0>
│   │   ├── <meta_files_1>
│   │   ├── ...
│   ├── scenes                <- Scans (3RScan/ScanNet)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
```

---

### 2. Preprocessing 3RScan Dataset

#### **2.1 Generating `labels.instances.align.annotated.v2.ply`**
To generate `labels.instances.align.annotated.v2.ply` for each 3RScan scan, refer to the repository:
[3DSSG Data Processing](https://github.com/ShunChengWu/3DSSG/blob/master/data_processing/transform_ply.py).

#### **2.2 Preprocessing Scene Graph Information**
1. The preprocessing code for 3RScan is located in the [dependencies/VLSG](dependencies/VLSG) directory.
2. Ensure the following environment variables are set:
   - `VLSG_SPACE` = Repository path
   - `DATA_ROOT_DIR` = Path to the downloaded dataset (i.e., `root_dir`)
   - `CONDA_BIN` = `.venv/bin` (as linked in installation)
3. Execute the preprocessing script:
   ```bash
   cd dependencies/VLSG && bash scripts/preprocess/scan3r_data_preprocess.sh
   ```

#### **2.3 Generating Ground Truth Patch-Object Annotation**
Run the following command to generate pixel-wise and patch-level ground truth annotations:
```bash
cd dependencies/VLSG && bash scripts/gt_annotations/scan3r_gt_annotations.sh
```

#### **2.4 Generating Patch-Level Features (Optional: Image Localization Training)**
Precompute patch-level features using [Dino v2](https://dinov2.metademolab.com/):
```bash
cd dependencies/VLSG && bash scripts/features2D/scan3r_dinov2.sh
```

#### **2.5 Generating Featured Voxel Annotations**
Run the following command to generate featured voxel annotations:
```bash
bash scripts/voxel_annotations/pipeline/voxelise_features_tmp.sh --split {split}
# 2.5.1 Generating Subscenes Annotations (Optional)
# Follow the instructions from SGAligner at dependencies/sgaligner to create subscenes annotations
# Then run the following command:
# bash scripts/voxel_annotations/variants/voxelise_features_scene_alignment.sh --split {split}
```

#### **2.6 Generating Gaussian Splat Annotations (Optional: Baseline Computation)**
Generate Gaussian splat annotations using the following commands:
```bash
bash scripts/gs_annotations/map_to_colmap.sh --split {split}
bash scripts/gs_annotations/annotate_gaussians.sh --split {split}
```

---

### 3. Preprocessing ScanNet Dataset

ScanNet requires scene graph annotations generated using [SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion).

#### **3.1 Download and Set Up SceneGraphFusion**
1. Download the pretrained model from [here](https://drive.google.com/file/d/1_745ofaOUyP_iFK8A3cSW60L4V7TlWa7/view) and move it to `dependencies/SCENE-GRAPH-FUSION/`.
2. Build SceneGraphFusion by following the instructions in its [repository](https://github.com/ShunChengWu/SceneGraphFusion?tab=readme-ov-file#prerequisites).

#### **3.2 Generating Scene Graph Annotations for ScanNet**
Run the following commands:
```bash
python preprocessing/scene_graph_anno/scenegraphfusion_prediction.py
python preprocessing/scene_graph_anno/scenegraphfusion2scan3r.py
```

#### **3.3 Generating Ground Truth Patch-Object Annotation**
Generate ground truth annotations with:
```bash
python preprocessing/gt_anno/scannet_obj_projector.py
```

#### **3.4 Generating Featured Voxel Annotations**
Run the following script to generate featured voxel annotations:
```bash
bash scripts/voxel_annotations/variants/voxelise_features_scannet.sh
```

#### **3.5 Generating Gaussian Splat Annotations (Optional: Baseline Computation)**
To generate Gaussian splat annotations, execute:
```bash
bash scripts/gs_annotations/map_to_colmap_scannet.sh
bash scripts/gs_annotations/annotate_gaussians_scannet.sh
```

---
### Outline
After the above preprocessing, the directory structure should look as follows:

```
├── <root_dir>
│   ├── files                      <- Meta files and annotations
│   │   ├── Features2D             <- (Step 2.4)
│   │   ├── gt_projection          <- (Step 2.3)
│   │   ├── orig                   <- (Step 2.1)
│   │   ├── patch_anno             <- (Step 2.3)
│   │   ├── gs_annotations         <- (Step 2.5/2.6)
│   │   ├── gs_annotations_scannet <- (Step 3.4/3.5)
│   │   ├── <meta_files_0>
│   │   ├── <meta_files_1>
│   │   ├── ...
│   ├── scenes                     <- Scans (3RScan/ScanNet)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
│   ├── scene_graph_fusion         <- (Step 3.1)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
│   ├── out                        <- (Step 2.5.1)
│   │   ├── files ...
│   │   ├── scenes ...
```

## 🏃‍♀️ Training

Refer to the [TRAIN.md](TRAIN.md) for training instructions.

## 📕 BibTeX 
```
@misc{dilorenzo2025objectxlearningreconstructmultimodal,
      title={Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations}, 
      author={Gaia Di Lorenzo and Federico Tombari and Marc Pollefeys and Daniel Barath},
      year={2025},
      eprint={2506.04789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04789}, 
}
 ```

## ♻️ Acknowledgments

In this project we use (parts of) the official implementations of the following works:

- SceneGraphLoc: [SceneGraphLoc](https://github.com/y9miao/VLSG)
- Trellis: [Trellis](https://github.com/Microsoft/TRELLIS)
- SGAligner: [SGAligner](https://github.com/sayands/sgaligner)
- GSplat: [GSplat](https://github.com/nerfstudio-project/gsplat)
- 2D Gaussian Splatting: [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- 3DSSG: [3DSSG](https://3dssg.github.io/)
- SceneGraphFusion: [SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion)
