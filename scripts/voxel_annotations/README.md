# Voxel Annotation Scripts Layout

This folder is now split by purpose.

## `pipeline/`

Current main voxelisation / 2.5 entrypoints.

- `run_scanwise_voxelise_tmp.py`
- `voxelise_features_tmp.sh`

## `variants/`

Alternative wrappers that are not the main Scan3R path.

- `voxelise_features_scannet.sh`
- `voxelise_features_scene_alignment.sh`

## `utils/`

Small maintenance helpers.

- `recompress_npz.py`

## `archive/`

Older direct wrappers kept only for reference.

- `voxelise_features.sh`

## Recommended entrypoint

Prefer the workflow wrapper for the current stable runs:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> voxelise
```
