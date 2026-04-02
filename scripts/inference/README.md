# Inference Scripts Layout

This folder is now split by purpose.

## `pipeline/`

Main inference entrypoints and wrappers.

- `run_pipeline_tmp.py`
- `run_pipeline_slat_tmp.sh`
- `run_pipeline_u3dgs_tmp.sh`

## `experiments/`

Specialized or less central helpers.

- `rerender_object_level_bright.sh`

## `archive/`

Older direct wrappers kept only for reference.

- `run_pipeline_slat.sh`
- `run_pipeline_u3dgs.sh`

## Recommended entrypoint

Prefer the workflow wrapper for the current stable runs:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> slat
bash scripts/workflows/run_scene_profile.sh <profile> u3dgs
```
