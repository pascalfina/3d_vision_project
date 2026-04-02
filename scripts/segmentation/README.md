# Segmentation Scripts Layout

This folder is now split by purpose instead of keeping all scripts in one flat list.

## `pipeline/`

Main pipeline helpers that feed the current Object-X path.

- `build_pred_ready_scene_root.py`
- `clean_projection_masks.py`

## `validation/`

Checks and diagnostics that validate reconstructed objects or scene metadata.

- `validate_pred_ready_scene_root.py`
- `compare_scene_arrangement.py`
- `diagnose_object_gaps.py`
- `worldspace_object_debug.py`

## `visualization/`

Scene/object inspection, HTML export, and MP4 rendering utilities.

- `render_depth_background.py`
- `render_joint_depth_background.py`
- `render_joint_depth_background_bundle.sh`
- `render_scene_mesh_background.py`
- `render_predseg_debug.py`
- `export_depth_background_interactive.py`
- `export_scene_mesh_background_interactive.py`
- `export_slat_depth_background_interactive.py`

## `experiments/`

Reserved for future active experiments that are not part of the main path.

## `archive/`

Older builders / selection files that are kept for reference but are not part of the current workflow.

- `bootstrap_pred_projection_from_gt.py`
- `build_mixed_scene_root.py`
- `build_object_level_pilot_root.py`
- `build_partial_context_mixed_root.py`
- archived selection JSON files

## Recommended entrypoint

For the stable current workflow, prefer:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> <action>
```

Example:

```bash
bash scripts/workflows/run_scene_profile.sh cabinet_predready_v2_floorfix u3dgs
```
