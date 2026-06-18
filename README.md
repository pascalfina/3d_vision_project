# Object-X Current Workflow

This repository currently uses a profile-based workflow. For the branch `Sam2Object_and_Pi3X`, the relevant setup is the kitchen/oven profile:

`configs/workflows/scene_profiles/oven_legacy_sam2_pi3x.json`

This file contains the parameters for the current oven scene pipeline, including the settings for `must3r`, `pi3x`, `samobject`, `voxelise`, `build-pred-ready`, and the visualization steps. If you want to change the run behavior for this scene, this is the main file to edit.

## Running The Workflow

In general, you do not need to call Python files directly. Use the workflow wrapper:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> <action>
```

For the current oven scene, the profile is:

```bash
oven_legacy_sam2_pi3x
```

The current execution order is:

1. Run MUSt3R first, because the current Pi3X setup uses the MUSt3R poses as external pose priors:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x must3r
```

2. Run Pi3X:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x pi3x
```

3. Optionally inspect the raw geometry before continuing:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x geom-debug
```

This writes an interactive HTML plus geometry exports so you can inspect the reconstruction from poses and depth before segmentation.

4. Run SAM2Object:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x samobject
```

5. Run voxelisation:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x voxelise
```

6. Build the pred-ready scene root:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x build-pred-ready
```

7. Visualize the segmented point cloud:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x plot-voxelised
```

This generates the interactive HTML for the voxelised and segmented point cloud.

## Optional SAM2-Only Stage

If you want to run the SAM2 stage from the workflow as well, the action is:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x segment-inputs
```

For the current Pi3X + SAM2Object workflow this is less central, but the action is still available.

## Downstream Object-X Stages

If you want to continue after `build-pred-ready`, the remaining workflow actions are:

1. Build object visual features:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x features3d
```

2. Run SLAT:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x slat
```

3. Run U3DGS:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x u3dgs
```

4. Render the final output bundle:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x render
```

So the full practical sequence for the current branch is:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x must3r
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x pi3x
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x geom-debug
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x samobject
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x voxelise
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x build-pred-ready
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x plot-voxelised
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x features3d
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x slat
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x u3dgs
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x render
```

## Switching The SAM2Object Checkpoint

You can change the SAM2Object checkpoint without editing Python code by setting:

```bash
export SAMOBJECT_CHECKPOINT=/work/scratch/pafina/object-x/models/sam2ckpt/sam2_hiera_base_plus.pt
export SAMOBJECT_MODEL_CFG=sam2_hiera_b+.yaml
```

Then run:

```bash
bash scripts/workflows/run_scene_profile.sh oven_legacy_sam2_pi3x samobject
```

Adjust the paths if your local setup differs.

## What The Main Config Sections Mean

The profile file `oven_legacy_sam2_pi3x.json` is split into sections:

- `must3r`: MUSt3R pose/depth run used as prior for Pi3X.
- `pi3x`: Pi3X geometry run and related Pi3X parameters.
- `segment_inputs`: optional SAM2-only stage.
- `samobject`: SAM2Object parameters, checkpoint settings, and scene preparation.
- `voxelise`: object voxelisation from the segmented geometry.
- `build_pred_ready`: rebuilds the pred-ready scene snapshot from the voxelised objects.
- `features3d`, `slat`, `u3dgs`: downstream Object-X latent stages.
- `render_bundle` / `plot-voxelised` / `render`: visualization settings and final outputs.

So in practice, if you want to tune one stage, you usually only need to edit the corresponding block in this JSON.

## Notes

- The current Pi3X path is not fully standalone in this profile: it expects MUSt3R to run first.
- The workflow wrapper is the intended entrypoint. Avoid calling individual stage scripts directly unless you are debugging internals.
- The current README intentionally focuses only on the active workflow used on this branch. Older instructions were removed because they were no longer representative of the current setup.
