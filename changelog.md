# Changelog / Current Working Notes

Short version:

- objects can now come from our own mask + depth + pose path instead of directly from GT-style object geometry
- we added better debugging for lifting, fusion, decode, and rendering
- we changed existing Object-X files, especially the 2.5 voxelisation path, so mask source switching and downstream inspection are now much easier
- this file lists the main code touch points and the commands we currently use

## 1. Biggest changes for GT-mesh replacement

Old behavior:

- object geometry could come directly from baseline / GT-style object annotations
- scene-level debugging was hard because failures in lifting, fusion, decode, and rendering all looked mixed together

New behavior:

- object geometry can now come from our own path based on mask + depth + pose
- we added object-level debugging in world space to verify whether lifted objects are placed correctly before full scene rendering
- we added diagnostics to separate:
  - missing visible information
  - lifting / mask / depth problems
  - fusion loss
  - downstream decode / rendering loss
- we added a more robust U3DGS decode path:
  - chunked decode by object
  - retry logic for occupancy / voxel limits
  - support-constrained cleanup that keeps decoded gaussians near the original sparse support

Important nuance:

- in the current best full-scene hybrid roots, the replaced object geometry under `files/gs_annotations/<scene>/<obj>/` is no longer just baseline GT object geometry
- however, some experiments still use GT masks (`gt_projection`) and some dataset metadata is still GT-derived (`objects.json`, `files/orig`, `Features3D`)

Main code touch points:

- `scripts/voxel_annotations/run_scanwise_voxelise_tmp.py`
  - new 2.5 path that can use `OBJECTX_MASK_SOURCE`
- `scripts/segmentation/worldspace_object_debug.py`
  - world-space object overlay debug
- `scripts/segmentation/diagnose_object_gaps.py`
  - object-level gap diagnosis
- `src/inference/unstructured_latent_inference.py`
  - chunked encode / decode
  - decode retry
  - support-constrained cleanup
- `scripts/segmentation/render_joint_depth_background.py`
  - MP4 + HTML inspection for final U3DGS joint output
- `scripts/segmentation/render_joint_depth_background_bundle.sh`
  - wrapper for the above

## 1.1 Original Object-X files we actually changed

This is the important distinction:

- the files below are not newly added helpers
- they are original Object-X files that already existed and that we changed during this work
- if someone wants to understand the real Object-X modifications, they should start here and not with the helper scripts

### `preprocessing/voxel_anno/voxelise_features.py`

This is the single most important original file for the GT-mesh replacement work.

Before:

- object voxel features were built around the old object geometry path
- the file mainly assumed the standard object annotations and mask handling

What we changed:

- added mask-source switching so the preprocessing step can use different mask folders instead of assuming one fixed GT mask source
- added multiple object-source modes:
  - `gt_mesh`
  - `lifted_masks`
  - `tsdf_masks`
  - `hybrid_masks`
- added the actual mask + depth + pose lifting path:
  - masked depth lifting
  - raw pinhole coordinate handling
  - pose handling / pose inversion helpers
  - point filtering and normalization
- added object fusion variants:
  - direct lifted voxel build
  - TSDF-based object fusion
  - hybrid fusion
- added frame-selection logic so we can compare `k` views and choose cleaner subsets instead of blindly taking everything
- added mask cleaning and visibility checks so only actually observed / consistent geometry is kept
- added exports for debug artifacts such as lifted point clouds and TSDF meshes
- changed some saving logic to be more robust for large runs

Why this matters:

- this is where object geometry stopped being tied to the old GT-style object geometry path
- this file is the real heart of the new "2.5" reconstruction path
- if the reconstructed object looks wrong, the bug is very often in logic that now lives here

### `utils/scan3r.py`

Before:

- mask loading largely assumed the default GT-style mask directory layout

What we changed:

- added `resolve_mask_source(...)`
- added `get_mask_dir(...)`
- updated mask loading helpers so they can explicitly take `mask_source`
- updated object-to-frame lookup helpers so the selected mask source propagates through the pipeline

Why this matters:

- this is the core utility that made `gt_projection` vs `pred_projection_clean` switching possible
- without this change, the rest of the pipeline would still silently snap back to the old GT mask folder assumptions

### `scripts/voxel_annotations/voxelise_features.sh`

Before:

- the old wrapper had its own environment bootstrap logic

What we changed:

- switched it to the shared `scripts/activate_objectx_env.sh` activation path

Why this matters:

- this made the preprocessing step much more reproducible across our new roots and debug runs
- it also reduced stupid environment drift between the original path and the new temp-root / debug workflows

### `src/datasets/scan3r_scene.py`

Before:

- the loader already defined much of the scene-level metadata Object-X consumes
- single-line split files could behave badly because `np.genfromtxt(...)` can collapse to a scalar

What we changed:

- wrapped the split-file load with `np.atleast_1d(...)` so one-scan custom split files still work

Why this matters:

- this looks small, but it was important for our targeted scene-wise runs
- many of our custom roots and debug selections only use one scene, so this fix made those targeted experiments stable
- this file is also still the main pointer to the remaining GT-derived scene metadata that we have not yet removed

### `src/inference/unstructured_latent_inference.py`

This is the most important original downstream file we changed after voxelisation.

Before:

- decode was much more monolithic
- if the decode was too large, it could OOM or force very blunt fallbacks
- the decoded gaussians could drift away from the original sparse support and create visually misleading blobs

What we changed:

- added chunked processing for encode / decode so large scenes can be handled object-wise instead of all-at-once
- added retry logic for sparse decode with fallback occupancy thresholds and voxel caps
- added support extraction from the original sparse representation
- added support-constrained cleanup after decode so gaussians are filtered back toward the original support
- added helpers for masking, slicing sparse batches, empty-gaussian handling, and chunked object decode
- added a lot of render-side controls through environment variables:
  - pruning
  - orbit settings
  - background color
  - exposure / gamma
  - render scale
  - mesh export toggles

Why this matters:

- this is where we fixed the big downstream failure mode where the object looked okay before SLAT/U3DGS but much worse afterward
- the later cabinet improvements came largely from changes in this file, especially support-constrained cleanup and safer decode behavior

### `utils/visualisation.py`

Before:

- PCA-based debug coloring was brittle for tiny or degenerate point sets

What we changed:

- hardened PCA colorization for debug exports
- handled empty / tiny feature sets more safely
- prevented divide-by-zero and bad PCA channel assumptions during PLY / point export

Why this matters:

- this reduced annoying crashes exactly in the phase where we needed many quick debug exports for tiny partial objects
- it made the debug artifacts much more reliable when inspecting broken objects

These are the original files I would tell someone to read first if they want to understand the actual modified Object-X internals:

- `preprocessing/voxel_anno/voxelise_features.py`
- `src/inference/unstructured_latent_inference.py`
- `utils/scan3r.py`
- `src/datasets/scan3r_scene.py`

Separate from that, we also added several new helper scripts around these core files, for example:

- `scripts/voxel_annotations/run_scanwise_voxelise_tmp.py`
- `scripts/inference/run_pipeline_tmp.py`
- `scripts/segmentation/worldspace_object_debug.py`
- `scripts/segmentation/diagnose_object_gaps.py`
- `scripts/segmentation/render_joint_depth_background_bundle.sh`

## 2. What the main new debug scripts do

### `scripts/segmentation/worldspace_object_debug.py`

Purpose:

- debug one object in world space
- compare:
  - GT voxels
  - lifted points
  - fused voxels

Useful when:

- object is clearly misplaced
- object has much fewer voxels than GT
- we want to know whether the main bug is pose / lifting / fusion

Typical outputs:

- `triptych.png`
- `overlay.png`
- `overlay_world.ply`
- `lifted_points_world.ply`
- `fused_voxels_world.ply`

Typical command template:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/worldspace_object_debug.py \
  --data-root <pred_or_hybrid_root> \
  --baseline-root /work/scratch/pafina/objectx-data-baseline \
  --scan-id <scan_id> \
  --obj-id <obj_id> \
  --label <label> \
  --mask-source pred_projection_clean \
  --pose-mode raw \
  --lift-coord-system pinhole \
  --frame-selection diverse_area \
  --max-views 12 \
  --object-source hybrid_masks
```

### `scripts/segmentation/diagnose_object_gaps.py`

Purpose:

- compute object-level metrics across several `k` values
- tell us whether the issue is mostly:
  - `missing_information`
  - `lifting_or_mask_depth`
  - `fusion_loss`
  - `mixed_or_downstream`

Useful when:

- we want to compare 6 vs 12 vs more views
- we want to know if adding views helps or hurts
- we want a cleaner read than a single screenshot

Typical outputs:

- per-object metrics JSON
- aggregate plot

Typical command template:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/diagnose_object_gaps.py \
  --data-root <pred_or_hybrid_root> \
  --baseline-root /work/scratch/pafina/objectx-data-baseline \
  --selection-file <selection_json> \
  --mask-source pred_projection_clean \
  --frame-selection diverse_area \
  --object-source hybrid_masks \
  --k 6 \
  --k 12 \
  --out-dir <out_dir>
```

### `scripts/segmentation/render_predseg_debug.py`

Purpose:

- visualize 2D masks on frames
- good for quick sanity checks on mask quality before going into 3D

### `scripts/segmentation/clean_projection_masks.py`

Purpose:

- clean raw projected / predicted masks
- creates a cleaner mask source such as `pred_projection_clean`

Typical command template:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/clean_projection_masks.py \
  --data-root <root> \
  --source pred_projection \
  --target pred_projection_clean
```

### `scripts/segmentation/render_depth_background.py`

Purpose:

- render a scene using a background reconstructed from depth + pose
- no GT scene mesh background
- replacement objects come from `gs_annotations`

### `scripts/segmentation/render_joint_depth_background.py`

Purpose:

- render the final U3DGS joint PLY together with the depth-based background
- outputs both MP4 and HTML

### `scripts/segmentation/render_joint_depth_background_bundle.sh`

Purpose:

- convenience wrapper around `render_joint_depth_background.py`
- activates env
- auto-resolves joint PLY and manifest when possible
- good final inspection command for a scene

## 3. The new "2.5" path

This is the step that generates featured voxel annotations from our chosen mask source.

Main implementation:

- `scripts/voxel_annotations/run_scanwise_voxelise_tmp.py`
- wrapper: `scripts/voxel_annotations/voxelise_features_tmp.sh`

What it produces:

- `files/gs_annotations/<scan_id>/<obj_id>/voxel_output_dense.npz`
- `files/gs_annotations/<scan_id>/<obj_id>/mean_scale_dense.npz`

Important behavior:

- `OBJECTX_MASK_SOURCE` controls which mask folder is used
- this lets us switch from `gt_projection` to `pred_projection_clean`
- internally the chosen mask source is also aliased to `gt_projection` in tmp because downstream code expects that folder name

Generic command:

```bash
cd /work/scratch/pafina/object-x

export DATA_ROOT_DIR=<your_scene_root>
export OBJECTX_MASK_SOURCE=pred_projection_clean
export SPLIT=val
export RESET_TMP=1
export MAX_SCANS=0

bash scripts/voxel_annotations/voxelise_features_tmp.sh
```

Current cabinet-style example:

```bash
cd /work/scratch/pafina/object-x

export DATA_ROOT_DIR=/work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-gtmask
export OBJECTX_MASK_SOURCE=gt_projection
export SPLIT=val
export RESET_TMP=1

bash scripts/voxel_annotations/voxelise_features_tmp.sh
```

## 4. How to get the object image features

There are two different "feature" concepts here:

1. `Features3D/obj_dinov2_top10_l3`
   - object visual features used by the dataset / Object-X side
2. `gs_annotations`
   - featured voxel annotations from step 2.5

To build `Features3D/obj_dinov2_top10_l3`, use:

```bash
cd /work/scratch/pafina/object-x

export Data_ROOT_DIR=<your_scene_root>
export SPLIT=val
export MAX_SCANS=0

bash scripts/features3D/obj_visual_embeddings_tmp.sh
```

Important note:

- the current script reads `Data_ROOT_DIR` with that exact capitalization, not `DATA_ROOT_DIR`
- if this gets fixed later, update this note

What it produces:

- `files/Features3D/obj_dinov2_top10_l3/<scan_id>.pkl`

## 5. How to run SLAT encode

Wrapper:

- `scripts/inference/run_pipeline_slat_tmp.sh`

What it produces:

- `files/gs_embeddings/<scan_id>_slat.npz`

Generic command:

```bash
cd /work/scratch/pafina/object-x

export DATA_ROOT_DIR=<your_scene_root>
export SPLIT=val
export SCENE_ID=<scan_id>
export RESET_TMP=1

bash scripts/inference/run_pipeline_slat_tmp.sh
```

## 6. How to run U3DGS encode + decode

Wrapper:

- `scripts/inference/run_pipeline_u3dgs_tmp.sh`

What it produces:

- `files/gs_embeddings/<scan_id>_ulat.npz`
- `vis/<scan_id>_joint.ply`
- `vis/rendered/<scan_id>_orbit_rendered.mp4`

Important new decode-side improvements:

- chunked decode by object:
  - `OBJECTX_INFER_DECODE_OBJECTS_PER_CHUNK`
- retry on different occupancy / max voxel settings:
  - `OBJECTX_INFER_OCC_THRESHOLD`
  - `OBJECTX_INFER_OCC_THRESHOLD_FALLBACKS`
  - `OBJECTX_INFER_MAX_VOXELS`
  - `OBJECTX_INFER_MAX_VOXELS_FALLBACKS`
- support-constrained cleanup:
  - `OBJECTX_VIS_SUPPORT_CONSTRAIN`
  - `OBJECTX_VIS_SUPPORT_DILATE_VOXELS`
  - `OBJECTX_VIS_SUPPORT_MAX_SCALE_VOXELS`
  - `OBJECTX_VIS_SUPPORT_MIN_KEEP`

Current stable cabinet-style debug command:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR=/work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-gtmask
export OBJECTX_MASK_SOURCE=gt_projection
export OBJECTX_SLAT_MAX_OBJECTS_PER_CHUNK=4
export SPLIT=val
export SCENE_ID=e61b0e04-bada-2f31-82d6-72831a602ba7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OBJECTX_VIS_SKIP_GS=1
export OBJECTX_VIS_ORBIT_MODE=scene
export OBJECTX_VIS_RENDER_SCALE=0.5
export OBJECTX_VIS_NUM_FRAMES=48

export OBJECTX_INFER_OCC_THRESHOLD=0.1
export OBJECTX_INFER_OCC_THRESHOLD_FALLBACKS=''
export OBJECTX_INFER_MAX_VOXELS=0
export OBJECTX_INFER_MAX_VOXELS_FALLBACKS=''
export OBJECTX_INFER_DECODE_OBJECTS_PER_CHUNK=1

export OBJECTX_VIS_SUPPORT_CONSTRAIN=1
export OBJECTX_VIS_SUPPORT_DILATE_VOXELS=0
export OBJECTX_VIS_SUPPORT_MAX_SCALE_VOXELS=1.0
export OBJECTX_VIS_SUPPORT_MIN_KEEP=4

rm -rf /tmp/$USER-objectx-infer
bash scripts/inference/run_pipeline_u3dgs_tmp.sh --visualize
```

More generic version:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR=<your_scene_root>
export OBJECTX_MASK_SOURCE=<gt_projection_or_pred_projection_clean>
export SPLIT=val
export SCENE_ID=<scan_id>
export RESET_TMP=1

export OBJECTX_INFER_OCC_THRESHOLD=0.1
export OBJECTX_INFER_OCC_THRESHOLD_FALLBACKS=''
export OBJECTX_INFER_MAX_VOXELS=0
export OBJECTX_INFER_MAX_VOXELS_FALLBACKS=''
export OBJECTX_INFER_DECODE_OBJECTS_PER_CHUNK=1

export OBJECTX_VIS_SUPPORT_CONSTRAIN=1
export OBJECTX_VIS_SUPPORT_DILATE_VOXELS=0
export OBJECTX_VIS_SUPPORT_MAX_SCALE_VOXELS=1.0
export OBJECTX_VIS_SUPPORT_MIN_KEEP=4

bash scripts/inference/run_pipeline_u3dgs_tmp.sh --visualize
```

## 7. How to generate MP4 + HTML for inspection

Best current wrapper:

- `scripts/segmentation/render_joint_depth_background_bundle.sh`

This wraps:

- `scripts/segmentation/render_joint_depth_background.py`

What it produces:

- `.../<scan_id>_joint_depth_bg_<label>.mp4`
- `.../<scan_id>_interactive_<label>.html`
- `contact_sheet.png`
- `summary.json`

Generic command:

```bash
cd /work/scratch/pafina/object-x

bash scripts/segmentation/render_joint_depth_background_bundle.sh \
  --scan-id <scan_id> \
  --replacement-root <your_scene_root> \
  --label <label>
```

Current cabinet example:

```bash
cd /work/scratch/pafina/object-x

bash scripts/segmentation/render_joint_depth_background_bundle.sh \
  --scan-id e61b0e04-bada-2f31-82d6-72831a602ba7 \
  --replacement-root /work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-gtmask \
  --label cabinet_fullscene_allobjects_depthbg_rgb_supportconstrained_v3
```

Typical output folder pattern:

```text
vis/rendered_joint_depth_bg/<scan_id>_<obj_slug>_<label>/
```

To serve the HTML locally from the exact output folder:

```bash
cd /work/scratch/pafina/object-x/vis/rendered_joint_depth_bg/<scan_folder>
python -m http.server 8000
```

## 8. Useful quick reminders

- If masks are noisy, first make `pred_projection_clean`.
- If object placement is wrong, use `worldspace_object_debug.py` before touching SLAT / U3DGS.
- If object recall drops after fusion, use `diagnose_object_gaps.py`.
- If U3DGS looks much worse than the pre-decode plot, inspect:
  - chunk decode
  - support-constrained cleanup
  - occupancy / voxel fallback settings

## 9. Current remaining GT dependencies

Even in the better current path, some GT-derived metadata still exists:

- `gt_projection` masks in some experiments
- `objects.json`
- `files/orig/...`
- `Features3D`

So the object geometry replacement is already much better than before, but the scene root is not yet fully GT-free.
