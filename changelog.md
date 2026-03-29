# Changelog / Current Working Notes

Short version:

- objects can now come from our own mask + depth + pose path instead of directly from GT-style object geometry
- we now also have a pred-ready scene-root step that rebuilds downstream scene metadata from reconstructed objects instead of directly reusing the old GT scene graph
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
  - support-constrained cleanup that keeps decoded gaussians near the original sparse support
  - optional pruning / render controls for visualization cleanup
- we added a pred-ready scene-root builder:
  - `objects.json` can now be rebuilt from the actually reconstructed object set
  - `files/orig/data.pkl.gz` can now be rebuilt from reconstructed object points and centers
  - arrangement checks can now be compared directly against GT without rerunning the full pipeline

Important nuance:

- in the current best full-scene hybrid roots, the replaced object geometry under `files/gs_annotations/<scene>/<obj>/` is no longer just baseline GT object geometry
- however, some experiments still use GT masks (`gt_projection`) and the object inventory / IDs are still not yet coming from the final segmentation + tracking stage
- the current pred-ready root removes the old GT spatial scene-graph dependency for downstream loading, but it is still a compatibility step, not yet the final SAM / tracking solution

Main code touch points:

- `scripts/voxel_annotations/run_scanwise_voxelise_tmp.py`
  - new 2.5 path that can use `OBJECTX_MASK_SOURCE`
- `scripts/segmentation/worldspace_object_debug.py`
  - world-space object overlay debug
- `scripts/segmentation/diagnose_object_gaps.py`
  - object-level gap diagnosis
- `scripts/segmentation/build_pred_ready_scene_root.py`
  - builds a new scene root whose `objects.json` and `files/orig/data.pkl.gz` come from reconstructed objects
- `scripts/segmentation/validate_pred_ready_scene_root.py`
  - validates that the pred-ready root is internally consistent and dataset-loadable
- `scripts/segmentation/compare_scene_arrangement.py`
  - compares pred-ready object arrangement against GT arrangement
- `src/inference/unstructured_latent_inference.py`
  - chunked encode / decode
  - support-constrained cleanup
  - end-of-run gaussian cleanup summaries
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
- changed TSDF / hybrid handling so large planar objects such as the floor do not get dropped just because the TSDF branch is weak
  - if TSDF ends up empty or under the TSDF voxel threshold, the code now falls back to the already valid lifted voxel grid instead of losing the object entirely

Why this matters:

- this is where object geometry stopped being tied to the old GT-style object geometry path
- this file is the real heart of the new "2.5" reconstruction path
- if the reconstructed object looks wrong, the bug is very often in logic that now lives here
- the later floor fix for `cabinet` / `oven` also landed here

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
- added support extraction from the original sparse representation
- added support-constrained cleanup after decode so gaussians are filtered back toward the original support
- added helpers for masking, slicing sparse batches, empty-gaussian handling, and chunked object decode
- added a safer quantile path for gaussian pruning so very large gaussian sets do not crash on `torch.quantile(...)`
- added a final log summary that reports:
  - decode sparse coords
  - support-cleanup removal
  - prune removal when pruning is active
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
- `scripts/segmentation/build_pred_ready_scene_root.py`
- `scripts/segmentation/validate_pred_ready_scene_root.py`
- `scripts/segmentation/compare_scene_arrangement.py`

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

### `scripts/segmentation/build_pred_ready_scene_root.py`

Purpose:

- build a separate scene root for downstream Object-X steps
- keep the same loader-compatible structure, but generate:
  - `files/objects.json`
  - `files/orig/data.pkl.gz`
  from reconstructed objects instead of directly reusing the old GT scene graph

What it actually does:

- reads reconstructed objects from `files/gs_annotations/<scene>/<obj>/`
- converts voxel coordinates back into world-space points
- rebuilds:
  - object inventory
  - object point sets
  - root object
  - `rel_trans`
  - KNN edges / pairs / triples
- keeps some semantic fields from baseline only for compatibility

### `scripts/segmentation/validate_pred_ready_scene_root.py`

Purpose:

- sanity-check a generated pred-ready root before SLAT / U3DGS
- verifies:
  - object order consistency
  - required scene-graph keys
  - point-level shapes
  - dataset loader compatibility

### `scripts/segmentation/compare_scene_arrangement.py`

Purpose:

- compare the arrangement encoded in the pred-ready root against GT
- gives a fast check without rerunning the whole downstream pipeline
- reports:
  - center error
  - relative center error
  - pairwise distance error

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

Current cabinet full-scene floor-fix example:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR=/work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-gtmask
export SPLIT=val
export OBJECTX_MASK_SOURCE=gt_projection
export OBJECTX_VOXEL_OBJECT_SOURCE=hybrid_masks
export OBJECTX_VOXEL_TSDF_FALLBACK_TO_LIFTED=1
export RESET_TMP=1

bash scripts/voxel_annotations/voxelise_features_tmp.sh 2>&1 | tee /work/scratch/pafina/object-x/debug/debug_fullscene_cabinet_2_5_floorfix_rerun.log
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

## 5. Pred-ready scene root and arrangement validation

Why this exists:

- the old downstream Object-X path still reads scene-level metadata from:
  - `files/objects.json`
  - `files/orig/data.pkl.gz`
- we now have a compatibility step that rebuilds these from our reconstructed objects
- this means downstream loading can follow our new reconstruction path instead of still depending on the old GT spatial scene graph

Current cabinet pred-ready build:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/build_pred_ready_scene_root.py \
  --baseline-root /work/scratch/pafina/objectx-data-baseline \
  --reconstruction-root /work/scratch/pafina/objectx-data-fullscene-cabinet-hybrid-gtmask \
  --target-root /work/scratch/pafina/objectx-data-fullscene-cabinet-predready-v2-floorfix \
  --scene-id e61b0e04-bada-2f31-82d6-72831a602ba7 \
  --split val \
  --knn 4 \
  --min-voxels 32 \
  --point-counts 64 128 256 512 \
  --overwrite
```

Validate the pred-ready root:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/validate_pred_ready_scene_root.py \
  --root /work/scratch/pafina/objectx-data-fullscene-cabinet-predready-v2-floorfix \
  --scene-id e61b0e04-bada-2f31-82d6-72831a602ba7 \
  --split val \
  --out-dir /work/scratch/pafina/object-x/vis/pred_ready_validation/cabinet_predready_v2_floorfix
```

Compare arrangement against GT:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

python scripts/segmentation/compare_scene_arrangement.py \
  --pred-root /work/scratch/pafina/objectx-data-fullscene-cabinet-predready-v2-floorfix \
  --gt-root /work/scratch/pafina/objectx-data-baseline \
  --scene-id e61b0e04-bada-2f31-82d6-72831a602ba7 \
  --out-dir /work/scratch/pafina/object-x/vis/arrangement_compare/cabinet_predready_v2_floorfix_vs_gt
```

## 6. How to run SLAT encode

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

## 7. How to run U3DGS encode + decode

Wrapper:

- `scripts/inference/run_pipeline_u3dgs_tmp.sh`

What it produces:

- `files/gs_embeddings/<scan_id>_ulat.npz`
- `vis/<scan_id>_joint.ply`
- `vis/rendered/<scan_id>_orbit_rendered.mp4`

Important new decode-side improvements:

- chunked decode by object:
  - `OBJECTX_INFER_DECODE_OBJECTS_PER_CHUNK`
- support-constrained cleanup:
  - `OBJECTX_VIS_SUPPORT_CONSTRAIN`
  - `OBJECTX_VIS_SUPPORT_DILATE_VOXELS`
  - `OBJECTX_VIS_SUPPORT_MAX_SCALE_VOXELS`
  - `OBJECTX_VIS_SUPPORT_MIN_KEEP`
- optional gaussian prune:
  - `OBJECTX_VIS_PRUNE_OPACITY_*`
  - `OBJECTX_VIS_PRUNE_SCALE_*`
  - `OBJECTX_VIS_PRUNE_MAX_POINTS`

Important current recommendation:

- for our current best downstream checks, keep:
  - chunked decode
  - `max_voxels=0`
  - empty fallback lists
  - support-constrained cleanup
- do **not** enable the stronger quantile prune unless we explicitly want that experiment

Current stable cabinet-style support-only command on the new pred-ready root:

```bash
cd /work/scratch/pafina/object-x
source scripts/activate_objectx_env.sh

export DATA_ROOT_DIR=/work/scratch/pafina/objectx-data-fullscene-cabinet-predready-v2-floorfix
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

unset OBJECTX_VIS_PRUNE_OPACITY_MIN
unset OBJECTX_VIS_PRUNE_OPACITY_QUANTILE
unset OBJECTX_VIS_PRUNE_SCALE_MAX
unset OBJECTX_VIS_PRUNE_SCALE_QUANTILE
unset OBJECTX_VIS_PRUNE_MAX_POINTS
unset OBJECTX_VIS_PRUNE_QUANTILE_SAMPLE_MAX

rm -rf /tmp/$USER-objectx-infer
bash scripts/inference/run_pipeline_u3dgs_tmp.sh --visualize 2>&1 | tee /work/scratch/pafina/object-x/debug/debug_cabinet_predready_v2_floorfix_u3dgs_supportonly.log
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

What to look for in the log:

- decode:
  - `Decoding gaussians with occ_threshold=0.1 max_voxels=0 ...`
- support cleanup:
  - `Support-constrained gaussian cleanup ...`
- final summary:
  - `Final decode summary ...`
  - `Final support-cleanup summary ...`
  - `Final prune summary ...` only if prune is actually enabled

Current cabinet support-only numbers:

- sparse coords into decode: about `703,900`
- hard voxel cutoff: `none`
- support cleanup:
  - `22,524,800 -> 21,621,216`
  - removed about `903,584` gaussians
  - about `4.0%`

## 8. How to generate MP4 + HTML for inspection

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

Current cabinet example for the new pred-ready support-only run:

```bash
cd /work/scratch/pafina/object-x

bash scripts/segmentation/render_joint_depth_background_bundle.sh \
  --scan-id e61b0e04-bada-2f31-82d6-72831a602ba7 \
  --replacement-root /work/scratch/pafina/objectx-data-fullscene-cabinet-predready-v2-floorfix \
  --joint-ply /work/scratch/pafina/object-x/vis/e61b0e04-bada-2f31-82d6-72831a602ba7_joint.ply \
  --manifest /work/scratch/pafina/object-x/debug/cabinet_fullscene_all_objects_manifest.json \
  --label cabinet_predready_v2_floorfix_supportonly \
  --mask-source gt_projection \
  --background-remove-mode loaded \
  --pose-mode raw \
  --lift-coord-system pinhole \
  --frame-selection all \
  --max-views 96 \
  --bg-max-points 250000 \
  --joint-max-points 250000
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

## 9. Useful quick reminders

- If masks are noisy, first make `pred_projection_clean`.
- If object placement is wrong, use `worldspace_object_debug.py` before touching SLAT / U3DGS.
- If object recall drops after fusion, use `diagnose_object_gaps.py`.
- If U3DGS looks much worse than the pre-decode plot, inspect:
  - chunk decode
  - support-constrained cleanup
  - whether prune was accidentally enabled
- If the floor disappears in full-scene hybrid:
  - check whether `obj_id=1` actually got written in `files/gs_annotations/.../1/`
  - if TSDF is weak for planar objects, make sure the lifted fallback path is active

## 10. Current remaining GT dependencies

Even in the better current path, some GT-derived metadata still exists:

- `gt_projection` masks in some experiments
- object IDs / inventory are still not yet coming from the final segmentation + tracking stage
- `Features3D`

What is already improved:

- object geometry can come from our own mask + depth + pose path
- downstream scene metadata can now be rebuilt from reconstructed objects through the pred-ready root

What is still not final:

- the final object inventory / IDs should come from segmentation + tracking, not from the old dataset setup
- some experiments still intentionally use GT masks as an intermediate step
