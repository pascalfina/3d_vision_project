# Geometry Evaluation

The geometry evaluator compares a reconstructed Pi3X point cloud against a GT
mesh before considering any object labels.

Recommended first run:

```bash
bash evaluation/geometry/run_oven_pi3x_geometry_eval.sh
```

Generic scene run:

```bash
SCAN_ID=<scene_id> \
METHOD_NAME=pi3x_samobject \
PRED_ROOT=<pi3x_pred_root> \
BASELINE_ROOT=<gt_root> \
bash evaluation/geometry/run_pi3x_geometry_eval.sh
```

For debug PLY outputs:

```bash
WRITE_DEBUG_PLY=1 bash evaluation/geometry/run_oven_pi3x_geometry_eval.sh
```

Every runner writes an interactive HTML overlay by default at
`<run_out_dir>/overlay_pred_gt.html`, so each metrics file has a matching visual
alignment check.  Set `WRITE_DEBUG_HTML=` to disable it for unusually large
batch jobs.  Use `WRITE_DEBUG_PLY=1` if you also want PLY debug files.

Each run writes:

- `metrics.json`: full machine-readable result.
- `metrics.csv`: one report row per GT scope.
- `report_summary.md`: compact report-ready table.
- `overlay_pred_gt.html`: interactive GT-vs-Pi3X overlay when requested.

Aggregate several completed scene runs:

```bash
python evaluation/geometry/summarize_geometry_runs.py \
  --metrics-glob 'evaluation/outputs/geometry/final/**/metrics.json' \
  --out-dir evaluation/outputs/geometry/final
```

This writes `geometry_summary.csv`, `geometry_summary.md` and
`geometry_summary.json`.

## Final Object-X Geometry

For the final Object-X output, do not evaluate the render images and do not stop
at voxelise.  The relevant decoded geometry is written by the `u3dgs` stage when
visualization is enabled:

- `files/gs_embeddings/<scene>_slat.npz`: structured latent, not final geometry.
- `files/gs_embeddings/<scene>_ulat.npz`: unstructured latent, not final geometry.
- `vis/<scene>_joint.ply`: final decoded U3DGS Gaussian scene geometry.

If you explicitly set `OBJECTX_VIS_EXPORT_MESH=1` for `u3dgs`, Object-X also
writes `vis/rendered/<scene>_mesh.ply`; pass that as `FINAL_PLY` if you want to
evaluate the optional splatted mesh instead of Gaussian centers.

Run the full final-geometry evaluation after `slat` and `u3dgs`:

```bash
bash scripts/workflows/run_scene_profile.sh <profile> objectx-final-geometry-eval
```

The action writes:

- `final_vs_gt/metrics.json`: final `*_joint.ply` aligned to GT via the same
  RGB-D correspondence alignment used for Pi3X-vs-GT.
- `final_vs_objectx_input/metrics.json`: final `*_joint.ply` compared in the
  Object-X coordinate frame against `files/gs_annotations/<scene>/<obj>`, i.e.
  the voxelized geometry that Object-X actually received.
- `final_vs_raw_pi3x/metrics.json`: final `*_joint.ply` compared in the
  Object-X/Pi3X coordinate frame against the raw fused Pi3X `frame-*.xyz.npy`
  sequence.
- `report_summary.md`: compact combined summary.
- HTML overlays for both comparisons when `WRITE_DEBUG_HTML=1`.

Standalone equivalent:

```bash
SCAN_ID=<scene_id> \
METHOD_NAME=<method_name> \
PRED_ROOT=<pi3x_reconstruction_root> \
PRED_READY_ROOT=<pred_ready_root_used_by_slat_u3dgs> \
BASELINE_ROOT=<gt_root> \
FINAL_PLY=vis/<scene_id>_joint.ply \
bash evaluation/geometry/run_objectx_final_geometry_eval.sh
```

Balanced 100-scene final benchmark from the existing 329-scene Pi3X geometry
run:

```bash
bash evaluation/geometry/run_objectx_final_benchmark_100.sh
```

The batch script selects 50 scenes from the best Pi3X geometry scores, 25 from
the middle, and 25 from the worst.  It runs the full pipeline per scene, keeps
only `evaluation/outputs/geometry/objectx_final_100/...`, and cleans the heavy
per-scene reconstruction/pred-ready/SAMObject/vis artifacts after each scene.
Use `DRY_RUN=1 LIMIT=1` to inspect the command sequence without running it.

MUSt3R-only geometry benchmark over the same under-300-frame scene list as the
Pi3X sequence benchmark:

```bash
bash evaluation/geometry/run_must3r_sequence_benchmark_under300.sh
```

For a non-interactive GPU job:

```bash
sbatch scripts/slurm/must3r_under300_geometry_benchmark.sbatch
tail -f "$(ls -t debug/slurm-must3r-geom329-*.out | head -n 1)"
```

This writes directly comparable outputs under
`evaluation/outputs/geometry/must3r_sequence_under300/`.  It runs only the
profile's `must3r` action, evaluates
`scenes_sam2_must3r/<scan>/sequence` with the same Geometry Eval backend used
for Pi3X, then removes the generated MUSt3R reconstruction artifacts unless
`KEEP_ARTIFACTS=1` is set.

SAM2 + MUSt3R final Object-X benchmark, using the same 100-scene selection as
`objectx_final_100` but replacing the Pi3X/SAMObject path with plain MUSt3R
geometry and SAM2 masks:

```bash
sbatch scripts/slurm/sam2_must3r_final_100_benchmark.sbatch
tail -f "$(ls -t debug/slurm-sam2-must3r-final100-*.out | head -n 1)"
```

This writes outputs under
`evaluation/outputs/geometry/objectx_final_100_sam2_must3r/`.  The runner
creates temporary generated profiles named `scene_<short>_sam2_must3r`, runs
`must3r -> segment-inputs -> voxelise -> build-pred-ready -> features3d ->
slat -> u3dgs -> objectx-final-geometry-eval`, and never runs the `samobject`
action.  It also verifies that reconstruction masks come from
`files/sam2_projection/obj_id_pkl/<scene>.pkl`, removes/overrides any
pred-ready `gt_projection`, and aliases pred-ready `gt_projection` to the SAM2
masks for Object-X compatibility.

After every scene, the batch script updates the global benchmark summaries:

- `objectx_final_100_summary.md`: report-friendly category aggregates plus a
  per-scene table.
- `objectx_final_100_summary.csv`: one row per selected scene with all measured
  final-vs-GT, final-vs-input and final-vs-raw-Pi3X metrics.
- `objectx_final_100_category_summary.csv`: global and bucket-level aggregates
  for each comparison category.
- `objectx_final_100_summary.json`: machine-readable version of both tables.

Main metrics:

- `pred_to_gt` / `accuracy`: low values mean reconstructed points lie close to
  the GT surface.
- `gt_to_pred` / `completeness`: low values mean GT surface points are covered
  by the reconstruction.
- `precision/recall/fscore@T`: percentage of points within distance threshold
  `T`, reported for thresholds such as 2 cm, 5 cm and 10 cm.

For paper-style tables, use the `visible_gt` scope when possible and report
`accuracy_mean_m`, `completeness_mean_m` and `fscore_at_0.050m`.  The full JSON
also contains medians and p95 values, which are useful for explaining outliers.

The oven convenience runner defaults to `PRED_INPUT_MODE=ply`.
That means it evaluates the self-computed Pi3X/SAMObject point cloud
`labels.instances.annotated.v2.ply` against the GT mesh.  The default alignment
is `ALIGN=rgbd_correspondence`: it estimates one global transform from paired
Pi3X world points and GT depth world points at matching frame/ray locations,
then applies that transform to the Pi3X PLY.  It does not replace the Pi3X
geometry with GT-lifted geometry.

Use this only if you explicitly want the diagnostic mode that lifts raw Pi3X
camera-frame `xyz.npy` with GT poses:

```bash
PRED_INPUT_MODE=sequence_gt_pose WRITE_DEBUG_HTML=1 bash evaluation/geometry/run_oven_pi3x_geometry_eval.sh
```

The script reports multiple GT scopes:

- `full_gt`: all sampled GT mesh surface points.
- `pred_bbox_gt`: GT points inside the reconstruction bounding box plus margin.
- `visible_gt`: GT points visible from the provided camera sequence, using a
  deterministic z-buffer over sampled GT points.  This is usually the fairest
  completeness metric for partial reconstructions.

Alignment:

- The default PLY evaluation uses `ALIGN=rgbd_correspondence`: robust trimmed
  Sim(3) from Pi3X-vs-GT RGB-D correspondences, applied to the Pi3X PLY.
- Use `ALIGN=coord_yz_flip` to inspect the simple explicit `x, -y, -z`
  coordinate conversion plus bbox centering.
- Use `ALIGN=coord_yz_flip_icp ICP_MAX_CORRESPONDENCE=0.35` for a local ICP
  refinement from that explicit conversion.
- Use `ALIGN=axis_bbox_icp` only as a diagnostic that searches all proper axis
  permutations/sign flips and refines the best candidates with ICP.
- `ALIGN=pose_rigid_icp` and `ALIGN=pose_similarity_icp` remain available as
  diagnostics for checking pose-based registration.
- `visible_gt` should use the GT camera sequence, not the Pi3X sequence,
  because GT mesh points are still in the GT coordinate frame.
- Use `--align none` if the global pose/world alignment itself should be
  penalized.
- `--align icp` and `--align pca_icp` remain available as diagnostics, but they
  are geometry-only guesses and should not be the default for Pi3X-vs-GT.
