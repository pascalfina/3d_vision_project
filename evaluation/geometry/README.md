# Geometry Evaluation

The geometry evaluator compares a reconstructed Pi3X point cloud against a GT
mesh before considering any object labels.

Recommended first run:

```bash
bash evaluation/geometry/run_oven_pi3x_geometry_eval.sh
```

For debug PLY outputs:

```bash
WRITE_DEBUG_PLY=1 bash evaluation/geometry/run_oven_pi3x_geometry_eval.sh
```

This also writes an interactive HTML overlay at
`evaluation/outputs/geometry/oven_pi3x/overlay_pred_gt.html`.
Use `WRITE_DEBUG_HTML=1` if you only want the HTML and no PLY files.

Main metrics:

- `pred_to_gt`: accuracy.  Low values mean reconstructed points lie close to
  the GT surface.
- `gt_to_pred`: completeness.  Low values mean GT surface points are covered by
  the reconstruction.
- `precision/recall/fscore@T`: percentage of points within distance threshold
  `T`, reported for thresholds such as 2 cm, 5 cm and 10 cm.

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
