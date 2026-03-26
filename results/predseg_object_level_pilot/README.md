# PredSeg Object-Level Pilot

This folder contains a small curated subset of visualization artifacts for the
object-level predicted-segmentation reconstruction pilot.

Contents:

- `videos/`
  - Bright object-centric renders for 5 pilot objects from 5 different scenes.
- `figures/`
  - `all_compare_frame36.png`: quick legacy vs object-centric comparison sheet.
- `diagnostics/`
  - `aggregate_metrics.png`: aggregate diagnostic plot across `k = 1,2,4,8,12`.
  - `diagnosis_summary.json`: compact per-object diagnosis summary.
  - `summary.csv`: full diagnostic table.

Interpretation:

- The object-level pilot is clearly better than the earlier scene-level
  predicted-segmentation attempt.
- However, the main bottleneck is still the quality of the 3D object geometry
  reconstructed from `mask + depth + pose`, not the renderer itself.
- The diagnostics suggest that the problem is often already present before
  fusion, not only caused by the final merging step.
