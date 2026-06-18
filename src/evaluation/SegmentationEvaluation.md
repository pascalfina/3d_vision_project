## Segmentation Evaluation

Evaluate the SAM2Object 3D class-agnostic instance segmentation against the
ground-truth meshes. The prediction `<scan>_labels_fine_global.npy` is 1:1
vertex-aligned with the GT mesh, so no registration is needed.

### Setup

```bash
source .venv/bin/activate   # has plyfile/numpy/scipy
cd 3d_vision_project
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

### Run the evaluation

**3RScan** — single scan:

```bash
python src/evaluation/evaluate_sam2object_3d.py \
  --dataset 3RScan \
  --gt_root ./data/3RScan/scenes \
  --scans 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
  --pred_npy './sam2object/scans/{scan}/results/{scan}_labels_fine_global.npy' \
  --mode both \
  --out ./sam2object/eval/report.json
```

A whole split — replace `--scans ...` with a scan-list file:

```bash
  --split_file ./data/3RScan/files/val_resplit_scans.txt
```

**ScanNet** — GT is reconstructed from the over-segmentation + aggregation:

```bash
python src/evaluation/evaluate_sam2object_3d.py \
  --dataset scannet \
  --gt_root ./data/scannet/scans \
  --split_file objectx_complete_scans_scannet.txt \
  --pred_npy './sam2object_scannet/scans/{scan}/results/{scan}_labels_fine_global.npy' \
  --mode both \
  --out ./sam2object_scannet/eval/report.json
```

### Options

- `--mode {objects-only,all,both}` — `objects-only` (headline) excludes the
  structural classes (wall/floor/ceiling); `all` scores every instance.
- `--iou_thresholds 0.25 0.5 0.75` — IoU thresholds for P/R/F1.
- `--out report.json` — full per-scan + aggregate report.
- `--append_txt ledger.txt --run_tag <tag>` — append one row per scan to a
  cumulative ledger.

The report gives Precision / Recall / F1 and mean matched IoU at each IoU
threshold (micro- and macro-averaged), plus threshold-free GT-coverage and
prediction-purity. Scans with a missing prediction or GT are skipped.

### Sanity check

Feeding the GT as the prediction must score 1.0 everywhere:

```bash
python src/evaluation/evaluate_sam2object_3d.py \
  --gt_root ./data/3RScan/scenes \
  --scans 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
  --pred gt --mode both
```

### Coloured point clouds (optional)

```bash
python src/evaluation/export_segmented_ply.py \
  --gt_root ./data/3RScan/scenes \
  --split_file <scan_list>.txt \
  --pred_npy './sam2object/scans/{scan}/results/{scan}_labels_fine_global.npy' \
  --out_dir ./sam2object/eval/plys \
  --mode objects-only --iou 0.25 --points_only
```
