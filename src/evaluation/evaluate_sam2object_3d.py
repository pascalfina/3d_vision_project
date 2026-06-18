#!/usr/bin/env python3
"""
3D class-agnostic instance-segmentation evaluation for SAM2Object vs 3RScan GT.

Why this is simple and fair:
  SAM2Object loads its points directly from the GT mesh
  (labels.instances.annotated.v2.ply), in vertex order. So the prediction
  array `<scan>_labels_fine_global.npy` is 1:1 vertex-aligned with the GT
  per-vertex objectId. No registration / nearest-neighbour transfer is needed.

What it computes (class-agnostic = masks only, no class labels):
  - IoU between every predicted instance and every GT instance over vertices,
    with void handling (pred vertices that fall on ignored GT regions do not
    count against the union -- the ScanNet convention).
  - Greedy IoU matching, then Precision / Recall / F1 at IoU 0.25 / 0.5 / 0.75,
    mean IoU over matched pairs, and threshold-free GT coverage / pred purity.

Modes:
  - objects-only (default): ignore unannotated (objectId==0) AND structural
    classes (wall / floor / ceiling). This is what Object-X cares about.
  - all: ignore only unannotated; score every GT instance.
  - both: print both reports.

Examples:
  # Sanity check: feed GT as the prediction -> everything must be 1.0
  python src/evaluation/evaluate_sam2object_3d.py \
    --gt_root /cluster/project/cvg/data/3RScan/scenes \
    --scans 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
    --pred gt

  # Real prediction
  python src/evaluation/evaluate_sam2object_3d.py \
    --gt_root /cluster/project/cvg/data/3RScan/scenes \
    --scans 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
    --pred_npy '/cluster/scratch/ealegret/sam2object/scans/{scan}/results/{scan}_labels_fine_global.npy' \
    --mode both --out /cluster/scratch/ealegret/sam2object/eval/report.json
"""

import argparse
import csv
import json
import os
import os.path as osp

import numpy as np
from plyfile import PlyData

DEFAULT_STRUCTURAL = {"wall", "floor", "ceiling"}
DEFAULT_IOU_THRESHOLDS = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_gt(dataset: str, gt_root: str, scan_id: str):
    """Dispatch to the per-dataset GT loader."""
    if dataset.lower() == "scannet":
        return load_gt_scannet(gt_root, scan_id)
    return load_gt_3rscan(gt_root, scan_id)


def load_gt_3rscan(gt_root: str, scan_id: str):
    """Return (objectId per vertex [N], {objectId: label_str}). objectId 0 = unannotated."""
    ply_path = osp.join(gt_root, scan_id, "labels.instances.annotated.v2.ply")
    if not osp.exists(ply_path):
        raise FileNotFoundError(f"GT ply not found: {ply_path}")
    vertex = PlyData.read(ply_path)["vertex"]
    object_ids = np.asarray(vertex["objectId"]).astype(np.int64)

    id2label = {}
    semseg_path = osp.join(gt_root, scan_id, "semseg.v2.json")
    if osp.exists(semseg_path):
        with open(semseg_path) as f:
            semseg = json.load(f)
        for g in semseg.get("segGroups", []):
            id2label[int(g["objectId"])] = str(g.get("label", "unknown")).strip().lower()
    else:
        print(f"[WARN] no semseg.v2.json for {scan_id}; "
              f"objects-only mode cannot drop structural classes.")
    return object_ids, id2label


def load_gt_scannet(gt_root: str, scan_id: str):
    """
    Reconstruct per-vertex instance ids for ScanNet from the oversegmentation +
    aggregation, mirroring utils/scannet.py (read_segmentation / read_obj_info_nyu40):
      - <scene>_vh_clean_2.0.010000.segs.json -> 'segIndices' (per-vertex segment id)
      - <scene>.aggregation.json (or <scene>_vh_clean.aggregation.json) -> 'segGroups'
    ScanNet objectId is 0-based, so we store objectId+1 and leave unassigned vertices
    at 0 (preserving the '0 = unannotated/ignore' convention used for 3RScan).
    """
    segs_path = osp.join(gt_root, scan_id, f"{scan_id}_vh_clean_2.0.010000.segs.json")
    if not osp.exists(segs_path):
        raise FileNotFoundError(f"ScanNet segs json not found: {segs_path}")
    agg_path = None
    for cand in (f"{scan_id}.aggregation.json", f"{scan_id}_vh_clean.aggregation.json"):
        if osp.exists(osp.join(gt_root, scan_id, cand)):
            agg_path = osp.join(gt_root, scan_id, cand)
            break
    if agg_path is None:
        raise FileNotFoundError(f"ScanNet aggregation json not found for {scan_id}")

    with open(segs_path) as f:
        seg_indices = np.asarray(json.load(f)["segIndices"]).astype(np.int64)
    with open(agg_path) as f:
        seg_groups = json.load(f).get("segGroups", [])

    object_ids = np.zeros(len(seg_indices), dtype=np.int64)
    id2label = {}
    for g in seg_groups:
        oid = int(g["objectId"]) + 1  # 0 reserved for unannotated
        id2label[oid] = str(g.get("label", "unknown")).strip().lower()
        object_ids[np.isin(seg_indices, np.asarray(g["segments"], dtype=np.int64))] = oid
    return object_ids, id2label


def load_pred(args, scan_id: str, n_vertices: int, gt_object_ids: np.ndarray):
    """Return predicted instance id per vertex [N] (background = value < 0)."""
    if args.pred == "gt":
        # Self-test: GT-as-prediction must score 1.0 everywhere.
        pred = gt_object_ids.copy()
        pred[pred == 0] = -1  # treat unannotated as background
        return pred

    if args.pred_npy:
        path = args.pred_npy.format(scan=scan_id)
        if not osp.exists(path):
            raise FileNotFoundError(f"pred npy not found: {path}")
        pred = np.load(path).astype(np.int64)
    elif args.pred_scannet_dir:
        d = args.pred_scannet_dir.format(scan=scan_id)
        pred = load_pred_scannet_format(d, scan_id, n_vertices)
    else:
        raise ValueError("Provide one of: --pred gt | --pred_npy | --pred_scannet_dir")

    if len(pred) != n_vertices:
        raise RuntimeError(
            f"[{scan_id}] pred has {len(pred)} labels but GT has {n_vertices} "
            f"vertices. Prediction is NOT vertex-aligned with GT -- check that "
            f"SAM2Object ran on labels.instances.annotated.v2.ply for this scan."
        )
    return pred


def load_pred_scannet_format(res_dir: str, scan_id: str, n_vertices: int):
    """Read SAM2Object's ScanNet export (<scan>.txt + per-mask .txt files)."""
    main_txt = osp.join(res_dir, f"{scan_id}.txt")
    if not osp.exists(main_txt):
        raise FileNotFoundError(f"ScanNet-format main file not found: {main_txt}")
    pred = np.full(n_vertices, -1, dtype=np.int64)
    with open(main_txt) as f:
        for inst_id, line in enumerate(f.read().splitlines()):
            if not line.strip():
                continue
            rel = line.split()[0]
            mask = np.loadtxt(osp.join(res_dir, rel)).astype(np.int64)
            pred[mask > 0] = inst_id
    return pred


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def evaluate_scan(
    gt_object_ids: np.ndarray,
    id2label: dict,
    pred: np.ndarray,
    objects_only: bool,
    structural_labels: set,
    iou_thresholds,
    min_gt_points: int,
    min_pred_points: int,
):
    # --- ignore mask: unannotated, plus structural in objects-only mode ----- #
    ignore = (gt_object_ids == 0)
    if objects_only:
        structural_ids = {
            oid for oid, lab in id2label.items() if lab in structural_labels
        }
        if structural_ids:
            ignore |= np.isin(gt_object_ids, list(structural_ids))

    # --- GT instances to evaluate ------------------------------------------ #
    gt_ids_all = np.unique(gt_object_ids[~ignore])
    gt_ids = sorted(int(g) for g in gt_ids_all if g > 0)
    gt_size = {g: int((gt_object_ids == g).sum()) for g in gt_ids}
    gt_ids = [g for g in gt_ids if gt_size[g] >= min_gt_points]

    # --- predicted instances (background = value < 0) ---------------------- #
    fg_pred = pred >= 0
    pred_ids_all = np.unique(pred[fg_pred])
    pred_total = {int(p): int((pred == p).sum()) for p in pred_ids_all}
    # how much of each prediction falls on ignored (void) GT regions
    pred_void = {int(p): int(((pred == p) & ignore).sum()) for p in pred_ids_all}
    # effective size = predicted vertices that are NOT in ignored GT regions
    pred_size_eff = {int(p): pred_total[int(p)] - pred_void[int(p)]
                     for p in pred_ids_all}
    # fraction of a prediction that lies in void; a void-dominated prediction
    # (e.g. a wall, in objects-only mode) must not be punished as a false positive
    prop_ignore = {int(p): (pred_void[int(p)] / pred_total[int(p)]
                            if pred_total[int(p)] else 1.0)
                   for p in pred_ids_all}
    pred_ids = sorted(int(p) for p in pred_ids_all
                      if pred_size_eff[int(p)] >= min_pred_points)

    n_gt, n_pred = len(gt_ids), len(pred_ids)

    # --- IoU matrix (n_pred x n_gt) with void handling --------------------- #
    iou = np.zeros((n_pred, n_gt), dtype=np.float64)
    if n_gt and n_pred:
        upred = np.array(pred_ids)
        ugt = np.array(gt_ids)
        sel = fg_pred & np.isin(pred, upred) & np.isin(gt_object_ids, ugt)
        pi = np.searchsorted(upred, pred[sel])
        gj = np.searchsorted(ugt, gt_object_ids[sel])
        inter = np.zeros((n_pred, n_gt), dtype=np.int64)
        np.add.at(inter, (pi, gj), 1)

        ps = np.array([pred_size_eff[p] for p in pred_ids])[:, None]
        gs = np.array([gt_size[g] for g in gt_ids])[None, :]
        union = ps + gs - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # --- threshold-free diagnostics ---------------------------------------- #
    # GT coverage: for each GT object, its best IoU with any prediction.
    gt_coverage = float(iou.max(axis=0).mean()) if n_gt and n_pred else 0.0
    # Pred purity: best IoU with any GT object, over predictions that are NOT
    # void-dominated (a wall prediction is irrelevant to objects-only purity).
    if n_gt and n_pred:
        keep = np.array([prop_ignore[p] <= 0.5 for p in pred_ids])
        pred_purity = float(iou[keep].max(axis=1).mean()) if keep.any() else 0.0
    else:
        pred_purity = 0.0

    # --- greedy matching per IoU threshold --------------------------------- #
    if n_gt and n_pred:
        pairs = [(iou[i, j], i, j)
                 for i in range(n_pred) for j in range(n_gt) if iou[i, j] > 0]
        pairs.sort(reverse=True)
    else:
        pairs = []

    per_threshold = {}
    for tau in iou_thresholds:
        used_p, used_g, matched_ious = set(), set(), []
        for v, i, j in pairs:
            if v < tau:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            matched_ious.append(v)
        tp = len(matched_ious)
        # An unmatched prediction is a false positive only if it is not
        # void-dominated (ScanNet convention: proportion in void <= threshold).
        fp = sum(1 for idx, pid in enumerate(pred_ids)
                 if idx not in used_p and prop_ignore[pid] <= tau)
        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_threshold[tau] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        }

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "gt_coverage": gt_coverage,
        "pred_purity": pred_purity,
        "per_threshold": per_threshold,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def aggregate(scan_results: dict, iou_thresholds):
    """Micro-average (pool TP/FP/FN across scans) + macro mean of per-scan F1."""
    agg = {"per_threshold": {}}
    for tau in iou_thresholds:
        tp = sum(r["per_threshold"][tau]["tp"] for r in scan_results.values())
        fp = sum(r["per_threshold"][tau]["fp"] for r in scan_results.values())
        fn = sum(r["per_threshold"][tau]["fn"] for r in scan_results.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        macro_f1 = float(np.mean(
            [s["per_threshold"][tau]["f1"] for s in scan_results.values()]
        )) if scan_results else 0.0
        agg["per_threshold"][tau] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1, "macro_f1": macro_f1,
        }
    agg["mean_gt_coverage"] = float(np.mean(
        [s["gt_coverage"] for s in scan_results.values()])) if scan_results else 0.0
    agg["mean_pred_purity"] = float(np.mean(
        [s["pred_purity"] for s in scan_results.values()])) if scan_results else 0.0
    return agg


def print_report(title, scan_results, agg, iou_thresholds):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    for scan_id, r in scan_results.items():
        print(f"\n[{scan_id}]  GT={r['n_gt']}  pred={r['n_pred']}  "
              f"GT-coverage={r['gt_coverage']:.3f}  pred-purity={r['pred_purity']:.3f}")
        print(f"  {'IoU':>5} {'P':>7} {'R':>7} {'F1':>7} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'mIoU':>6}")
        for tau in iou_thresholds:
            t = r["per_threshold"][tau]
            print(f"  {tau:>5.2f} {t['precision']:>7.3f} {t['recall']:>7.3f} "
                  f"{t['f1']:>7.3f} {t['tp']:>4} {t['fp']:>4} {t['fn']:>4} "
                  f"{t['mean_matched_iou']:>6.3f}")

    print(f"\n--- AGGREGATE ({len(scan_results)} scan(s)) ---")
    print(f"  mean GT-coverage={agg['mean_gt_coverage']:.3f}  "
          f"mean pred-purity={agg['mean_pred_purity']:.3f}")
    print(f"  {'IoU':>5} {'P(micro)':>9} {'R(micro)':>9} "
          f"{'F1(micro)':>10} {'F1(macro)':>10}")
    for tau in iou_thresholds:
        t = agg["per_threshold"][tau]
        print(f"  {tau:>5.2f} {t['precision']:>9.3f} {t['recall']:>9.3f} "
              f"{t['f1']:>10.3f} {t['macro_f1']:>10.3f}")


def append_ledger(path, run_tag, dataset, results, modes, thresholds):
    """Append one tab-separated row per (scan, mode) to a never-overwritten .txt
    ledger; write the header only when the file is new/empty. Pure append, so the
    file accumulates across runs and re-runs add a new (run_tag-stamped) row."""
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    need_header = (not osp.exists(path)) or osp.getsize(path) == 0
    cols = ["run_tag", "dataset", "scan", "mode",
            "n_gt", "n_pred", "gt_coverage", "pred_purity"]
    for tau in thresholds:
        cols += [f"{k}@{tau:g}" for k in ("p", "r", "f1", "miou", "tp", "fp", "fn")]
    n_rows = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if need_header:
            w.writerow(cols)
        for m in modes:
            for scan_id, r in results[m].items():
                row = [run_tag, dataset, scan_id, m, r["n_gt"], r["n_pred"],
                       f"{r['gt_coverage']:.6f}", f"{r['pred_purity']:.6f}"]
                for tau in thresholds:
                    t = r["per_threshold"][tau]
                    row += [f"{t['precision']:.6f}", f"{t['recall']:.6f}",
                            f"{t['f1']:.6f}", f"{t['mean_matched_iou']:.6f}",
                            t["tp"], t["fp"], t["fn"]]
                w.writerow(row)
                n_rows += 1
    return n_rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", required=True,
                    help="scenes dir with per-scan GT: 3RScan /cluster/project/cvg/data/3RScan/scenes "
                         "or ScanNet /cluster/project/cvg/data/scannet/scans")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scans", nargs="+", help="scan ids")
    g.add_argument("--split_file", help="text file with one scan id per line")

    p = ap.add_mutually_exclusive_group(required=True)
    p.add_argument("--pred", choices=["gt"],
                   help="'gt' = use GT as prediction (self-test, must score 1.0)")
    p.add_argument("--pred_npy",
                   help="path to <scan>_labels_fine_global.npy; use {scan} placeholder")
    p.add_argument("--pred_scannet_dir",
                   help="dir with SAM2Object ScanNet export; use {scan} placeholder")

    ap.add_argument("--dataset", default="3RScan", choices=["3RScan", "scannet", "ScanNet", "3rscan"],
                    help="GT format/layout to load (case-insensitive)")
    ap.add_argument("--mode", choices=["objects-only", "all", "both"],
                    default="objects-only")
    ap.add_argument("--structural_labels", nargs="+", default=sorted(DEFAULT_STRUCTURAL),
                    help="labels treated as structure (objects-only mode)")
    ap.add_argument("--iou_thresholds", nargs="+", type=float,
                    default=list(DEFAULT_IOU_THRESHOLDS))
    ap.add_argument("--min_gt_points", type=int, default=0)
    ap.add_argument("--min_pred_points", type=int, default=0)
    ap.add_argument("--out", help="optional path to dump full JSON report")
    ap.add_argument("--append_txt",
                    help="append one tab-separated row per (scan,mode) to this .txt "
                         "ledger (header written if new; never overwritten)")
    ap.add_argument("--run_tag", default="",
                    help="tag written into each appended ledger row, e.g. <timestamp>_<jobid>")
    args = ap.parse_args()

    if args.split_file:
        with open(args.split_file) as f:
            scans = [l.strip() for l in f if l.strip()]
    else:
        scans = args.scans

    structural = {s.strip().lower() for s in args.structural_labels}
    thresholds = list(args.iou_thresholds)
    modes = ["all", "objects-only"] if args.mode == "both" else [args.mode]

    results = {m: {} for m in modes}
    skipped = []
    for scan_id in scans:
        try:
            gt_ids, id2label = load_gt(args.dataset, args.gt_root, scan_id)
            pred = load_pred(args, scan_id, len(gt_ids), gt_ids)
        except (FileNotFoundError, RuntimeError) as e:
            # Missing prediction/GT, or vertex-count mismatch: skip, keep going.
            print(f"[SKIP] {scan_id}: {e}")
            skipped.append(scan_id)
            continue
        n_fg = int((pred >= 0).sum())
        print(f"[{scan_id}] vertices={len(gt_ids)}  "
              f"pred fg(>=0)={n_fg}  pred bg(<0)={len(pred) - n_fg}  "
              f"pred instances={len(np.unique(pred[pred >= 0]))}")
        for m in modes:
            results[m][scan_id] = evaluate_scan(
                gt_object_ids=gt_ids, id2label=id2label, pred=pred,
                objects_only=(m == "objects-only"),
                structural_labels=structural, iou_thresholds=thresholds,
                min_gt_points=args.min_gt_points,
                min_pred_points=args.min_pred_points,
            )

    n_eval = len(scans) - len(skipped)
    print(f"\n[SUMMARY] evaluated {n_eval}, skipped {len(skipped)} "
          f"(of {len(scans)} listed)")
    if n_eval == 0:
        print("[WARN] no scans evaluated; nothing to report.")
        return

    full = {}
    for m in modes:
        agg = aggregate(results[m], thresholds)
        title = ("OBJECTS-ONLY (excl. " + ", ".join(sorted(structural)) + ")"
                 if m == "objects-only" else "ALL INSTANCES (incl. structure)")
        print_report(title, results[m], agg, thresholds)
        full[m] = {"per_scan": results[m], "aggregate": agg}

    if args.out:
        os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)

        def keys_to_str(o):  # JSON object keys must be strings
            if isinstance(o, dict):
                return {str(k): keys_to_str(v) for k, v in o.items()}
            return o

        with open(args.out, "w") as f:
            json.dump(keys_to_str(full), f, indent=2)
        print(f"\n[OK] wrote report: {args.out}")

    if args.append_txt:
        n_rows = append_ledger(args.append_txt, args.run_tag, args.dataset,
                               results, modes, thresholds)
        print(f"[OK] appended {n_rows} row(s) to ledger: {args.append_txt}")


if __name__ == "__main__":
    main()
