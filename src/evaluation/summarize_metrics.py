#!/usr/bin/env python3
"""
Summarize the append-only per-scan metrics ledger written by
evaluate_sam2object_3d.py (--append_txt). Produces two .txt artifacts:
  - best_metrics.txt    : the top-N scenes ranked by a metric (default
                          objects-only f1 @ IoU 0.5), with their numbers.
  - general_metrics.txt : the average over ALL accumulated scenes (micro
                          P/R/F1, macro-F1, mean coverage/purity per IoU) -- the
                          single general metric to report.

The ledger may hold multiple rows per scan (one per run); the LATEST row per
(scan, mode) is used (file order = chronological, last wins). Reuses
aggregate() from evaluate_sam2object_3d.py.

Example:
  python src/evaluation/summarize_metrics.py \
    --ledger results/3RScan/metrics_per_scan.txt --mode both --top 10 \
    --out_best results/3RScan/best_metrics.txt \
    --out_avg  results/3RScan/general_metrics.txt
"""
import argparse
import csv
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from evaluate_sam2object_3d import aggregate  # noqa: E402


def parse_ledger(path):
    """Return (thresholds_sorted, by_mode) where by_mode[mode][scan] is a
    reconstructed per-scan result dict (latest row per scan,mode wins)."""
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        taus = sorted({float(c.split("@", 1)[1]) for c in cols if c.startswith("f1@")})
        by_mode = {}
        for row in reader:
            pt = {}
            for tau in taus:
                g = f"{tau:g}"
                pt[tau] = {
                    "precision": float(row[f"p@{g}"]),
                    "recall": float(row[f"r@{g}"]),
                    "f1": float(row[f"f1@{g}"]),
                    "mean_matched_iou": float(row[f"miou@{g}"]),
                    "tp": int(row[f"tp@{g}"]),
                    "fp": int(row[f"fp@{g}"]),
                    "fn": int(row[f"fn@{g}"]),
                }
            rec = {
                "n_gt": int(row["n_gt"]),
                "n_pred": int(row["n_pred"]),
                "gt_coverage": float(row["gt_coverage"]),
                "pred_purity": float(row["pred_purity"]),
                "per_threshold": pt,
            }
            by_mode.setdefault(row["mode"], {})[row["scan"]] = rec  # latest wins
    return taus, by_mode


def write_best(by_mode, taus, rank_mode, metric, iou, top, out_path):
    if rank_mode not in by_mode:
        rank_mode = next(iter(by_mode))
    tau = float(iou)
    if tau not in taus:
        print(f"[WARN] IoU {iou} not in ledger thresholds {taus}; best ranking degenerate.")
    scans = by_mode[rank_mode]

    def key(scan):
        pt = scans[scan]["per_threshold"].get(tau, {})
        return (pt.get(metric, 0.0), pt.get("mean_matched_iou", 0.0),
                scans[scan]["gt_coverage"])

    ranked = sorted(scans, key=key, reverse=True)[:top]
    lines = [f"# best {len(ranked)} scenes by {rank_mode} {metric}@{iou} "
             f"(of {len(scans)} scenes in ledger)",
             "\t".join(["rank", "scan", f"{metric}@{iou}", f"precision@{iou}",
                        f"recall@{iou}", f"miou@{iou}", "n_gt", "n_pred",
                        "gt_coverage", "pred_purity"])]
    for i, scan in enumerate(ranked, 1):
        s = scans[scan]
        pt = s["per_threshold"].get(tau, {})
        lines.append("\t".join([
            str(i), scan, f"{pt.get(metric, 0):.4f}", f"{pt.get('precision', 0):.4f}",
            f"{pt.get('recall', 0):.4f}", f"{pt.get('mean_matched_iou', 0):.4f}",
            str(s["n_gt"]), str(s["n_pred"]),
            f"{s['gt_coverage']:.4f}", f"{s['pred_purity']:.4f}"]))
    text = "\n".join(lines) + "\n"
    print(text)
    if out_path:
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
        print(f"[OK] wrote {out_path}")


def fmt_general(by_mode, taus, modes):
    blocks = []
    for mode in modes:
        if mode not in by_mode:
            continue
        scan_results = by_mode[mode]
        agg = aggregate(scan_results, taus)
        b = [f"=== GENERAL METRICS  |  {mode}  |  {len(scan_results)} scenes ===",
             f"mean GT-coverage={agg['mean_gt_coverage']:.4f}  "
             f"mean pred-purity={agg['mean_pred_purity']:.4f}",
             "\t".join(["IoU", "P_micro", "R_micro", "F1_micro", "F1_macro"])]
        for tau in taus:
            t = agg["per_threshold"][tau]
            b.append("\t".join([f"{tau:g}", f"{t['precision']:.4f}",
                                 f"{t['recall']:.4f}", f"{t['f1']:.4f}",
                                 f"{t['macro_f1']:.4f}"]))
        blocks.append("\n".join(b))
    return "\n\n".join(blocks) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True, help="the append-only metrics .txt")
    ap.add_argument("--mode", default="both", choices=["objects-only", "all", "both"],
                    help="which mode(s) to average for the general metric")
    ap.add_argument("--metric", default="f1",
                    choices=["f1", "mean_matched_iou", "precision", "recall"])
    ap.add_argument("--iou", default="0.5")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out_best")
    ap.add_argument("--out_avg")
    args = ap.parse_args()

    if not osp.exists(args.ledger) or osp.getsize(args.ledger) == 0:
        print(f"[WARN] ledger empty or missing: {args.ledger}")
        return
    taus, by_mode = parse_ledger(args.ledger)
    if not taus or not by_mode:
        print("[WARN] no usable rows/thresholds in ledger.")
        return

    rank_mode = "objects-only" if "objects-only" in by_mode else next(iter(by_mode))
    write_best(by_mode, taus, rank_mode, args.metric, args.iou, args.top, args.out_best)

    avg_modes = ["all", "objects-only"] if args.mode == "both" else [args.mode]
    text = fmt_general(by_mode, taus, avg_modes)
    print(text)
    if args.out_avg:
        os.makedirs(osp.dirname(args.out_avg) or ".", exist_ok=True)
        with open(args.out_avg, "w") as f:
            f.write(text)
        print(f"[OK] wrote {args.out_avg}")


if __name__ == "__main__":
    main()
