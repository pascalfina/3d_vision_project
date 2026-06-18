#!/usr/bin/env python3
"""
Print the top-N scenes from an eval report.json, ranked by a metric.

Output: one line per winner, "<scan_id>\\t<score>" (so it doubles as a manifest;
the runner takes column 1 as the id list). Ranking is by `metric` at IoU `iou`,
tie-broken by mean matched IoU then GT-coverage.

Example:
  python src/evaluation/select_top_scenes.py \
    --report /cluster/scratch/ealegret/sam2object/eval_batch/report.json \
    --mode objects-only --metric f1 --iou 0.5 --top 5
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--mode", default="objects-only", choices=["objects-only", "all"])
    ap.add_argument("--metric", default="f1",
                    choices=["f1", "mean_matched_iou", "precision", "recall"])
    ap.add_argument("--iou", default="0.5")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    per_scan = report.get(args.mode, {}).get("per_scan", {})
    iou_key = str(args.iou)

    def rank_key(scan):
        s = per_scan[scan]
        pt = s.get("per_threshold", {}).get(iou_key, {})
        return (pt.get(args.metric, 0.0),
                pt.get("mean_matched_iou", 0.0),
                s.get("gt_coverage", 0.0))

    ranked = sorted(per_scan.keys(), key=rank_key, reverse=True)
    for scan in ranked[:args.top]:
        score = per_scan[scan].get("per_threshold", {}).get(iou_key, {}).get(args.metric, 0.0)
        print(f"{scan}\t{score:.4f}")


if __name__ == "__main__":
    main()
