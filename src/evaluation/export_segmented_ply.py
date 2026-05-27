#!/usr/bin/env python3
"""
Export a per-scene PAIR of coloured meshes for visual comparison:
  <scan>_gt.ply    -- GT mesh coloured by GT object instance
  <scan>_pred.ply  -- same mesh coloured by SAM2Object prediction

Colours are MATCHED across the two files: a predicted instance that is matched
(by greedy IoU) to a GT object is painted with that GT object's colour. So when
you open both meshes:
  - same colour in the same place  -> correct segmentation
  - a GT colour with no counterpart in pred -> a MISSED object (false negative)
  - black in the pred mesh                  -> a SPURIOUS instance (false positive)
  - grey                                    -> ignored (unannotated / structure)

Both meshes share the GT geometry (vertices + faces), so they overlay exactly.

Example:
  python src/evaluation/export_segmented_ply.py \
    --gt_root /cluster/project/cvg/data/3RScan/scenes \
    --scan 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
    --pred_npy '/cluster/scratch/ealegret/sam2object/scans/{scan}/results/{scan}_labels_fine_global.npy' \
    --out_dir /cluster/scratch/ealegret/sam2object/vis_compare \
    --mode objects-only --iou 0.25
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from plyfile import PlyData, PlyElement

DEFAULT_STRUCTURAL = {"wall", "floor", "ceiling"}
GREY = np.array([180, 180, 180], dtype=np.uint8)   # ignored / background
BLACK = np.array([15, 15, 15], dtype=np.uint8)      # false-positive prediction


def read_gt_mesh(gt_root, scan_id):
    ply = PlyData.read(osp.join(gt_root, scan_id, "labels.instances.annotated.v2.ply"))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    object_ids = np.asarray(v["objectId"]).astype(np.int64)
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int32)
    id2label = {}
    semseg = osp.join(gt_root, scan_id, "semseg.v2.json")
    if osp.exists(semseg):
        for g in json.load(open(semseg)).get("segGroups", []):
            id2label[int(g["objectId"])] = str(g.get("label", "unknown")).strip().lower()
    return xyz, faces, object_ids, id2label


def greedy_match(gt_ids, pred, gt_obj, ignore, iou_thr):
    """Return (match {pred_id: gt_id}, iou matrix, pred_ids, prop_ignore)."""
    fg = pred >= 0
    pred_ids = sorted(int(p) for p in np.unique(pred[fg]))
    pred_total = {p: int((pred == p).sum()) for p in pred_ids}
    pred_void = {p: int(((pred == p) & ignore).sum()) for p in pred_ids}
    prop_ignore = {p: (pred_void[p] / pred_total[p] if pred_total[p] else 1.0)
                   for p in pred_ids}
    pred_eff = {p: pred_total[p] - pred_void[p] for p in pred_ids}
    gt_size = {g: int((gt_obj == g).sum()) for g in gt_ids}

    n_p, n_g = len(pred_ids), len(gt_ids)
    iou = np.zeros((n_p, n_g))
    if n_p and n_g:
        up, ug = np.array(pred_ids), np.array(gt_ids)
        sel = fg & np.isin(pred, up) & np.isin(gt_obj, ug)
        inter = np.zeros((n_p, n_g), dtype=np.int64)
        np.add.at(inter, (np.searchsorted(up, pred[sel]),
                          np.searchsorted(ug, gt_obj[sel])), 1)
        ps = np.array([pred_eff[p] for p in pred_ids])[:, None]
        gs = np.array([gt_size[g] for g in gt_ids])[None, :]
        union = ps + gs - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    pairs = sorted(((iou[i, j], i, j) for i in range(n_p) for j in range(n_g)
                    if iou[i, j] >= iou_thr), reverse=True)
    used_p, used_g, match = set(), set(), {}
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        match[pred_ids[i]] = gt_ids[j]
    return match, iou, pred_ids, prop_ignore


def write_ply(path, xyz, faces, colors):
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    vd = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vd["x"], vd["y"], vd["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vd["red"], vd["green"], vd["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    fd = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    fd["vertex_indices"] = faces
    PlyData([PlyElement.describe(vd, "vertex"),
             PlyElement.describe(fd, "face")], text=False).write(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--pred_npy", required=True, help="use {scan} placeholder")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["objects-only", "all"], default="objects-only")
    ap.add_argument("--structural_labels", nargs="+",
                    default=sorted(DEFAULT_STRUCTURAL))
    ap.add_argument("--iou", type=float, default=0.25,
                    help="IoU threshold used to decide matched colour")
    args = ap.parse_args()

    xyz, faces, gt_obj, id2label = read_gt_mesh(args.gt_root, args.scan)
    pred = np.load(args.pred_npy.format(scan=args.scan)).astype(np.int64)
    if len(pred) != len(gt_obj):
        raise RuntimeError(f"pred {len(pred)} != GT vertices {len(gt_obj)}")

    structural = {s.lower() for s in args.structural_labels}
    ignore = gt_obj == 0
    if args.mode == "objects-only":
        struct_ids = [o for o, l in id2label.items() if l in structural]
        if struct_ids:
            ignore |= np.isin(gt_obj, struct_ids)

    gt_ids = sorted(int(g) for g in np.unique(gt_obj[~ignore]) if g > 0)
    match, iou, pred_ids, prop_ignore = greedy_match(
        gt_ids, pred, gt_obj, ignore, args.iou)

    # one stable colour per GT object
    rng = np.random.default_rng(0)
    palette = {g: rng.integers(40, 230, size=3, dtype=np.uint8) for g in gt_ids}

    # --- GT mesh: object colour, else grey -------------------------------- #
    gt_colors = np.tile(GREY, (len(xyz), 1))
    for g in gt_ids:
        gt_colors[gt_obj == g] = palette[g]

    # --- PRED mesh: matched->GT colour, FP->black, ignored->grey ---------- #
    pred_colors = np.tile(GREY, (len(xyz), 1))
    matched_gt = set(match.values())
    n_fp = 0
    for p in pred_ids:
        m = pred == p
        if p in match:
            pred_colors[m] = palette[match[p]]
        elif prop_ignore[p] <= args.iou:   # genuine spurious instance
            pred_colors[m] = BLACK
            n_fp += 1
        # else: void-dominated prediction -> leave grey

    gt_path = osp.join(args.out_dir, f"{args.scan}_gt.ply")
    pred_path = osp.join(args.out_dir, f"{args.scan}_pred.ply")
    write_ply(gt_path, xyz, faces, gt_colors)
    write_ply(pred_path, xyz, faces, pred_colors)

    # --- legend ----------------------------------------------------------- #
    missed = [g for g in gt_ids if g not in matched_gt]
    print(f"\nscan {args.scan}  mode={args.mode}  iou_thr={args.iou}")
    print(f"  GT objects={len(gt_ids)}  predictions={len(pred_ids)}  "
          f"matched={len(match)}  missed(FN)={len(missed)}  spurious(FP)={n_fp}")
    print("  MATCHED (pred -> GT object @ IoU):")
    for p, g in sorted(match.items(), key=lambda kv: kv[1]):
        j = gt_ids.index(g); i = pred_ids.index(p)
        rgb = palette[g]
        print(f"    pred {p:>4} -> GT {g:>3} '{id2label.get(g,'?'):<12}'  "
              f"IoU={iou[i, j]:.3f}  rgb=({rgb[0]},{rgb[1]},{rgb[2]})")
    if missed:
        print("  MISSED GT objects (appear in GT mesh, absent in pred):")
        for g in missed:
            best = iou[:, gt_ids.index(g)].max() if pred_ids else 0.0
            print(f"    GT {g:>3} '{id2label.get(g,'?'):<12}'  best IoU={best:.3f}")
    print(f"\n[OK] wrote {gt_path}")
    print(f"[OK] wrote {pred_path}")
    print("Open both in MeshLab/open3d: same colour+place = correct; "
          "black = false positive; a GT colour missing from pred = missed.")


if __name__ == "__main__":
    main()
