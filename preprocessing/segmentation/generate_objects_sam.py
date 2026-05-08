"""
generate_objects_sam.py

Creates objects_sam.json from SAM2Object mask outputs.
Scans the mask_data_merge directory of each scene and creates one
object entry per unique pseudo-instance ID found across frames.

Output structure (minimal — only fields Object-X actually needs):
{
  "scans": [
    {
      "scan": "<scan_id>",
      "objects": [
        {"id": "1", "nyu40": "0"},
        {"id": "2", "nyu40": "0"},
        ...
      ]
    },
    ...
  ]
}

Run it as: 
python preprocessing/segmentation/generate_objects_sam.py \
    --sam2obj_dir /cluster/scratch/ealegret/3Rscan \
    --output /cluster/scratch/ealegret/3Rscan/objects_sam.json \
    --mask_group mask_data_merge
"""

import os
import os.path as osp
import json
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from argparse import ArgumentParser


def get_pseudo_instance_ids(mask_dir: str) -> list[int]:
    """
    Read all maskraw_*.png files in mask_dir and collect
    all unique non-zero instance IDs across all frames.
    """
    mask_files = natsorted(glob(osp.join(mask_dir, "maskraw_*.png")))
    all_ids = set()
    for mf in mask_files:
        mask = cv2.imread(mf, cv2.IMREAD_UNCHANGED)  # uint16
        if mask is None:
            continue
        ids = np.unique(mask)
        all_ids.update(int(i) for i in ids if i != 0)
    return sorted(all_ids)


def main(args):
    sam2obj_masks_root = osp.join(args.sam2obj_dir, "2D_masks")
    scan_ids = sorted(os.listdir(sam2obj_masks_root))

    # Optionally restrict to a split txt file
    if args.split_file:
        split_ids = set(np.genfromtxt(args.split_file, dtype=str).tolist())
        scan_ids = [s for s in scan_ids if s in split_ids]

    result = {"scans": []}

    for scan_id in tqdm(scan_ids, desc="Building objects_sam.json"):
        mask_dir = osp.join(sam2obj_masks_root, scan_id, args.mask_group)
        if not osp.isdir(mask_dir):
            print(f"  [WARN] No mask dir for {scan_id}, skipping")
            continue

        pseudo_ids = get_pseudo_instance_ids(mask_dir)
        if not pseudo_ids:
            print(f"  [WARN] No masks found for {scan_id}, skipping")
            continue

        objects = [
            {"id": str(pid), "nyu40": "0"}
            for pid in pseudo_ids
        ]
        result["scans"].append({"scan": scan_id, "objects": objects})

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(result['scans'])} scans → {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sam2obj_dir", required=True,
                        help="Root SAM2Object output dir (contains 2D_masks/)")
    parser.add_argument("--output", required=True,
                        help="Output path, e.g. <root_dir>/files/objects_sam.json")
    parser.add_argument("--mask_group", default="mask_data_merge",
                        help="Subdirectory name inside each scene's mask folder")
    parser.add_argument("--split_file", default=None,
                        help="Optional .txt file with scan IDs to restrict output")
    args = parser.parse_args()
    main(args)