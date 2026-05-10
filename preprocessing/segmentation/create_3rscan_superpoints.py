#!/usr/bin/env python3
"""
Convert 3RScan mesh.refined.0.010000.segs.v2.json to superpoint.pts
format expected by SAM2Object graphclustering.

Usage:
    python create_3rscan_superpoints.py \
        --segs_json  <path>/mesh.refined.0.010000.segs.v2.json \
        --out_dir    <samobject_root>/superpoints/<scan_id>/
"""
import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_json", required=True, help="Path to mesh.refined.0.010000.segs.v2.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for superpoint.pts")
    args = parser.parse_args()

    with open(args.segs_json) as f:
        data = json.load(f)

    seg_indices = np.array(data["segIndices"], dtype=np.int64)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "superpoint.pts")
    np.savetxt(out_path, seg_indices, fmt="%d")
    print(f"[OK] Wrote {len(seg_indices)} superpoint assignments to {out_path}")


if __name__ == "__main__":
    main()
