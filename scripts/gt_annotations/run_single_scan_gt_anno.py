#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlsG-space", dest="vlsG_space", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.vlsG_space)

    from configs import config, update_config
    from preprocessing.gt_anno_2D.scan3r_obj_projector import Scan3RIMGProjector
    from preprocessing.gt_anno_2D.scan3r_obj_img_associate import Scan3ROBJAssociator

    projector = Scan3RIMGProjector(args.data_root, split=args.split, use_rescan=True)
    projector.scan_ids = [args.scan_id]
    projector.project(0, step=1)

    cfg = update_config(config, args.config, ensure_dir=False)
    associator = Scan3ROBJAssociator(args.data_root, split=args.split, cfg=cfg)
    associator.scan_ids = [args.scan_id]
    associator.annotate_scans()


if __name__ == "__main__":
    main()
