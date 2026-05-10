#!/usr/bin/env python3
"""
Prepare SAM2Object Graph Clustering output for Object-X voxelise_features.py.

Outputs:
  1. <root_dir>/scenes/<scan_id>/labels.instances.annotated.v2.ply
  2. <root_dir>/files/objects.json
  3. <root_dir>/files/gt_projection/obj_id_pkl/<scan_id>.pkl

Expected SAM2Object inputs:
  - points.npy or points.pts with shape [N, 3]
  - labels_fine_global.npy with shape [N]

python preprocessing/segmentation/prepare_sam2object_for_objectx.py \
  --root_dir /cluster/scratch/ealegret/sam2object \
  --scan_id 5341b7e3-8a66-2cdd-8709-66a2159f0017 \
  --mesh_path /cluster/scratch/ealegret/sam2object/scenes/5341b7e3-8a66-2cdd-8709-66a2159f0017/mesh.refined.v2.obj \
  --sam_points /cluster/scratch/ealegret/sam2object/scenes/5341b7e3-8a66-2cdd-8709-66a2159f0017/results/5341b7e3-8a66-2cdd-8709-66a2159f0017_points.npy \
  --sam_labels /cluster/scratch/ealegret/sam2object/scenes/5341b7e3-8a66-2cdd-8709-66a2159f0017/results/5341b7e3-8a66-2cdd-8709-66a2159f0017_labels_fine_global.npy \
  --projection_dilation 2
"""

import argparse
import json
import os
import os.path as osp
import pickle
from glob import glob
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def load_points(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    return np.loadtxt(path).astype(np.float32)


def remap_labels_to_objectx_ids(labels: np.ndarray):
    """
    SAM2Object labels may be arbitrary and may include 0 (background) or -1.
    Object-X should use:
      0 = background / unlabeled
      1..K = valid object ids
    """
    labels = labels.astype(np.int64)
    valid = labels > 0  # exclude 0 (background) and -1
    unique = np.unique(labels[valid])

    mapping = {int(old): int(new_id) for new_id, old in enumerate(unique, start=1)}

    objectx_labels = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        objectx_labels[labels == old] = new

    return objectx_labels, mapping


def read_mesh_vertices_faces(mesh_path: str):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Could not read valid mesh from {mesh_path}")

    colors = None
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)

    return vertices, faces, colors


def transfer_point_labels_to_mesh_vertices(
    mesh_vertices: np.ndarray,
    sam_points: np.ndarray,
    sam_object_ids: np.ndarray,
):
    """
    If SAM2Object points are exactly the mesh vertices and in the same order,
    this returns labels directly. Otherwise, nearest-neighbour transfer.
    """
    if len(mesh_vertices) == len(sam_points):
        max_abs_diff = np.max(np.abs(mesh_vertices - sam_points))
        if max_abs_diff < 1e-5:
            print("[INFO] SAM2Object points match mesh vertices directly.")
            return sam_object_ids.astype(np.int32)

    print("[INFO] Transferring SAM2Object labels to mesh vertices using nearest neighbour.")
    tree = cKDTree(sam_points)
    dist, nn_idx = tree.query(mesh_vertices, k=1)

    print(
        f"[INFO] NN transfer distances: "
        f"mean={dist.mean():.6f}, median={np.median(dist):.6f}, max={dist.max():.6f}"
    )

    return sam_object_ids[nn_idx].astype(np.int32)


def write_annotated_ply(
    out_ply: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    object_ids: np.ndarray,
    colors: Optional[np.ndarray] = None,
):
    os.makedirs(osp.dirname(out_ply), exist_ok=True)

    if colors is not None:
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("objectId", "i4"),
        ]
        vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
        vertex_data["x"] = vertices[:, 0]
        vertex_data["y"] = vertices[:, 1]
        vertex_data["z"] = vertices[:, 2]
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]
        vertex_data["objectId"] = object_ids
    else:
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("objectId", "i4"),
        ]
        vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
        vertex_data["x"] = vertices[:, 0]
        vertex_data["y"] = vertices[:, 1]
        vertex_data["z"] = vertices[:, 2]
        vertex_data["objectId"] = object_ids

    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    ).write(out_ply)

    print(f"[OK] Wrote annotated mesh: {out_ply}")


def update_objects_json(root_dir: str, scan_id: str, object_ids: np.ndarray):
    objects_path = osp.join(root_dir, "files", "objects.json")
    os.makedirs(osp.dirname(objects_path), exist_ok=True)

    valid_ids = sorted(int(i) for i in np.unique(object_ids) if int(i) > 0)

    new_scan_entry = {
        "scan": scan_id,
        "objects": [
            {
                "id": int(obj_id),
                "label": "object",
                "global_id": int(obj_id),
            }
            for obj_id in valid_ids
        ],
    }

    if osp.exists(objects_path):
        with open(objects_path, "r") as f:
            data = json.load(f)
    else:
        data = {"scans": []}

    data["scans"] = [s for s in data["scans"] if s.get("scan") != scan_id]
    data["scans"].append(new_scan_entry)

    with open(objects_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Updated objects.json with {len(valid_ids)} objects: {objects_path}")


def load_intrinsics_from_3rscan_info(root_dir: str, scan_id: str):
    info_path = osp.join(root_dir, "scenes", scan_id, "sequence", "_info.txt")

    width = height = None
    K = None

    with open(info_path, "r") as f:
        for line in f:
            if "m_colorWidth" in line:
                width = int(float(line.split("=")[1]))
            elif "m_colorHeight" in line:
                height = int(float(line.split("=")[1]))
            elif "m_calibrationColorIntrinsic" in line:
                vals = [float(v) for v in line.split("=")[1].split()]
                K = np.array(
                    [
                        [vals[0], 0.0, vals[2]],
                        [0.0, vals[5], vals[6]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

    if width is None or height is None or K is None:
        raise RuntimeError(f"Could not parse intrinsics from {info_path}")

    return K, width, height


def load_frame_ids(root_dir: str, scan_id: str, skip: Optional[int] = None):
    pattern = osp.join(root_dir, "scenes", scan_id, "sequence", "frame-*.color.jpg")
    paths = sorted(glob(pattern))

    frame_ids = []
    for p in paths:
        name = osp.basename(p)
        frame_id = name.replace("frame-", "").replace(".color.jpg", "")
        frame_ids.append(frame_id)

    if skip is not None and skip > 1:
        frame_ids = frame_ids[::skip]

    return frame_ids


def project_labeled_points_to_frame(
    points_world: np.ndarray,
    point_object_ids: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    dilation: int = 2,
):
    """
    Creates one HxW objectId map.

    Assumption:
      3RScan frame pose is camera-to-world, so we invert it to get world-to-camera.
    """
    T_w2c = np.linalg.inv(pose_c2w)

    pts_h = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    pts_cam = (T_w2c @ pts_h.T).T[:, :3]

    z_all = pts_cam[:, 2]

    valid = np.isfinite(z_all)
    valid &= z_all > 1e-6
    valid &= point_object_ids > 0

    pts_cam_valid = pts_cam[valid]
    obj_valid = point_object_ids[valid]

    z = pts_cam_valid[:, 2]
    x = pts_cam_valid[:, 0]
    y = pts_cam_valid[:, 1]

    u_float = K[0, 0] * (x / z) + K[0, 2]
    v_float = K[1, 1] * (y / z) + K[1, 2]

    valid_img = np.isfinite(u_float)
    valid_img &= np.isfinite(v_float)
    valid_img &= u_float >= 0
    valid_img &= u_float < width
    valid_img &= v_float >= 0
    valid_img &= v_float < height

    u = np.round(u_float[valid_img]).astype(np.int32)
    v = np.round(v_float[valid_img]).astype(np.int32)
    z = z[valid_img]
    obj = obj_valid[valid_img]

    obj_map = np.zeros((height, width), dtype=np.int32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    # nearest point wins
    order = np.argsort(z)
    for idx in order:
        uu = u[idx]
        vv = v[idx]
        zz = z[idx]
        oid = obj[idx]

        if dilation <= 0:
            if zz < depth_map[vv, uu]:
                depth_map[vv, uu] = zz
                obj_map[vv, uu] = oid
        else:
            y0 = max(0, vv - dilation)
            y1 = min(height, vv + dilation + 1)
            x0 = max(0, uu - dilation)
            x1 = min(width, uu + dilation + 1)

            patch_depth = depth_map[y0:y1, x0:x1]
            update = zz < patch_depth
            patch_obj = obj_map[y0:y1, x0:x1]

            patch_depth[update] = zz
            patch_obj[update] = oid

            depth_map[y0:y1, x0:x1] = patch_depth
            obj_map[y0:y1, x0:x1] = patch_obj

    return obj_map


def create_2d_mask_pkl(
    sam_root: str,
    scan_id: str,
    sam_points: np.ndarray,
    sam_labels: np.ndarray,
    objectx_ids: np.ndarray,
    out_root_dir: str,
    skip: Optional[int] = None,
    min_proj_points: int = 5,
):
    """
    Convert SAM2Object 2D tracking masks directly to ObjectX pkl format.

    Per-frame strategy: for each frame, project 3D cluster points to find which
    2D track ID corresponds to which cluster IN THAT FRAME. This handles SAM2's
    track ID restarts across video batches (e.g. IDs 2-31 in frames 0-47, then
    IDs 100+ in later frames). The dense 2D masks give correct pixel-surface
    correspondence for Pi3X depth lifting.
    """
    posed_dir = osp.join(sam_root, "posed_images", scan_id)
    mask_dir = osp.join(sam_root, "2D_masks", scan_id, "semantic-sam")

    if not osp.isdir(posed_dir):
        raise FileNotFoundError(f"posed_images not found: {posed_dir}")
    if not osp.isdir(mask_dir):
        raise FileNotFoundError(f"2D_masks not found: {mask_dir}")

    # Load intrinsics (K_color is 4x4)
    K4 = np.loadtxt(osp.join(posed_dir, "intrinsics_color.txt"))
    K = K4[:3, :3].astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Frame list from posed_images
    jpg_files = sorted(glob(osp.join(posed_dir, "*.jpg")))
    frame_ids = [osp.basename(f).replace(".jpg", "") for f in jpg_files]
    if skip is not None and skip > 1:
        frame_ids = frame_ids[::skip]

    # Pre-compute: cluster raw_label -> objectx_id (1-to-1 mapping)
    cluster_to_oid = {}
    for cl in np.unique(sam_labels[sam_labels > 0]):
        cl = int(cl)
        oid = int(objectx_ids[sam_labels == cl][0])
        if oid > 0:
            cluster_to_oid[cl] = oid

    # Homogeneous coordinates for all points (computed once)
    pts_h = np.hstack([sam_points, np.ones((len(sam_points), 1), dtype=np.float32)])

    out = {}
    frames_with_content = 0

    for fid in frame_ids:
        pose_path = osp.join(posed_dir, f"{fid}.txt")
        mask_path = osp.join(mask_dir, f"maskraw_{fid}.png")
        if not osp.exists(pose_path) or not osp.exists(mask_path):
            continue

        pose_c2w = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        w2c = np.linalg.inv(pose_c2w)

        # Project all PLY points into this frame
        pts_cam = (w2c @ pts_h.T).T[:, :3]
        in_front = pts_cam[:, 2] > 0.1

        u_all = np.empty(len(pts_cam), dtype=np.int32)
        v_all = np.empty(len(pts_cam), dtype=np.int32)
        np.divide(pts_cam[:, 0], pts_cam[:, 2], out=u_all.astype(np.float32))
        u_all = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(np.int32)
        v_all = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(np.int32)

        mask2d = np.array(Image.open(mask_path))
        H, W = mask2d.shape

        in_image = in_front & (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
        if in_image.sum() < min_proj_points:
            continue

        u_in = u_all[in_image]
        v_in = v_all[in_image]
        labels_in = sam_labels[in_image]

        # Per-frame vote: cluster -> {track_id: count}
        frame_votes = {}
        for uu, vv, cl in zip(u_in, v_in, labels_in):
            if cl <= 0:
                continue
            tid = int(mask2d[vv, uu])
            if tid == 0:
                continue
            cl = int(cl)
            if cl not in frame_votes:
                frame_votes[cl] = {}
            frame_votes[cl][tid] = frame_votes[cl].get(tid, 0) + 1

        if not frame_votes:
            continue

        # Winner track per cluster in this frame
        cluster_to_track_frame = {
            cl: max(tvotes, key=tvotes.get)
            for cl, tvotes in frame_votes.items()
            if tvotes
        }

        # Resolve collisions: if multiple clusters claim the same track_id,
        # the one with the highest vote count for that track wins.
        track_winner_votes = {}  # track_id -> (winning_cluster, vote_count)
        for cl, tid in cluster_to_track_frame.items():
            vote_count = frame_votes[cl][tid]
            if tid not in track_winner_votes or vote_count > track_winner_votes[tid][1]:
                track_winner_votes[tid] = (cl, vote_count)

        # Build track_id -> objectx_id for this frame
        track_to_oid_frame = {}
        for tid, (cl, _) in track_winner_votes.items():
            oid = cluster_to_oid.get(cl, 0)
            if oid > 0:
                track_to_oid_frame[tid] = oid

        if not track_to_oid_frame:
            continue

        # Apply dense 2D mask
        obj_map = np.zeros(mask2d.shape, dtype=np.int32)
        for tid, oid in track_to_oid_frame.items():
            obj_map[mask2d == tid] = oid

        if (obj_map > 0).any():
            out[fid] = obj_map
            frames_with_content += 1

    print(f"[INFO] Built per-frame 2D mask pkl: {frames_with_content}/{len(frame_ids)} frames with content")

    save_dir = osp.join(out_root_dir, "files", "gt_projection", "obj_id_pkl")
    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, f"{scan_id}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[OK] Wrote 2D-mask pkl with {len(out)} frames: {out_path}")


def create_gt_projection_pkl(
    root_dir: str,
    scan_id: str,
    points_world: np.ndarray,
    point_object_ids: np.ndarray,
    skip: Optional[int],
    dilation: int,
    seq_root_dir: Optional[str] = None,
):
    _seq = seq_root_dir if seq_root_dir else root_dir
    K, width, height = load_intrinsics_from_3rscan_info(_seq, scan_id)
    frame_ids = load_frame_ids(_seq, scan_id, skip=skip)

    out = {}

    for frame_id in frame_ids:
        pose_path = osp.join(
            _seq,
            "scenes",
            scan_id,
            "sequence",
            f"frame-{frame_id}.pose.txt",
        )
        pose_c2w = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)

        obj_map = project_labeled_points_to_frame(
            points_world=points_world,
            point_object_ids=point_object_ids,
            pose_c2w=pose_c2w,
            K=K,
            width=width,
            height=height,
            dilation=dilation,
        )

        out[frame_id] = obj_map

    save_dir = osp.join(root_dir, "files", "gt_projection", "obj_id_pkl")
    os.makedirs(save_dir, exist_ok=True)

    out_path = osp.join(save_dir, f"{scan_id}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[OK] Wrote gt_projection pkl with {len(out)} frames: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--scan_id", required=True)

    parser.add_argument(
        "--mesh_path",
        required=True,
        help="Original mesh without annotations, e.g. mesh.refined.v2.obj or .ply",
    )
    parser.add_argument(
        "--sam_points",
        required=True,
        help="SAM2Object points.npy or points.pts used by graph clustering",
    )
    parser.add_argument(
        "--sam_labels",
        required=True,
        help="SAM2Object labels_fine_global.npy",
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=None,
        help="Optional frame skip for gt_projection. Use 5 if you want every 5th frame.",
    )
    parser.add_argument(
        "--projection_dilation",
        type=int,
        default=2,
        help="Pixel dilation radius for projected point masks.",
    )

    parser.add_argument(
        "--output_root_dir",
        default=None,
        help=(
            "Root dir for writing outputs (objects.json, gt_projection pkl, PLY). "
            "Defaults to --root_dir. Use this to write masks into a different "
            "reconstruction root while reading sequence data from root_dir."
        ),
    )
    parser.add_argument(
        "--use_2d_masks",
        action="store_true",
        help=(
            "Use SAM2Object 2D tracking masks (2D_masks/) directly instead of "
            "projecting the 3D mesh. Produces denser masks with no depth ambiguity."
        ),
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    out_root_dir = args.output_root_dir if args.output_root_dir else root_dir
    scan_id = args.scan_id

    sam_points = load_points(args.sam_points)
    raw_labels = np.load(args.sam_labels)

    if len(sam_points) != len(raw_labels):
        raise RuntimeError(
            f"SAM points and labels length mismatch: "
            f"{len(sam_points)} vs {len(raw_labels)}"
        )

    sam_object_ids, old_to_new = remap_labels_to_objectx_ids(raw_labels)

    print(f"[INFO] SAM2Object → Object-X id mapping:")
    for old, new in old_to_new.items():
        print(f"  SAM label {old} -> Object-X objectId {new}")

    mesh_vertices, faces, colors = read_mesh_vertices_faces(args.mesh_path)

    vertex_object_ids = transfer_point_labels_to_mesh_vertices(
        mesh_vertices=mesh_vertices,
        sam_points=sam_points,
        sam_object_ids=sam_object_ids,
    )

    out_ply = osp.join(
        out_root_dir,
        "scenes",
        scan_id,
        "labels.instances.annotated.v2.ply",
    )

    write_annotated_ply(
        out_ply=out_ply,
        vertices=mesh_vertices,
        faces=faces,
        object_ids=vertex_object_ids,
        colors=colors,
    )

    update_objects_json(
        root_dir=out_root_dir,
        scan_id=scan_id,
        object_ids=vertex_object_ids,
    )

    if args.use_2d_masks:
        create_2d_mask_pkl(
            sam_root=root_dir,
            scan_id=scan_id,
            sam_points=sam_points,
            sam_labels=raw_labels,
            objectx_ids=sam_object_ids,
            out_root_dir=out_root_dir,
            skip=args.frame_skip,
        )
    else:
        create_gt_projection_pkl(
            root_dir=out_root_dir,
            scan_id=scan_id,
            points_world=mesh_vertices,
            point_object_ids=vertex_object_ids,
            skip=args.frame_skip,
            dilation=args.projection_dilation,
            seq_root_dir=root_dir,
        )


if __name__ == "__main__":
    main()


"""
Color Version:
ROOT="/cluster/scratch/ealegret/sam2object"
SCAN_ID="5341b7e3-8a66-2cdd-8709-66a2159f0017"

python - <<PY
from plyfile import PlyData, PlyElement
import numpy as np
import os

in_ply = f"$ROOT/scenes/$SCAN_ID/labels.instances.annotated.v2.ply"
out_ply = f"$ROOT/scenes/$SCAN_ID/labels.instances.sam2object_coloured.ply"

ply = PlyData.read(in_ply)
v = ply["vertex"].data
object_ids = v["objectId"].astype(np.int32)

rng = np.random.default_rng(0)
max_id = int(object_ids.max())
palette = rng.integers(40, 255, size=(max_id + 1, 3), dtype=np.uint8)
palette[0] = 0

colors = palette[object_ids]

vertex_dtype = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("objectId", "i4"),
]

vertex_data = np.empty(len(v), dtype=vertex_dtype)
vertex_data["x"] = v["x"]
vertex_data["y"] = v["y"]
vertex_data["z"] = v["z"]
vertex_data["red"] = colors[:, 0]
vertex_data["green"] = colors[:, 1]
vertex_data["blue"] = colors[:, 2]
vertex_data["objectId"] = object_ids

elements = [PlyElement.describe(vertex_data, "vertex")]

if "face" in ply:
    face_data = ply["face"].data
    elements.append(PlyElement.describe(face_data, "face"))

PlyData(elements, text=False).write(out_ply)

print("saved:", out_ply)
PY
"""