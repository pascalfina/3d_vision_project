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
import shutil
from glob import glob
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

try:
    from preprocessing.segmentation.refine_pi3x_superpoints_with_sam_masks import (
        collect_point_mask_evidence,
        dominant_point_signatures,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    try:
        from refine_pi3x_superpoints_with_sam_masks import (
            collect_point_mask_evidence,
            dominant_point_signatures,
        )
    except ImportError:
        collect_point_mask_evidence = None
        dominant_point_signatures = None


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
    faces: Optional[np.ndarray],
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

    elements = [PlyElement.describe(vertex_data, "vertex")]
    if faces is not None and len(faces) > 0:
        face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
        face_data["vertex_indices"] = faces
        elements.append(PlyElement.describe(face_data, "face"))

    PlyData(elements, text=False).write(out_ply)

    print(f"[OK] Wrote annotated PLY ({len(vertices)} vertices, {len(faces) if faces is not None else 0} faces): {out_ply}")


def _update_single_objects_json(objects_path: str, scan_id: str, object_ids: np.ndarray):
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

    if osp.islink(objects_path):
        os.unlink(objects_path)

    with open(objects_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(valid_ids)


def update_objects_json(root_dir: str, scan_id: str, object_ids: np.ndarray):
    files_dir = osp.join(root_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    filenames = ["objects.json", "objects_sam2.json"]
    updated = []
    object_count = None
    for filename in filenames:
        objects_path = osp.join(files_dir, filename)
        object_count = _update_single_objects_json(objects_path, scan_id, object_ids)
        updated.append(objects_path)
    print(
        f"[OK] Updated object registries with {object_count} objects: "
        + ", ".join(updated)
    )


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

    # Float coordinates can be inside the image but round exactly onto the
    # exclusive width/height border.  Filter after rounding so edge samples do
    # not get splatted back onto the image boundary.
    valid_rounded = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_rounded]
    v = v[valid_rounded]
    z = z[valid_rounded]
    obj = obj[valid_rounded]

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


def _visible_projected_samples(
    pts_cam: np.ndarray,
    point_labels: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    min_depth: float = 0.1,
):
    """Project labeled 3D points and keep only the front-most sample per pixel."""
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1)
    valid &= z > float(min_depth)
    valid &= point_labels > 0
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    pts_cam_valid = pts_cam[valid]
    labels_valid = point_labels[valid].astype(np.int32, copy=False)
    z_valid = pts_cam_valid[:, 2]
    x = pts_cam_valid[:, 0]
    y = pts_cam_valid[:, 1]

    u = np.round(fx * (x / z_valid) + cx).astype(np.int32)
    v = np.round(fy * (y / z_valid) + cy).astype(np.int32)

    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_image):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    u = u[in_image]
    v = v[in_image]
    z_valid = z_valid[in_image]
    labels_valid = labels_valid[in_image]

    pixel_index = v.astype(np.int64) * int(width) + u.astype(np.int64)
    order = np.argsort(z_valid, kind="stable")
    pixel_sorted = pixel_index[order]
    keep = np.empty(len(pixel_sorted), dtype=bool)
    keep[0] = True
    keep[1:] = pixel_sorted[1:] != pixel_sorted[:-1]
    visible_idx = order[keep]

    return u[visible_idx], v[visible_idx], labels_valid[visible_idx], z_valid[visible_idx]


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

        pts_cam = (w2c @ pts_h.T).T[:, :3]
        mask2d = np.array(Image.open(mask_path))
        H, W = mask2d.shape
        u_in, v_in, labels_in, _ = _visible_projected_samples(
            pts_cam=pts_cam,
            point_labels=sam_labels,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=W,
            height=H,
            min_depth=0.1,
        )
        if len(u_in) < min_proj_points:
            continue

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


def _read_scene_colors_if_aligned(scene_ply: str, expected_points: int) -> Optional[np.ndarray]:
    if not osp.exists(scene_ply):
        return None
    ply = PlyData.read(scene_ply)
    vertex = ply["vertex"].data
    if len(vertex) != expected_points:
        return None
    names = vertex.dtype.names or ()
    if not all(name in names for name in ("red", "green", "blue")):
        return None
    return np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)


def _link_scene_ply_for_voxelise(out_ply: str, pi3x_seq_dir: str) -> None:
    pi3x_scene_dir = osp.dirname(osp.abspath(pi3x_seq_dir))
    os.makedirs(pi3x_scene_dir, exist_ok=True)
    alt_ply = osp.join(pi3x_scene_dir, "labels.instances.annotated.v2.ply")
    if osp.abspath(out_ply) == osp.abspath(alt_ply):
        return
    if osp.islink(alt_ply) or osp.exists(alt_ply):
        os.unlink(alt_ply)
    try:
        os.symlink(osp.abspath(out_ply), alt_ply)
        print(f"[OK] Symlinked SAMObject-segmented Pi3X PLY into scene dir: {alt_ply}")
    except OSError:
        shutil.copy2(out_ply, alt_ply)
        print(f"[OK] Copied SAMObject-segmented Pi3X PLY into scene dir: {alt_ply}")


def _component_labels(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int32)
    tree = cKDTree(points)
    parent = np.arange(len(points), dtype=np.int32)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = int(parent[idx])
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in tree.query_pairs(float(radius)):
        union(int(left), int(right))

    roots = np.asarray([find(i) for i in range(len(points))], dtype=np.int32)
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.int32)


def clean_small_object_components(
    points: np.ndarray,
    object_ids: np.ndarray,
    *,
    radius: float,
    min_points: int,
    min_fraction_of_largest: float,
    max_removed_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Remove only small detached 3D islands from each object label."""
    cleaned = object_ids.astype(np.int32, copy=True)
    object_stats = []
    total_removed = 0
    processed = 0
    skipped_by_safety = 0

    for obj_id in sorted(int(i) for i in np.unique(object_ids) if int(i) > 0):
        indices = np.flatnonzero(cleaned == obj_id)
        if indices.size < max(2, int(min_points)):
            continue

        component_ids = _component_labels(points[indices], radius=float(radius))
        counts = np.bincount(component_ids)
        if counts.size <= 1:
            continue

        largest = int(counts.max())
        keep_threshold = max(int(min_points), int(np.ceil(float(min_fraction_of_largest) * largest)))
        keep_components = np.flatnonzero(counts >= keep_threshold)
        keep_mask = np.isin(component_ids, keep_components)
        remove_count = int((~keep_mask).sum())
        if remove_count == 0:
            continue

        removed_fraction = remove_count / float(indices.size)
        if removed_fraction > float(max_removed_fraction):
            skipped_by_safety += 1
            object_stats.append(
                {
                    "object_id": obj_id,
                    "points": int(indices.size),
                    "components": int(counts.size),
                    "largest_component": largest,
                    "removed_points": 0,
                    "skipped_by_safety": True,
                    "candidate_removed_fraction": round(removed_fraction, 4),
                }
            )
            continue

        cleaned[indices[~keep_mask]] = 0
        total_removed += remove_count
        processed += 1
        object_stats.append(
            {
                "object_id": obj_id,
                "points": int(indices.size),
                "components": int(counts.size),
                "largest_component": largest,
                "removed_points": remove_count,
                "removed_fraction": round(removed_fraction, 4),
                "skipped_by_safety": False,
            }
        )

    stats = {
        "enabled": True,
        "radius": float(radius),
        "min_points": int(min_points),
        "min_fraction_of_largest": float(min_fraction_of_largest),
        "max_removed_fraction": float(max_removed_fraction),
        "objects_changed": int(processed),
        "objects_skipped_by_safety": int(skipped_by_safety),
        "removed_points": int(total_removed),
        "object_stats": object_stats,
    }
    print(
        "[INFO] Component cleanup: "
        f"removed_points={total_removed} objects_changed={processed} "
        f"skipped_by_safety={skipped_by_safety}"
    )
    return cleaned, stats


def clean_mask_unsupported_tails(
    points: np.ndarray,
    object_ids: np.ndarray,
    *,
    sam_root: str,
    scan_id: str,
    view_stride: int,
    vis_rtol: float,
    min_observations: int,
    dominant_ratio: float,
    core_radius: float,
    core_min_points: int,
    core_min_fraction: float,
    signature_keep_fraction: float,
    max_signatures: int,
    max_removed_fraction: float,
    large_core_min_points: int,
    large_core_max_removed_fraction: float,
) -> tuple[np.ndarray, dict]:
    """
    Remove long, weakly-supported object tails without shrinking the object core.

    SAMObject's graph can occasionally attach Pi3X wall/slab surfels to a good
    object cluster.  Detached islands are handled by component cleanup; this
    pass targets attached tails by anchoring each object to points that are
    repeatedly seen inside consistent SAMObject 2D masks.  Ambiguous points are
    kept when they are spatially close to that supported core.
    """
    if collect_point_mask_evidence is None or dominant_point_signatures is None:
        raise RuntimeError(
            "Mask-tail cleanup requires refine_pi3x_superpoints_with_sam_masks.py imports."
        )

    posed_images_dir = osp.join(sam_root, "posed_images", scan_id)
    mask_dir = osp.join(sam_root, "2D_masks", scan_id, "semantic-sam")

    point_labels, _seen_counts, evidence_stats = collect_point_mask_evidence(
        points.astype(np.float32, copy=False),
        posed_images_dir=posed_images_dir,
        mask_dir=mask_dir,
        view_stride=int(view_stride),
        vis_rtol=float(vis_rtol),
    )
    signatures, _dominant_counts, nonzero_counts, signature_stats = dominant_point_signatures(
        point_labels,
        min_observations=int(min_observations),
        min_dominant_ratio=float(dominant_ratio),
    )

    cleaned = object_ids.astype(np.int32, copy=True)
    object_stats = []
    total_removed = 0
    processed = 0
    skipped_insufficient_core = 0
    skipped_by_safety = 0

    for obj_id in sorted(int(i) for i in np.unique(object_ids) if int(i) > 0):
        indices = np.flatnonzero(cleaned == obj_id)
        if indices.size < max(2, int(core_min_points)):
            continue

        obj_signatures = signatures[indices]
        positive = obj_signatures[obj_signatures > 0]
        if positive.size == 0:
            skipped_insufficient_core += 1
            continue

        values, counts = np.unique(positive, return_counts=True)
        order = np.argsort(counts)[::-1]
        values = values[order]
        counts = counts[order]

        keep_labels = []
        cumulative = 0
        target = max(1, int(np.ceil(float(signature_keep_fraction) * positive.size)))
        for value, count in zip(values, counts):
            if len(keep_labels) >= max(1, int(max_signatures)):
                break
            keep_labels.append(int(value))
            cumulative += int(count)
            if cumulative >= target:
                break

        core_mask = np.isin(obj_signatures, np.asarray(keep_labels, dtype=np.int32))
        min_core = max(
            int(core_min_points),
            int(np.ceil(float(core_min_fraction) * float(indices.size))),
        )
        core_count = int(core_mask.sum())
        if core_count < min_core:
            skipped_insufficient_core += 1
            object_stats.append(
                {
                    "object_id": obj_id,
                    "points": int(indices.size),
                    "positive_signature_points": int(positive.size),
                    "core_points": core_count,
                    "min_core_points": int(min_core),
                    "skipped_insufficient_core": True,
                }
            )
            continue

        core_points = points[indices[core_mask]]
        tree = cKDTree(core_points)
        distances, _ = tree.query(points[indices], k=1)
        keep_mask = core_mask | (distances <= float(core_radius))
        remove_count = int((~keep_mask).sum())
        if remove_count == 0:
            continue

        removed_fraction = remove_count / float(indices.size)
        allowed_removed_fraction = float(max_removed_fraction)
        if core_count >= int(large_core_min_points):
            allowed_removed_fraction = max(
                allowed_removed_fraction,
                float(large_core_max_removed_fraction),
            )

        if removed_fraction > allowed_removed_fraction:
            skipped_by_safety += 1
            object_stats.append(
                {
                    "object_id": obj_id,
                    "points": int(indices.size),
                    "positive_signature_points": int(positive.size),
                    "core_points": core_count,
                    "kept_signature_labels": keep_labels,
                    "candidate_removed_points": remove_count,
                    "candidate_removed_fraction": round(removed_fraction, 4),
                    "allowed_removed_fraction": round(allowed_removed_fraction, 4),
                    "skipped_by_safety": True,
                }
            )
            continue

        cleaned[indices[~keep_mask]] = 0
        total_removed += remove_count
        processed += 1
        object_stats.append(
            {
                "object_id": obj_id,
                "points": int(indices.size),
                "positive_signature_points": int(positive.size),
                "nonzero_observation_points": int((nonzero_counts[indices] > 0).sum()),
                "core_points": core_count,
                "kept_signature_labels": keep_labels,
                "removed_points": remove_count,
                "removed_fraction": round(removed_fraction, 4),
                "allowed_removed_fraction": round(allowed_removed_fraction, 4),
                "skipped_by_safety": False,
            }
        )

    stats = {
        "enabled": True,
        "view_stride": int(view_stride),
        "vis_rtol": float(vis_rtol),
        "min_observations": int(min_observations),
        "dominant_ratio": float(dominant_ratio),
        "core_radius": float(core_radius),
        "core_min_points": int(core_min_points),
        "core_min_fraction": float(core_min_fraction),
        "signature_keep_fraction": float(signature_keep_fraction),
        "max_signatures": int(max_signatures),
        "max_removed_fraction": float(max_removed_fraction),
        "large_core_min_points": int(large_core_min_points),
        "large_core_max_removed_fraction": float(large_core_max_removed_fraction),
        "objects_changed": int(processed),
        "objects_skipped_insufficient_core": int(skipped_insufficient_core),
        "objects_skipped_by_safety": int(skipped_by_safety),
        "removed_points": int(total_removed),
        "evidence": evidence_stats,
        "point_signatures": signature_stats,
        "object_stats": object_stats,
    }
    print(
        "[INFO] Mask-tail cleanup: "
        f"removed_points={total_removed} objects_changed={processed} "
        f"skipped_core={skipped_insufficient_core} skipped_safety={skipped_by_safety}"
    )
    return cleaned, stats


def create_pi3x_graph_segmented_scene(
    sam_root: str,
    scan_id: str,
    sam_points: np.ndarray,
    objectx_ids: np.ndarray,
    pi3x_seq_dir: str,
    out_root_dir: str,
    skip: Optional[int] = None,
    projection_dilation: int = 2,
    clean_components: bool = False,
    component_radius: float = 0.10,
    component_min_points: int = 24,
    component_min_fraction: float = 0.03,
    component_max_removed_fraction: float = 0.20,
    clean_mask_tails: bool = False,
    mask_tail_view_stride: int = 1,
    mask_tail_vis_rtol: float = 0.15,
    mask_tail_min_observations: int = 2,
    mask_tail_dominant_ratio: float = 0.45,
    mask_tail_core_radius: float = 0.20,
    mask_tail_core_min_points: int = 32,
    mask_tail_core_min_fraction: float = 0.05,
    mask_tail_signature_keep_fraction: float = 0.75,
    mask_tail_max_signatures: int = 8,
    mask_tail_max_removed_fraction: float = 0.25,
    mask_tail_large_core_min_points: int = 512,
    mask_tail_large_core_max_removed_fraction: float = 0.50,
) -> np.ndarray:
    """
    Export the actual SAMObject graph result on the Pi3X support surface.

    In Pi3X mode the SAMObject graph input is already a stable Pi3X surfel
    scene.  The correct Object-X output is therefore the same surfel cloud
    labeled by SAMObject's final 3D graph clusters.  This avoids the old
    raw-mask lifting shortcut, where dense 2D masks could paint large noisy
    Pi3X slabs as objects after SAMObject had already finished.
    """
    scene_ply = osp.join(sam_root, "scenes", scan_id, "labels.instances.annotated.v2.ply")
    colors = _read_scene_colors_if_aligned(scene_ply, expected_points=len(sam_points))
    export_ids = objectx_ids.astype(np.int32)
    if clean_components:
        export_ids, cleanup_stats = clean_small_object_components(
            sam_points,
            export_ids,
            radius=component_radius,
            min_points=component_min_points,
            min_fraction_of_largest=component_min_fraction,
            max_removed_fraction=component_max_removed_fraction,
        )
        stats_path = osp.join(out_root_dir, "scenes", scan_id, "samobject_export_cleanup_stats.json")
        os.makedirs(osp.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(cleanup_stats, f, indent=2)
    if clean_mask_tails:
        export_ids, tail_stats = clean_mask_unsupported_tails(
            sam_points,
            export_ids,
            sam_root=sam_root,
            scan_id=scan_id,
            view_stride=mask_tail_view_stride,
            vis_rtol=mask_tail_vis_rtol,
            min_observations=mask_tail_min_observations,
            dominant_ratio=mask_tail_dominant_ratio,
            core_radius=mask_tail_core_radius,
            core_min_points=mask_tail_core_min_points,
            core_min_fraction=mask_tail_core_min_fraction,
            signature_keep_fraction=mask_tail_signature_keep_fraction,
            max_signatures=mask_tail_max_signatures,
            max_removed_fraction=mask_tail_max_removed_fraction,
            large_core_min_points=mask_tail_large_core_min_points,
            large_core_max_removed_fraction=mask_tail_large_core_max_removed_fraction,
        )
        stats_path = osp.join(out_root_dir, "scenes", scan_id, "samobject_export_mask_tail_stats.json")
        os.makedirs(osp.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(tail_stats, f, indent=2)

    out_ply = osp.join(out_root_dir, "scenes", scan_id, "labels.instances.annotated.v2.ply")
    write_annotated_ply(
        out_ply=out_ply,
        vertices=sam_points,
        faces=None,
        object_ids=export_ids,
        colors=colors,
    )
    _link_scene_ply_for_voxelise(out_ply, pi3x_seq_dir)

    create_gt_projection_pkl(
        root_dir=out_root_dir,
        scan_id=scan_id,
        points_world=sam_points,
        point_object_ids=export_ids,
        skip=skip,
        dilation=projection_dilation,
        seq_root_dir=sam_root,
    )
    return export_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--scan_id", required=True)

    parser.add_argument(
        "--mesh_path",
        required=False,
        default=None,
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
    parser.add_argument(
        "--use_pi3x_surface",
        action="store_true",
        help=(
            "Export SAMObject's final 3D graph labels on the Pi3X support surface. "
            "Use with voxelise OBJECTX_VOXEL_OBJECT_SOURCE=gt_mesh."
        ),
    )
    parser.add_argument(
        "--use_pi3x_mesh",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pi3x_seq_dir",
        default=None,
        help="Path to the Pi3X sequence dir; used to publish the labeled PLY next to that scene.",
    )
    parser.add_argument("--clean_components", action="store_true")
    parser.add_argument("--component_radius", type=float, default=0.10)
    parser.add_argument("--component_min_points", type=int, default=24)
    parser.add_argument("--component_min_fraction", type=float, default=0.03)
    parser.add_argument("--component_max_removed_fraction", type=float, default=0.20)
    parser.add_argument("--clean_mask_tails", action="store_true")
    parser.add_argument("--mask_tail_view_stride", type=int, default=1)
    parser.add_argument("--mask_tail_vis_rtol", type=float, default=0.15)
    parser.add_argument("--mask_tail_min_observations", type=int, default=2)
    parser.add_argument("--mask_tail_dominant_ratio", type=float, default=0.45)
    parser.add_argument("--mask_tail_core_radius", type=float, default=0.20)
    parser.add_argument("--mask_tail_core_min_points", type=int, default=32)
    parser.add_argument("--mask_tail_core_min_fraction", type=float, default=0.05)
    parser.add_argument("--mask_tail_signature_keep_fraction", type=float, default=0.75)
    parser.add_argument("--mask_tail_max_signatures", type=int, default=8)
    parser.add_argument("--mask_tail_max_removed_fraction", type=float, default=0.25)
    parser.add_argument("--mask_tail_large_core_min_points", type=int, default=512)
    parser.add_argument("--mask_tail_large_core_max_removed_fraction", type=float, default=0.50)

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

    if args.use_pi3x_surface or args.use_pi3x_mesh:
        if not args.pi3x_seq_dir:
            raise ValueError("--pi3x_seq_dir is required when --use_pi3x_surface is set.")
        oids_final = create_pi3x_graph_segmented_scene(
            sam_root=root_dir,
            scan_id=scan_id,
            sam_points=sam_points,
            objectx_ids=sam_object_ids,
            pi3x_seq_dir=args.pi3x_seq_dir,
            out_root_dir=out_root_dir,
            skip=args.frame_skip,
            projection_dilation=args.projection_dilation,
            clean_components=args.clean_components,
            component_radius=args.component_radius,
            component_min_points=args.component_min_points,
            component_min_fraction=args.component_min_fraction,
            component_max_removed_fraction=args.component_max_removed_fraction,
            clean_mask_tails=args.clean_mask_tails,
            mask_tail_view_stride=args.mask_tail_view_stride,
            mask_tail_vis_rtol=args.mask_tail_vis_rtol,
            mask_tail_min_observations=args.mask_tail_min_observations,
            mask_tail_dominant_ratio=args.mask_tail_dominant_ratio,
            mask_tail_core_radius=args.mask_tail_core_radius,
            mask_tail_core_min_points=args.mask_tail_core_min_points,
            mask_tail_core_min_fraction=args.mask_tail_core_min_fraction,
            mask_tail_signature_keep_fraction=args.mask_tail_signature_keep_fraction,
            mask_tail_max_signatures=args.mask_tail_max_signatures,
            mask_tail_max_removed_fraction=args.mask_tail_max_removed_fraction,
            mask_tail_large_core_min_points=args.mask_tail_large_core_min_points,
            mask_tail_large_core_max_removed_fraction=args.mask_tail_large_core_max_removed_fraction,
        )
        update_objects_json(
            root_dir=out_root_dir,
            scan_id=scan_id,
            object_ids=oids_final,
        )
        return

    if not args.mesh_path:
        raise ValueError("--mesh_path is required unless --use_pi3x_surface is set.")

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
