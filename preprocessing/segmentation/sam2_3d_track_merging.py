from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

def _norm_frame_id(frame_id: int | str) -> str:
    if isinstance(frame_id, int):
        return f"{frame_id:06d}"
    text = str(frame_id).strip()
    if text.startswith("frame-"):
        text = text.split("frame-", 1)[1]
    text = text.split(".", 1)[0]
    return f"{int(text):06d}" if text.isdigit() else text


def _load_intrinsics_local(scenes_root: Path, scan_id: str, sensor: str = "depth") -> dict:
    info_path = scenes_root / scan_id / "sequence" / "_info.txt"
    width_key = "m_colorWidth" if sensor == "color" else "m_depthWidth"
    height_key = "m_colorHeight" if sensor == "color" else "m_depthHeight"
    calib_key = "m_calibrationColorIntrinsic" if sensor == "color" else "m_calibrationDepthIntrinsic"
    intrinsic_mat = None
    intrinsic_width = None
    intrinsic_height = None
    with open(info_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if width_key in line:
            intrinsic_width = float(line.split("= ", 1)[1].strip())
        elif height_key in line:
            intrinsic_height = float(line.split("= ", 1)[1].strip())
        elif calib_key in line:
            vals = line.split("= ", 1)[1].strip().split(" ")
            fx, cx = float(vals[0]), float(vals[2])
            fy, cy = float(vals[5]), float(vals[6])
            intrinsic_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if intrinsic_mat is None or intrinsic_width is None or intrinsic_height is None:
        raise ValueError(f"Could not parse intrinsics from {info_path} ({sensor})")
    return {"width": intrinsic_width, "height": intrinsic_height, "intrinsic_mat": intrinsic_mat}


def _load_depth_map_local(depth_file: Path, scale: float) -> np.ndarray:
    depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Missing depth map: {depth_file}")
    return depth.astype(np.float32) / float(scale)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def _bbox_intersects(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> bool:
    return bool(np.all(max_a >= min_b) and np.all(max_b >= min_a))


def _bbox_iou_3d(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dim = np.maximum(inter_max - inter_min, 0.0)
    inter = float(np.prod(inter_dim))
    va = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vb = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union = va + vb - inter
    return inter / max(union, 1e-12)


def _validate_geometry_source_flags(gcfg: dict) -> tuple[bool, bool]:
    use_gt_depth = bool(gcfg.get("use_gt_depth", False))
    use_pred_depth = bool(gcfg.get("use_pred_depth", False))
    use_gt_pose = bool(gcfg.get("use_gt_pose", False))
    use_pred_pose = bool(gcfg.get("use_pred_pose", False))
    if not (use_gt_depth and use_gt_pose):
        raise ValueError("GT-only mode requires use_gt_depth=true and use_gt_pose=true")
    if use_pred_depth or use_pred_pose:
        raise ValueError("GT-only mode requires use_pred_depth=false and use_pred_pose=false")
    return use_gt_depth, use_gt_pose


@dataclass
class Track3DHypothesis:
    hypothesis_id: int
    source_track_id: int
    component_id: int
    active_frame_ids: list[str]
    masks_by_frame: dict[str, np.ndarray]
    probs_by_frame: dict[str, np.ndarray]
    world_voxels_core: set[tuple[int, int, int]]
    world_voxels_soft: set[tuple[int, int, int]]
    voxel_support_count: dict[tuple[int, int, int], int]
    voxel_prob_sum: dict[tuple[int, int, int], float]
    points_world: np.ndarray | None
    center_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    scale_world: float
    num_lifted_points: int
    num_core_voxels: int
    num_soft_voxels: int
    num_support_views: int
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeEdge:
    i: int
    j: int
    voxel_iou: float
    containment_i_to_j: float
    containment_j_to_i: float
    centroid_dist_norm: float
    bbox_iou_3d: float
    reproj_i_to_j: float | None
    reproj_j_to_i: float | None
    reproj_mean: float | None
    coview_conflict_frames: int
    should_block: bool
    should_merge: bool
    score: float
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Consolidated3DObject:
    object_id: int
    member_hypothesis_ids: list[int]
    member_track_ids: list[int]
    world_voxels_core: set[tuple[int, int, int]]
    world_voxels_soft: set[tuple[int, int, int]]
    center_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    scale_world: float
    active_frame_ids: list[str]
    confidence: float
    stats: dict[str, Any] = field(default_factory=dict)


class GeometryProvider:
    def __init__(self, scene_id: str, frame_ids: list[str], cfg: dict, dataset_root: str):
        self.scene_id = scene_id
        self.frame_ids = [_norm_frame_id(fid) for fid in frame_ids]
        self.cfg = cfg
        self.dataset_root = Path(dataset_root)
        self.gcfg = cfg["sam3d_merge"]["geometry_source"]
        self.use_gt_depth, self.use_gt_pose = _validate_geometry_source_flags(self.gcfg)
        self.gt_root = self.dataset_root / "scenes"

        self.depth_shift = 1000.0
        self.depth_intr = _load_intrinsics_local(self.gt_root, scene_id, sensor="depth")
        self.color_intr = _load_intrinsics_local(self.gt_root, scene_id, sensor="color")

    def get_depth(self, frame_id: str) -> np.ndarray:
        fid = _norm_frame_id(frame_id)
        depth_path = self.gt_root / self.scene_id / "sequence" / f"frame-{fid}.depth.pgm"
        return _load_depth_map_local(depth_path, self.depth_shift)

    def get_pose_world_from_cam(self, frame_id: str) -> np.ndarray:
        fid = _norm_frame_id(frame_id)
        pose_path = self.gt_root / self.scene_id / "sequence" / f"frame-{fid}.pose.txt"
        return np.loadtxt(pose_path).astype(np.float32).reshape(4, 4)

    def get_intrinsics(self, frame_id: str, for_depth: bool = True) -> dict:
        _ = frame_id
        return self.depth_intr if for_depth else self.color_intr


def _points_to_voxels(points_world: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points_world / float(voxel_size)).astype(np.int32)


def _connected_components_26(voxels: set[tuple[int, int, int]]) -> list[set[tuple[int, int, int]]]:
    if not voxels:
        return []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)]
    unseen = set(voxels)
    components = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        comp = {seed}
        while stack:
            v = stack.pop()
            for o in offsets:
                n = (v[0] + o[0], v[1] + o[1], v[2] + o[2])
                if n in unseen:
                    unseen.remove(n)
                    comp.add(n)
                    stack.append(n)
        components.append(comp)
    return components


def build_track_3d_hypotheses(tracks: dict, frame_ids: list[str], geometry_provider: GeometryProvider, cfg: dict) -> list[Track3DHypothesis]:
    mcfg = cfg["sam3d_merge"]["masks"]
    lcfg = cfg["sam3d_merge"]["lifting"]
    fcfg = cfg["sam3d_merge"]["track_filtering"]
    voxel_size = float(lcfg["voxel_size_world"])
    hyps: list[Track3DHypothesis] = []
    hid = 0

    for track_id, track_data in tracks.items():
        masks_by_frame = track_data.get("masks_by_frame", {})
        probs_by_frame = track_data.get("probs_by_frame", {})
        active = []
        core_set: set[tuple[int, int, int]] = set()
        soft_set: set[tuple[int, int, int]] = set()
        support: dict[tuple[int, int, int], int] = {}
        prob_sum: dict[tuple[int, int, int], float] = {}
        points_all = []

        for frame_id in frame_ids:
            fid = _norm_frame_id(frame_id)
            prob = probs_by_frame.get(fid)
            mask = masks_by_frame.get(fid)
            if prob is None and mask is None:
                continue
            if prob is None:
                prob = mask.astype(np.float32)
            if mask is None:
                mask = prob > 0.5

            depth = geometry_provider.get_depth(fid)
            pose = geometry_provider.get_pose_world_from_cam(fid)
            dintr = geometry_provider.get_intrinsics(fid, for_depth=True)
            dw, dh = int(dintr["width"]), int(dintr["height"])
            kmat = dintr["intrinsic_mat"].astype(np.float32)
            kinv = np.linalg.inv(kmat).astype(np.float32)

            if prob.shape[:2] != (dh, dw):
                prob_depth = cv2.resize(prob.astype(np.float32), (dw, dh), interpolation=cv2.INTER_LINEAR)
            else:
                prob_depth = prob.astype(np.float32)

            core_depth = prob_depth > float(mcfg["core_prob_thresh"])
            soft_depth = prob_depth > float(mcfg["soft_prob_thresh"])
            if int(soft_depth.sum()) < int(mcfg["min_active_area"]):
                continue
            if int(soft_depth.sum()) > int(mcfg["max_pixels_per_mask"]):
                yy_all, xx_all = np.nonzero(soft_depth)
                keep_n = int(mcfg["max_pixels_per_mask"])
                sel = np.linspace(0, len(xx_all) - 1, num=keep_n, dtype=int)
                sparse = np.zeros_like(soft_depth, dtype=bool)
                sparse[yy_all[sel], xx_all[sel]] = True
                soft_depth = sparse
                core_depth = core_depth & soft_depth

            y, x = np.nonzero(soft_depth)
            if x.size == 0:
                continue
            z = depth[y, x].astype(np.float32)
            valid = (z > float(lcfg["min_valid_depth"])) & (z < float(lcfg["max_valid_depth"])) & np.isfinite(z)
            if not np.any(valid):
                continue
            x, y, z = x[valid], y[valid], z[valid]
            if bool(lcfg.get("remove_depth_outliers", True)) and z.size >= 20:
                lo = np.quantile(z, float(lcfg["depth_outlier_quantile_low"]))
                hi = np.quantile(z, float(lcfg["depth_outlier_quantile_high"]))
                keep = (z >= lo) & (z <= hi)
                x, y, z = x[keep], y[keep], z[keep]
                if z.size == 0:
                    continue
            pix = np.stack([x.astype(np.float32), y.astype(np.float32), np.ones_like(x, dtype=np.float32)], axis=0)
            cam = (kinv @ pix) * z[None, :]
            world = (pose[:3, :3].astype(np.float32) @ cam + pose[:3, 3:4].astype(np.float32)).T
            points_all.append(world)
            active.append(fid)

            core_keep = core_depth[y, x]
            core_world = world[core_keep]
            soft_world = world
            core_vox = _points_to_voxels(core_world, voxel_size)
            soft_vox = _points_to_voxels(soft_world, voxel_size)
            for v in core_vox:
                key = (int(v[0]), int(v[1]), int(v[2]))
                core_set.add(key)
                support[key] = support.get(key, 0) + 1
            for idx, v in enumerate(soft_vox):
                key = (int(v[0]), int(v[1]), int(v[2]))
                soft_set.add(key)
                prob_sum[key] = prob_sum.get(key, 0.0) + float(prob_depth[y[idx], x[idx]])

        if not points_all:
            continue
        points = np.concatenate(points_all, axis=0).astype(np.float32)
        center = points.mean(axis=0)
        bmin = points.min(axis=0)
        bmax = points.max(axis=0)
        scale = float(np.linalg.norm(bmax - bmin))
        num_support_views = len(set(active))
        if len(set(active)) < int(fcfg["min_active_frames"]):
            continue
        if points.shape[0] < int(fcfg["min_lifted_points"]):
            continue
        if len(core_set) < int(fcfg["min_occupied_voxels"]):
            continue
        if num_support_views < int(fcfg["min_support_views"]):
            continue

        hyps.append(
            Track3DHypothesis(
                hypothesis_id=hid,
                source_track_id=int(track_id),
                component_id=0,
                active_frame_ids=sorted(set(active)),
                masks_by_frame=masks_by_frame,
                probs_by_frame=probs_by_frame,
                world_voxels_core=core_set,
                world_voxels_soft=soft_set,
                voxel_support_count=support,
                voxel_prob_sum=prob_sum,
                points_world=points,
                center_world=center,
                bbox_min_world=bmin,
                bbox_max_world=bmax,
                scale_world=scale,
                num_lifted_points=int(points.shape[0]),
                num_core_voxels=len(core_set),
                num_soft_voxels=len(soft_set),
                num_support_views=num_support_views,
            )
        )
        hid += 1
    return hyps


def split_hypothesis_by_3d_components(h: Track3DHypothesis, cfg: dict) -> list[Track3DHypothesis]:
    scfg = cfg["sam3d_merge"]["split"]
    if not bool(scfg["enabled"]):
        return [h]
    comps = _connected_components_26(h.world_voxels_core)
    if len(comps) <= 1:
        return [h]
    total = max(len(h.world_voxels_core), 1)
    kept = [c for c in comps if len(c) >= int(scfg["min_component_voxels"]) and (len(c) / total) >= float(scfg["min_component_ratio"])]
    if not kept and bool(scfg.get("keep_largest_if_all_small", True)):
        kept = [max(comps, key=len)]
    if len(kept) <= 1:
        return [h]
    out = []
    voxel_size = float(cfg["sam3d_merge"]["lifting"]["voxel_size_world"])
    for cidx, comp in enumerate(kept):
        soft = {v for v in h.world_voxels_soft if v in comp}
        points = None
        if h.points_world is not None and h.points_world.shape[0] > 0:
            pw = h.points_world
            vox = np.floor(pw / voxel_size).astype(np.int32)
            comp_set = set(comp)
            inside = np.array([(int(v[0]), int(v[1]), int(v[2])) in comp_set for v in vox], dtype=bool)
            points = pw[inside]
        if points is None or points.shape[0] == 0:
            vox_arr = np.array(list(comp), dtype=np.float32)
            points = (vox_arr + 0.5) * voxel_size
        center = points.mean(axis=0)
        bmin = points.min(axis=0)
        bmax = points.max(axis=0)
        scale = float(np.linalg.norm(bmax - bmin))
        out.append(
            Track3DHypothesis(
                hypothesis_id=h.hypothesis_id * 1000 + cidx + 1,
                source_track_id=h.source_track_id,
                component_id=cidx + 1,
                active_frame_ids=h.active_frame_ids,
                masks_by_frame=h.masks_by_frame,
                probs_by_frame=h.probs_by_frame,
                world_voxels_core=set(comp),
                world_voxels_soft=soft,
                voxel_support_count={k: v for k, v in h.voxel_support_count.items() if k in comp},
                voxel_prob_sum={k: v for k, v in h.voxel_prob_sum.items() if k in soft},
                points_world=points,
                center_world=center,
                bbox_min_world=bmin,
                bbox_max_world=bmax,
                scale_world=scale,
                num_lifted_points=int(points.shape[0]),
                num_core_voxels=len(comp),
                num_soft_voxels=len(soft),
                num_support_views=h.num_support_views,
                stats=dict(h.stats),
            )
        )
    return out


def _compute_coview_conflict(h1: Track3DHypothesis, h2: Track3DHypothesis, cfg: dict) -> tuple[int, bool]:
    mcfg = cfg["sam3d_merge"]["merge"]
    shared = sorted(set(h1.active_frame_ids) & set(h2.active_frame_ids))
    conflicts = 0
    for fid in shared:
        m1 = h1.masks_by_frame.get(fid)
        m2 = h2.masks_by_frame.get(fid)
        if m1 is None or m2 is None:
            continue
        if int(m1.sum()) == 0 or int(m2.sum()) == 0:
            continue
        iou = _mask_iou(m1.astype(bool), m2.astype(bool))
        if iou < float(mcfg["coview_conflict_max_mask_iou"]):
            conflicts += 1
    block = conflicts >= int(mcfg["coview_conflict_min_frames"])
    return conflicts, block


def _project_world_to_image(points_world: np.ndarray, pose_world_from_cam: np.ndarray, intr: dict, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    pose_cam_from_world = np.linalg.inv(pose_world_from_cam).astype(np.float32)
    p_h = np.concatenate([points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (pose_cam_from_world @ p_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool)
    cam = cam[valid]
    z = z[valid]
    k = intr["intrinsic_mat"].astype(np.float32)
    u = (k[0, 0] * (cam[:, 0] / z) + k[0, 2]).round().astype(np.int32)
    v = (k[1, 1] * (cam[:, 1] / z) + k[1, 2]).round().astype(np.int32)
    in_im = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    proj = np.zeros((h, w), dtype=bool)
    proj[v[in_im], u[in_im]] = True
    return proj


def _compute_reprojection_consistency(hi: Track3DHypothesis, hj: Track3DHypothesis, geometry_provider: GeometryProvider, cfg: dict) -> tuple[float | None, float | None, float | None]:
    voxel_size = float(cfg["sam3d_merge"]["lifting"]["voxel_size_world"])
    max_points = 5000
    vi = np.array(list(hi.world_voxels_core), dtype=np.float32)
    vj = np.array(list(hj.world_voxels_core), dtype=np.float32)
    if vi.size == 0 or vj.size == 0:
        return None, None, None
    if vi.shape[0] > max_points:
        vi = vi[np.linspace(0, vi.shape[0] - 1, num=max_points, dtype=int)]
    if vj.shape[0] > max_points:
        vj = vj[np.linspace(0, vj.shape[0] - 1, num=max_points, dtype=int)]
    wi = (vi + 0.5) * voxel_size
    wj = (vj + 0.5) * voxel_size

    def dir_score(src_world: np.ndarray, dst: Track3DHypothesis) -> float | None:
        vals = []
        for fid in dst.active_frame_ids:
            dst_mask = dst.masks_by_frame.get(fid)
            dst_prob = dst.probs_by_frame.get(fid)
            if dst_mask is None and dst_prob is None:
                continue
            target = dst_prob if dst_prob is not None else dst_mask.astype(np.float32)
            intr = geometry_provider.get_intrinsics(fid, for_depth=True)
            h, w = int(intr["height"]), int(intr["width"])
            if target.shape[:2] != (h, w):
                target = cv2.resize(target.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            pose = geometry_provider.get_pose_world_from_cam(fid)
            proj = _project_world_to_image(src_world, pose, intr, (h, w))
            if proj.sum() == 0:
                continue
            target_bin = target > float(cfg["sam3d_merge"]["masks"]["soft_prob_thresh"])
            vals.append(_mask_iou(proj, target_bin))
        if not vals:
            return None
        return float(np.mean(vals))

    i2j = dir_score(wi, hj)
    j2i = dir_score(wj, hi)
    if i2j is None or j2i is None:
        mean = i2j if j2i is None else j2i
        if i2j is not None and j2i is not None:
            mean = 0.5 * (i2j + j2i)
        return i2j, j2i, mean
    return i2j, j2i, 0.5 * (i2j + j2i)


def compute_merge_edges(hypotheses: list[Track3DHypothesis], geometry_provider: GeometryProvider, cfg: dict) -> list[MergeEdge]:
    mcfg = cfg["sam3d_merge"]["merge"]
    edges: list[MergeEdge] = []
    eps = 1e-6
    for i in range(len(hypotheses)):
        hi = hypotheses[i]
        mar_i = (hi.bbox_max_world - hi.bbox_min_world) * float(mcfg["candidate_bbox_margin"])
        e_i_min, e_i_max = hi.bbox_min_world - mar_i, hi.bbox_max_world + mar_i
        for j in range(i + 1, len(hypotheses)):
            hj = hypotheses[j]
            mar_j = (hj.bbox_max_world - hj.bbox_min_world) * float(mcfg["candidate_bbox_margin"])
            e_j_min, e_j_max = hj.bbox_min_world - mar_j, hj.bbox_max_world + mar_j
            if not _bbox_intersects(e_i_min, e_i_max, e_j_min, e_j_max):
                continue

            vi = hi.world_voxels_core
            vj = hj.world_voxels_core
            inter = len(vi & vj)
            union = len(vi | vj)
            voxel_iou = float(inter / max(union, 1))
            c_ij = float(inter / max(len(vi), 1))
            c_ji = float(inter / max(len(vj), 1))
            dist = float(np.linalg.norm(hi.center_world - hj.center_world))
            dnorm = dist / max(hi.scale_world, hj.scale_world, eps)
            bb_iou = _bbox_iou_3d(hi.bbox_min_world, hi.bbox_max_world, hj.bbox_min_world, hj.bbox_max_world)

            conflict_frames, raw_block = _compute_coview_conflict(hi, hj, cfg) if bool(mcfg.get("use_coview_conflict_check", True)) else (0, False)
            geometry_match = (voxel_iou > float(mcfg["voxel_iou_thresh"]) or c_ij > float(mcfg["containment_thresh"]) or c_ji > float(mcfg["containment_thresh"]))
            distance_ok = dnorm < float(mcfg["centroid_dist_norm_thresh"])
            support_ok = min(hi.num_support_views, hj.num_support_views) >= int(mcfg["min_pair_support_views"])
            reproj_i_to_j = None
            reproj_j_to_i = None
            reproj_mean = None
            reprojection_ok = True
            if bool(mcfg.get("use_reprojection_check", False)):
                reproj_i_to_j, reproj_j_to_i, reproj_mean = _compute_reprojection_consistency(hi, hj, geometry_provider, cfg)
                if reproj_mean is not None:
                    reprojection_ok = reproj_mean > float(mcfg["reproj_iou_thresh"])

            weak_geom = (
                voxel_iou < float(mcfg.get("coview_block_max_voxel_iou", 0.05))
                and max(c_ij, c_ji) < float(mcfg.get("coview_block_max_containment", 0.20))
                and dnorm > float(mcfg.get("coview_block_min_centroid_dist_norm", 0.20))
            )
            should_block = bool(raw_block and weak_geom)
            should_merge = (not should_block) and geometry_match and distance_ok and support_ok and reprojection_ok
            score = 1.0 * voxel_iou + 0.7 * max(c_ij, c_ji) + 0.5 * (reproj_mean if reproj_mean is not None else 0.0) - 0.5 * dnorm
            edges.append(
                MergeEdge(
                    i=i, j=j,
                    voxel_iou=voxel_iou,
                    containment_i_to_j=c_ij,
                    containment_j_to_i=c_ji,
                    centroid_dist_norm=dnorm,
                    bbox_iou_3d=bb_iou,
                    reproj_i_to_j=reproj_i_to_j,
                    reproj_j_to_i=reproj_j_to_i,
                    reproj_mean=reproj_mean,
                    coview_conflict_frames=conflict_frames,
                    should_block=should_block,
                    should_merge=should_merge,
                    score=float(score),
                )
            )
    return edges


def agglomerative_merge_hypotheses(hypotheses: list[Track3DHypothesis], edges: list[MergeEdge]) -> list[list[int]]:
    parent = list(range(len(hypotheses)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_lookup = {(min(e.i, e.j), max(e.i, e.j)): e for e in edges}
    pos_edges = sorted([e for e in edges if e.should_merge], key=lambda e: e.score, reverse=True)

    def can_merge(ca: list[int], cb: list[int]) -> bool:
        pairs = []
        for a in ca:
            for b in cb:
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                e = edge_lookup.get(key)
                if e is not None:
                    pairs.append(e)
        if not pairs:
            return False
        if any(e.should_block for e in pairs):
            return False
        pos = [e for e in pairs if e.should_merge]
        if not pos:
            return False
        max_score = max(e.score for e in pos)
        avg_score = float(np.mean([e.score for e in pairs])) if pairs else -1.0
        pos_ratio = float(len(pos)) / float(len(pairs))
        if max_score < 0.10:
            return False
        if avg_score < -0.05:
            return False
        if pos_ratio < 0.5:
            return False
        return True

    for e in pos_edges:
        ra, rb = find(e.i), find(e.j)
        if ra == rb:
            continue
        ca = [k for k in range(len(hypotheses)) if find(k) == ra]
        cb = [k for k in range(len(hypotheses)) if find(k) == rb]
        if can_merge(ca, cb):
            union(ra, rb)

    clusters = {}
    for i in range(len(hypotheses)):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return [sorted(v) for _, v in sorted(clusters.items(), key=lambda kv: min(kv[1]))]


def build_consolidated_objects(hypotheses: list[Track3DHypothesis], clusters: list[list[int]]) -> list[Consolidated3DObject]:
    out = []
    for oid, members in enumerate(clusters, start=1):
        mh = [hypotheses[i] for i in members]
        core = set().union(*[h.world_voxels_core for h in mh]) if mh else set()
        soft = set().union(*[h.world_voxels_soft for h in mh]) if mh else set()
        frames = sorted(set().union(*[set(h.active_frame_ids) for h in mh])) if mh else []
        pts = np.concatenate([h.points_world for h in mh if h.points_world is not None and h.points_world.shape[0] > 0], axis=0)
        center = pts.mean(axis=0)
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        scale = float(np.linalg.norm(bmax - bmin))
        conf = min(1.0, len(frames) / 5.0)
        out.append(
            Consolidated3DObject(
                object_id=oid,
                member_hypothesis_ids=[h.hypothesis_id for h in mh],
                member_track_ids=sorted(set(int(h.source_track_id) for h in mh)),
                world_voxels_core=core,
                world_voxels_soft=soft,
                center_world=center,
                bbox_min_world=bmin,
                bbox_max_world=bmax,
                scale_world=scale,
                active_frame_ids=frames,
                confidence=float(conf),
                stats={"num_core_voxels": len(core), "num_support_views": len(frames)},
            )
        )
    return out


def compose_grouped_2d_labels(final_objects: list[Consolidated3DObject], tracks: dict, frame_ids: list[str], image_shape: tuple[int, int], prob_thresh: float = 0.3) -> dict[int, np.ndarray]:
    h, w = image_shape
    frame_ids_norm = [_norm_frame_id(fid) for fid in frame_ids]
    obj_id_imgs = {int(fid): np.zeros((h, w), dtype=np.int32) for fid in frame_ids_norm}
    score_imgs = {int(fid): np.zeros((h, w), dtype=np.float32) for fid in frame_ids_norm}

    for obj in final_objects:
        for tid in obj.member_track_ids:
            tdata = tracks.get(int(tid), {})
            probs = tdata.get("probs_by_frame", {})
            masks = tdata.get("masks_by_frame", {})
            for fid in frame_ids_norm:
                prob = probs.get(fid)
                mask = masks.get(fid)
                if prob is None and mask is None:
                    continue
                if prob is None:
                    prob = mask.astype(np.float32)
                if prob.shape != (h, w):
                    prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                valid = prob >= float(prob_thresh)
                update = valid & (prob > score_imgs[int(fid)])
                score_imgs[int(fid)][update] = prob[update]
                obj_id_imgs[int(fid)][update] = int(obj.object_id)
    return obj_id_imgs


def merge_sam2_tracks_in_3d(tracks: dict, frame_ids: list[str], scene_id: str, cfg: dict, geometry_provider: GeometryProvider, debug_dir: str | None = None) -> list[Consolidated3DObject]:
    hyps = build_track_3d_hypotheses(tracks, frame_ids, geometry_provider, cfg)
    split_hyps = []
    for h in hyps:
        split_hyps.extend(split_hypothesis_by_3d_components(h, cfg))
    for idx, h in enumerate(split_hyps):
        h.hypothesis_id = idx
    edges = compute_merge_edges(split_hyps, geometry_provider, cfg)
    clusters = agglomerative_merge_hypotheses(split_hyps, edges)
    objects = build_consolidated_objects(split_hyps, clusters)

    if debug_dir is not None and bool(cfg["sam3d_merge"]["output"].get("save_debug_json", True)):
        out = {
            "scene_id": scene_id,
            "num_input_tracks": len(tracks),
            "num_lifted_hypotheses": len(hyps),
            "num_after_splitting": len(split_hyps),
            "num_final_objects": len(objects),
            "tracks": [
                {
                    "hypothesis_id": int(h.hypothesis_id),
                    "source_track_id": int(h.source_track_id),
                    "num_active_frames": len(h.active_frame_ids),
                    "num_lifted_points": int(h.num_lifted_points),
                    "num_core_voxels": int(h.num_core_voxels),
                    "num_support_views": int(h.num_support_views),
                    "center_world": [float(x) for x in h.center_world.tolist()],
                    "scale_world": float(h.scale_world),
                }
                for h in split_hyps
            ],
            "edges": [
                {
                    "i": int(e.i),
                    "j": int(e.j),
                    "voxel_iou": float(e.voxel_iou),
                    "containment_i_to_j": float(e.containment_i_to_j),
                    "containment_j_to_i": float(e.containment_j_to_i),
                    "centroid_dist_norm": float(e.centroid_dist_norm),
                    "reproj_mean": None if e.reproj_mean is None else float(e.reproj_mean),
                    "coview_conflict_frames": int(e.coview_conflict_frames),
                    "should_block": bool(e.should_block),
                    "should_merge": bool(e.should_merge),
                    "score": float(e.score),
                }
                for e in edges
            ],
            "objects": [
                {
                    "object_id": int(o.object_id),
                    "member_track_ids": [int(t) for t in o.member_track_ids],
                    "num_core_voxels": int(len(o.world_voxels_core)),
                    "num_support_views": int(len(o.active_frame_ids)),
                    "confidence": float(o.confidence),
                }
                for o in objects
            ],
        }
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(debug_dir) / f"{scene_id}_sam3d_debug.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return objects
