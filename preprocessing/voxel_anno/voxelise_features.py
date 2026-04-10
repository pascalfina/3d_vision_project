import itertools
import logging
import os
import os.path as osp
import gc
import resource
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scan3r
from utils import visualisation as vis

_LOGGER = logging.getLogger(__name__)


def _lookup_mask_frame(mask: Dict, frame_id):
    if not isinstance(mask, dict):
        return None

    candidates = []
    if isinstance(frame_id, str):
        candidates.append(frame_id)
        if frame_id.isdigit():
            frame_idx = int(frame_id)
            candidates.extend([frame_idx, f"{frame_idx:06d}", str(frame_idx)])
    else:
        candidates.append(frame_id)
        try:
            frame_idx = int(frame_id)
        except (TypeError, ValueError):
            frame_idx = None
        if frame_idx is not None:
            candidates.extend([f"{frame_idx:06d}", str(frame_idx)])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in mask:
            return mask[candidate]
    return None


def _load_dino_model(model_name: str):
    local_hub_dir = os.getenv("OBJECTX_DINOV2_HUB_DIR")
    if not local_hub_dir:
        torch_home = os.getenv("TORCH_HOME")
        if torch_home:
            local_hub_dir = osp.join(torch_home, "hub", "facebookresearch_dinov2_main")
    if not local_hub_dir:
        cache_root = os.getenv("OBJECTX_CACHE_ROOT")
        if cache_root:
            local_hub_dir = osp.join(
                cache_root, "torch", "hub", "facebookresearch_dinov2_main"
            )
    if not local_hub_dir:
        fallback_cache = "/work/scratch/pafina/objectx-cache/torch/hub/facebookresearch_dinov2_main"
        if osp.isdir(fallback_cache):
            local_hub_dir = fallback_cache
    if not local_hub_dir:
        local_hub_dir = osp.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if osp.isdir(local_hub_dir):
        _LOGGER.info("Loading DINOv2 from local hub cache: %s", local_hub_dir)
        return torch.hub.load(local_hub_dir, model_name, source="local")

    _LOGGER.info("Loading DINOv2 from torch hub repo")
    return torch.hub.load("facebookresearch/dinov2", model_name)


def _get_dino_embedding(images: torch.Tensor) -> torch.Tensor:
    images = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
    inputs = transform(images).cuda()
    outputs = model(inputs, is_training=True)

    n_patch = 518 // 14
    bs = images.shape[0]
    patch_embeddings = (
        outputs["x_prenorm"][:, model.num_register_tokens + 1 :]
        .permute(0, 2, 1)
        .reshape(bs, 1024, n_patch, n_patch)
    )
    return patch_embeddings


def _log_rss(prefix: str) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    _LOGGER.info("%s | max_rss_mb=%.1f", prefix, rss_mb)


def _save_featured_voxel(
    voxel: torch.Tensor, output_file: str = "voxel_output_dense.npz"
):
    # Keep float16 storage for space, but avoid compressed npz because the
    # compression step can cause large transient RAM spikes on the cluster.
    voxel_np = voxel.half().cpu().numpy()
    output_dir = osp.dirname(output_file)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_voxel_", suffix=".npz", dir=output_dir
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, arr_0=voxel_np)
        os.replace(tmp_path, output_file)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)
        del voxel_np
    _LOGGER.info(f"Voxel saved to {output_file}")


def _project_to_image(
    voxel: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    grid_size: tuple[int] = (64, 64, 64),
):
    voxel_size = 1.0 / grid_size[0]
    voxel = voxel.float() * voxel_size
    assert voxel.min() >= 0.0 and voxel.max() <= 1.0

    voxel = voxel * 2.0 - 1.0
    assert voxel.min() >= -1.0 and voxel.max() <= 1.0
    voxel = voxel * scale + mean
    uv, linear_depth = utils3d.torch.project_cv(
        voxel.float(), extrinsics.float(), intrinsics.float()
    )
    return uv, linear_depth


def _load_depth_shift(data_dir: str, scan_id: str) -> float:
    info_path = osp.join(data_dir, scan_id, "sequence", "_info.txt")
    with open(info_path) as f:
        for line in f:
            if line.startswith("m_depthShift"):
                return float(line.split("=", 1)[1].strip())
    return 1000.0


def _compute_voxel_observations(
    projection_color: torch.Tensor,
    projection_depth: torch.Tensor,
    linear_depth: torch.Tensor,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    color_size: Tuple[float, float],
    depth_size: Tuple[float, float],
    depth_abs_tol: float,
    depth_rel_tol: float,
) -> np.ndarray:
    color_uv = projection_color.cpu().numpy()
    depth_uv = projection_depth.cpu().numpy()
    linear_depth = linear_depth.cpu().numpy()
    masks = np.stack(selected_masks).astype(np.bool_)
    depth_maps = np.stack(selected_depths).astype(np.float32)

    color_width, color_height = color_size
    depth_width, depth_height = depth_size
    num_views = color_uv.shape[0]
    view_idx = np.arange(num_views)[:, None]

    color_x = np.rint(color_uv[..., 0]).astype(np.int64)
    color_y = np.rint(color_uv[..., 1]).astype(np.int64)
    color_in_bounds = (
        (color_x >= 0)
        & (color_x < int(color_width))
        & (color_y >= 0)
        & (color_y < int(color_height))
    )
    color_x_clip = np.clip(color_x, 0, int(color_width) - 1)
    color_y_clip = np.clip(color_y, 0, int(color_height) - 1)
    color_hits = masks[view_idx, color_y_clip, color_x_clip]
    color_visible = color_in_bounds & color_hits

    depth_x = np.rint(depth_uv[..., 0]).astype(np.int64)
    depth_y = np.rint(depth_uv[..., 1]).astype(np.int64)
    depth_in_bounds = (
        (depth_x >= 0)
        & (depth_x < int(depth_width))
        & (depth_y >= 0)
        & (depth_y < int(depth_height))
    )
    depth_x_clip = np.clip(depth_x, 0, int(depth_width) - 1)
    depth_y_clip = np.clip(depth_y, 0, int(depth_height) - 1)
    sampled_depth = depth_maps[view_idx, depth_y_clip, depth_x_clip]
    positive_depth = sampled_depth > 0.0
    depth_tolerance = np.maximum(depth_abs_tol, depth_rel_tol * sampled_depth)
    depth_visible = (
        depth_in_bounds
        & positive_depth
        & (np.abs(sampled_depth - linear_depth) <= depth_tolerance)
    )

    return color_visible & depth_visible


def _segment_mesh(
    mesh: o3d.geometry.TriangleMesh, annos: np.ndarray, obj_id: int, scan_id: str
):
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertex_mask = annos == obj_id
    selected_vertices = np.where(vertex_mask)[0]
    index_map = {
        old_idx: dense_idx for dense_idx, old_idx in enumerate(selected_vertices)
    }

    # Filter faces that only contain selected vertices
    face_mask = np.all(np.isin(faces, selected_vertices), axis=1)
    selected_faces = faces[face_mask]
    reindexed_faces = np.vectorize(index_map.get)(selected_faces)

    # Create the segmented mesh
    segmented_mesh = o3d.geometry.TriangleMesh()
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices[selected_vertices])
    segmented_mesh.triangles = o3d.utility.Vector3iVector(reindexed_faces)
    if args.visualize:
        o3d.io.write_triangle_mesh(
            f"vis/{scan_id}_{obj_id}_no_scale_segmented_mesh.ply", segmented_mesh
        )
    return segmented_mesh


def _dilate_voxels(voxel_grid: o3d.geometry.VoxelGrid) -> np.ndarray:
    voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # densify voxel grid
    dilated_voxels = set()
    directions = [d for d in itertools.product([-1, 0, 1], repeat=3) if d != (0, 0, 0)]
    for v in voxel_grid:
        dilated_voxels.add(tuple(v))
        for d in directions:
            neighbor = tuple(v + np.array(d))
            if all(0 <= n < 64 for n in neighbor):
                dilated_voxels.add(neighbor)
    voxel_grid = np.array(list(set(dilated_voxels)))
    return voxel_grid


def _voxelize_normalized_points(
    normalized_points: np.ndarray, dilate_iters: int = 1
) -> np.ndarray:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        normalized_points.astype(np.float64)
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        point_cloud,
        1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    if voxel_indices.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    voxel_indices = np.unique(voxel_indices.astype(np.int32), axis=0)
    for _ in range(max(0, dilate_iters)):
        voxel_centers = ((voxel_indices.astype(np.float32) + 0.5) / 64.0) - 0.5
        tmp_point_cloud = o3d.geometry.PointCloud()
        tmp_point_cloud.points = o3d.utility.Vector3dVector(
            voxel_centers.astype(np.float64)
        )
        tmp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            tmp_point_cloud,
            1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        voxel_indices = _dilate_voxels(tmp_voxel_grid)
    return voxel_indices


def _clean_object_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.astype(bool)
    if cleaned.sum() == 0:
        return cleaned.astype(np.uint8)

    keep_largest = os.getenv("OBJECTX_VOXEL_MASK_KEEP_LARGEST", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if keep_largest:
        labeled, num_labels = ndimage.label(cleaned)
        if num_labels > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            cleaned = labeled == counts.argmax()

    erode_iters = int(os.getenv("OBJECTX_VOXEL_MASK_ERODE_ITERS", "0"))
    if erode_iters > 0 and cleaned.sum() > 0:
        eroded = ndimage.binary_erosion(cleaned, iterations=erode_iters)
        if eroded.sum() >= max(16, cleaned.sum() // 10):
            cleaned = eroded

    min_area = int(os.getenv("OBJECTX_VOXEL_MASK_MIN_AREA", "0"))
    if cleaned.sum() < max(0, min_area):
        return np.zeros_like(mask, dtype=np.uint8)
    return cleaned.astype(np.uint8)


def _filter_selected_masks(
    selected_frame_ids: list[str], selected_masks: list[np.ndarray], object_source: str
) -> tuple[list[str], list[np.ndarray]]:
    if object_source not in {"lifted_masks", "tsdf_masks", "hybrid_masks"}:
        return selected_frame_ids, selected_masks

    clean_masks = os.getenv("OBJECTX_VOXEL_CLEAN_MASKS", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if not clean_masks:
        return selected_frame_ids, selected_masks

    filtered_ids = []
    filtered_masks = []
    for frame_id, obj_mask in zip(selected_frame_ids, selected_masks):
        cleaned = _clean_object_mask(obj_mask)
        if cleaned.sum() > 0:
            filtered_ids.append(frame_id)
            filtered_masks.append(cleaned)
    return filtered_ids, filtered_masks


def _resolve_frame_selection_mode() -> str:
    source = (os.getenv("OBJECTX_VOXEL_FRAME_SELECTION") or "all").strip().lower()
    aliases = {
        "all": "all",
        "legacy": "all",
        "first": "all",
        "top_area": "top_area",
        "mask_area": "top_area",
        "area": "top_area",
        "diverse": "diverse_area",
        "diverse_area": "diverse_area",
    }
    return aliases.get(source, "all")


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def _camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    camera_to_world = np.linalg.inv(extrinsic)
    return camera_to_world[:3, 3].astype(np.float32)


def _select_object_frames(
    frame_ids: list[str],
    masks: list[np.ndarray],
    extrinsics: dict[str, np.ndarray],
    max_views: int,
) -> tuple[list[str], list[np.ndarray]]:
    if max_views <= 0 or len(frame_ids) <= max_views:
        return frame_ids, masks

    mode = _resolve_frame_selection_mode()
    if mode == "all":
        keep = np.arange(min(max_views, len(frame_ids)), dtype=np.int64)
    else:
        areas = np.array([mask.sum() for mask in masks], dtype=np.float32)
        order = np.argsort(-areas, kind="stable")

        if mode == "top_area":
            keep = order[:max_views]
        else:
            centers = np.stack(
                [_camera_center_from_extrinsic(extrinsics[frame_id]) for frame_id in frame_ids],
                axis=0,
            )
            area_scores = _normalize_scores(areas)
            pairwise = np.linalg.norm(
                centers[:, None, :] - centers[None, :, :], axis=-1
            )
            max_pairwise = float(pairwise.max()) if pairwise.size > 0 else 1.0
            area_weight = float(os.getenv("OBJECTX_VOXEL_FRAME_AREA_WEIGHT", "0.35"))
            diversity_weight = float(
                os.getenv("OBJECTX_VOXEL_FRAME_DIVERSITY_WEIGHT", "0.65")
            )

            selected = [int(order[0])]
            remaining = [int(idx) for idx in order[1:]]
            while remaining and len(selected) < max_views:
                selected_centers = centers[np.array(selected, dtype=np.int64)]
                candidate_centers = centers[np.array(remaining, dtype=np.int64)]
                min_dist = np.min(
                    np.linalg.norm(
                        candidate_centers[:, None, :] - selected_centers[None, :, :],
                        axis=-1,
                    ),
                    axis=1,
                )
                dist_scores = min_dist / max(max_pairwise, 1e-6)
                rem_area = area_scores[np.array(remaining, dtype=np.int64)]
                scores = area_weight * rem_area + diversity_weight * dist_scores
                best_pos = int(np.argmax(scores))
                selected.append(remaining.pop(best_pos))
            keep = np.array(selected, dtype=np.int64)

    keep = np.sort(keep.astype(np.int64))
    selected_frame_ids = [frame_ids[i] for i in keep.tolist()]
    selected_masks = [masks[i] for i in keep.tolist()]
    selected_areas = [int(masks[i].sum()) for i in keep.tolist()]
    preview = ",".join(selected_frame_ids[: min(5, len(selected_frame_ids))])
    _LOGGER.info(
        "[2.5] frame_selection mode=%s kept=%s/%s area[min=%s mean=%.1f max=%s] ids=%s",
        mode,
        len(selected_frame_ids),
        len(frame_ids),
        int(np.min(selected_areas)) if selected_areas else 0,
        float(np.mean(selected_areas)) if selected_areas else 0.0,
        int(np.max(selected_areas)) if selected_areas else 0,
        preview,
    )
    return selected_frame_ids, selected_masks


def _keep_largest_voxel_component(voxel_indices: np.ndarray) -> np.ndarray:
    if voxel_indices.size == 0:
        return voxel_indices
    keep_largest = os.getenv(
        "OBJECTX_VOXEL_KEEP_LARGEST_COMPONENT", "1"
    ).lower() not in {"0", "false", "no", "off", ""}
    if not keep_largest:
        return voxel_indices

    occupancy = np.zeros((64, 64, 64), dtype=np.uint8)
    occupancy[
        voxel_indices[:, 0].astype(np.int32),
        voxel_indices[:, 1].astype(np.int32),
        voxel_indices[:, 2].astype(np.int32),
    ] = 1
    labeled, num_labels = ndimage.label(occupancy, structure=np.ones((3, 3, 3)))
    if num_labels <= 1:
        return voxel_indices

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    kept = np.argwhere(labeled == largest_label).astype(np.int32)
    return kept


def _normalize_segmented_mesh(segmented_mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(segmented_mesh.vertices)
    mean = vertices.mean(axis=0)
    vertices -= mean
    scale = np.max(np.abs(vertices))
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mean, scale


def _resolve_object_source() -> str:
    source = (
        getattr(args, "object_source", None)
        or os.getenv("OBJECTX_VOXEL_OBJECT_SOURCE")
        or "gt_mesh"
    ).strip()
    aliases = {
        "gt": "gt_mesh",
        "mesh": "gt_mesh",
        "gt_mesh": "gt_mesh",
        "mask": "lifted_masks",
        "masks": "lifted_masks",
        "lifted": "lifted_masks",
        "lifted_masks": "lifted_masks",
        "tsdf": "tsdf_masks",
        "fused": "tsdf_masks",
        "fused_masks": "tsdf_masks",
        "tsdf_masks": "tsdf_masks",
        "hybrid": "hybrid_masks",
        "hybrid_masks": "hybrid_masks",
        "union": "hybrid_masks",
    }
    return aliases.get(source, source)


def _resolve_pose_camera_to_world(
    extrinsics: Dict[str, np.ndarray], frame_ids: list[str]
) -> list[np.ndarray]:
    mode = (os.getenv("OBJECTX_VOXEL_POSE_MODE") or "raw").strip().lower()
    aliases = {
        "raw": "raw",
        "direct": "raw",
        "camera_to_world": "raw",
        "cam2world": "raw",
        "invert": "invert",
        "inverse": "invert",
        "world_to_camera": "invert",
        "world2cam": "invert",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"raw", "invert"}:
        raise ValueError(
            f"Unsupported OBJECTX_VOXEL_POSE_MODE={mode!r}; expected 'raw' or 'invert'"
        )
    if mode == "raw":
        return [np.array(extrinsics[frame_id], copy=True) for frame_id in frame_ids]
    return [np.linalg.inv(extrinsics[frame_id]) for frame_id in frame_ids]


def _invert_pose_list(poses: list[np.ndarray]) -> list[np.ndarray]:
    return [np.linalg.inv(pose).astype(np.float32) for pose in poses]


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    robust = os.getenv("OBJECTX_VOXEL_LIFT_ROBUST_NORMALIZE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if robust and points.shape[0] >= 32:
        center = np.median(points, axis=0)
        centered = points - center
        scale = float(np.percentile(np.abs(centered), 99.0, axis=0).max())
    else:
        center = points.mean(axis=0)
        centered = points - center
        scale = float(np.max(np.abs(centered)))
    if scale <= 1e-8:
        scale = 1.0
    normalized = centered * (1.0 / (2.0 * scale))
    normalized = np.clip(normalized, -0.5 + 1e-6, 0.5 - 1e-6)
    return normalized, center.astype(np.float32), float(scale)


def _filter_lifted_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 64:
        return points

    filtered = points
    lower_q = float(os.getenv("OBJECTX_VOXEL_LIFT_TRIM_LOW_Q", "1.0"))
    upper_q = float(os.getenv("OBJECTX_VOXEL_LIFT_TRIM_HIGH_Q", "99.0"))
    if 0.0 <= lower_q < upper_q <= 100.0:
        lower = np.percentile(filtered, lower_q, axis=0)
        upper = np.percentile(filtered, upper_q, axis=0)
        keep = np.all((filtered >= lower) & (filtered <= upper), axis=1)
        if keep.sum() >= max(32, int(0.2 * filtered.shape[0])):
            filtered = filtered[keep]

    if filtered.shape[0] >= 64:
        center = np.median(filtered, axis=0)
        distances = np.linalg.norm(filtered - center[None, :], axis=1)
        keep_q = float(os.getenv("OBJECTX_VOXEL_LIFT_KEEP_RADIUS_Q", "99.0"))
        keep_radius = np.percentile(distances, keep_q)
        keep = distances <= keep_radius
        if keep.sum() >= max(32, int(0.2 * filtered.shape[0])):
            filtered = filtered[keep]

    return filtered.astype(np.float32, copy=False)


def _normalize_points_with_reference(
    points: np.ndarray, scan_id: str, obj_id: int
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    reference_root = os.getenv("OBJECTX_VOXEL_REFERENCE_MEAN_SCALE_ROOT", "").strip()
    if not reference_root:
        return None

    reference_path = osp.join(
        reference_root,
        "files",
        "gs_annotations",
        scan_id,
        str(obj_id),
        "mean_scale_dense.npz",
    )
    if not osp.exists(reference_path):
        _LOGGER.warning(
            "[2.5] reference mean/scale missing for %s/%s at %s",
            scan_id,
            obj_id,
            reference_path,
        )
        return None

    reference = np.load(reference_path)
    mean = reference["mean"].astype(np.float32)
    scale = float(reference["scale"])
    if scale <= 1e-8:
        _LOGGER.warning(
            "[2.5] invalid reference scale for %s/%s at %s",
            scan_id,
            obj_id,
            reference_path,
        )
        return None

    normalized = (points - mean[None, :]) * (1.0 / (2.0 * scale))
    normalized = np.clip(normalized, -0.5 + 1e-6, 0.5 - 1e-6)
    _LOGGER.info(
        "[2.5] using reference mean/scale for %s/%s from %s",
        scan_id,
        obj_id,
        reference_root,
    )
    return normalized.astype(np.float32), mean, scale


def _save_point_cloud(points: np.ndarray, output_file: str) -> None:
    if points.size == 0:
        return
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(output_file, point_cloud)


def _save_triangle_mesh(mesh: o3d.geometry.TriangleMesh, output_file: str) -> None:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return
    o3d.io.write_triangle_mesh(output_file, mesh)


def _lift_masked_points(
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
) -> np.ndarray:
    if len(selected_masks) != len(selected_depths) or len(selected_masks) != len(
        pose_camera_to_world
    ):
        raise ValueError("Mismatched mask/depth/pose lengths for lifted object path")

    depth_width = int(depth_intrinsics["width"])
    depth_height = int(depth_intrinsics["height"])
    intrinsic = depth_intrinsics["intrinsic_mat"].astype(np.float32)
    intrinsic_inv = np.linalg.inv(intrinsic).astype(np.float32)
    pixel_stride = max(1, int(os.getenv("OBJECTX_VOXEL_LIFT_PIXEL_STRIDE", "1")))
    coord_system = os.getenv("OBJECTX_VOXEL_LIFT_COORD_SYSTEM", "pinhole").strip().lower()

    lifted_points = []
    for obj_mask, depth_map, camera_to_world in zip(
        selected_masks, selected_depths, pose_camera_to_world
    ):
        mask_depth = np.array(
            Image.fromarray((obj_mask > 0).astype(np.uint8)).resize(
                (depth_width, depth_height), resample=Image.NEAREST
            ),
            dtype=bool,
        )
        y, x = np.nonzero(mask_depth)
        if x.size == 0:
            continue
        depth = depth_map[y, x]
        valid = depth > 0.0
        if not np.any(valid):
            continue
        x = x[valid]
        y = y[valid]
        depth = depth[valid].astype(np.float32)

        if pixel_stride > 1:
            x = x[::pixel_stride]
            y = y[::pixel_stride]
            depth = depth[::pixel_stride]

        pixels = np.stack(
            [
                x.astype(np.float32),
                y.astype(np.float32),
                np.ones_like(x, dtype=np.float32),
            ],
            axis=0,
        )
        if coord_system in {"scan3r", "kitti"}:
            x3 = (x.astype(np.float32) - intrinsic[0, 2]) * depth / intrinsic[0, 0]
            y3 = (y.astype(np.float32) - intrinsic[1, 2]) * depth / intrinsic[1, 1]
            cam_points = np.stack([depth, -x3, -y3], axis=0)
        else:
            cam_points = (intrinsic_inv @ pixels) * depth[None, :]
        world_points = (
            camera_to_world[:3, :3].astype(np.float32) @ cam_points
            + camera_to_world[:3, 3:4].astype(np.float32)
        )
        lifted_points.append(world_points.T)

    if not lifted_points:
        return np.zeros((0, 3), dtype=np.float32)
    lifted_points = np.concatenate(lifted_points, axis=0).astype(np.float32)
    filtered_points = _filter_lifted_points(lifted_points)
    _LOGGER.info(
        "[2.5] lifted points filtered %s -> %s using coord_system=%s",
        int(lifted_points.shape[0]),
        int(filtered_points.shape[0]),
        coord_system,
    )
    return filtered_points


def _prepare_lifted_geometry(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lifted_points = _lift_masked_points(
        selected_masks=selected_masks,
        selected_depths=selected_depths,
        pose_camera_to_world=pose_camera_to_world,
        depth_intrinsics=depth_intrinsics,
    )
    _LOGGER.info(
        "[2.5] object %s/%s lifted_points=%s",
        scan_id,
        obj_id,
        int(lifted_points.shape[0]),
    )
    if lifted_points.shape[0] == 0:
        raise ValueError("No valid 3D points after lifting masked depth")
    min_lifted_points = int(os.getenv("OBJECTX_VOXEL_MIN_LIFTED_POINTS", "0"))
    if lifted_points.shape[0] < max(0, min_lifted_points):
        raise ValueError(
            f"Too few lifted points ({lifted_points.shape[0]}) for object quality threshold"
        )

    normalized_with_reference = _normalize_points_with_reference(
        lifted_points, scan_id=scan_id, obj_id=obj_id
    )
    if normalized_with_reference is None:
        normalized_points, mean, scale = _normalize_points(lifted_points)
    else:
        normalized_points, mean, scale = normalized_with_reference
    return lifted_points, normalized_points, mean, scale


def _finalize_object_voxel_grid(
    voxel_grid: np.ndarray,
    *,
    scan_id: str,
    obj_id: int,
    variant: str,
) -> np.ndarray:
    voxel_grid = np.unique(voxel_grid.astype(np.int32), axis=0)
    voxel_grid = _keep_largest_voxel_component(voxel_grid)
    if voxel_grid.size == 0:
        raise ValueError(f"No voxels created from {variant}")
    min_voxels = int(os.getenv("OBJECTX_VOXEL_MIN_VOXELS", "0"))
    if voxel_grid.shape[0] < max(0, min_voxels):
        raise ValueError(
            f"Too few {variant} voxels ({voxel_grid.shape[0]}) for object quality threshold"
        )
    _LOGGER.info(
        "[2.5] object %s/%s %s_voxels=%s",
        scan_id,
        obj_id,
        variant,
        int(voxel_grid.shape[0]),
    )
    return voxel_grid


def _build_lifted_object_voxel_grid(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    lifted_points, normalized_points, mean, scale = _prepare_lifted_geometry(
        scan_id=scan_id,
        obj_id=obj_id,
        selected_masks=selected_masks,
        selected_depths=selected_depths,
        pose_camera_to_world=pose_camera_to_world,
        depth_intrinsics=depth_intrinsics,
    )
    voxel_grid = _voxelize_normalized_points(normalized_points, dilate_iters=1)
    voxel_grid = _finalize_object_voxel_grid(
        voxel_grid,
        scan_id=scan_id,
        obj_id=obj_id,
        variant="lifted",
    )

    if args.visualize or os.getenv("OBJECTX_VOXEL_WRITE_LIFTED_PLY", "0") == "1":
        _save_point_cloud(lifted_points, f"vis/{scan_id}_{obj_id}_lifted_points_world.ply")
        _save_point_cloud(
            normalized_points,
            f"vis/{scan_id}_{obj_id}_lifted_points_normalized.ply",
        )

    return voxel_grid, mean, scale


def _build_tsdf_object_voxel_grid(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    lifted_points, normalized_points, mean, scale = _prepare_lifted_geometry(
        scan_id=scan_id,
        obj_id=obj_id,
        selected_masks=selected_masks,
        selected_depths=selected_depths,
        pose_camera_to_world=pose_camera_to_world,
        depth_intrinsics=depth_intrinsics,
    )
    _LOGGER.info(
        "[2.5] object %s/%s lifted_points=%s before TSDF",
        scan_id,
        obj_id,
        int(lifted_points.shape[0]),
    )
    fallback_to_lifted = os.getenv(
        "OBJECTX_VOXEL_TSDF_FALLBACK_TO_LIFTED", "1"
    ).lower() not in {"0", "false", "no", "off", ""}

    def _fallback_to_lifted(reason: str) -> tuple[np.ndarray, np.ndarray, float]:
        _LOGGER.warning(
            "[2.5] object %s/%s TSDF fallback to lifted-point voxelization: %s",
            scan_id,
            obj_id,
            reason,
        )
        voxel_grid = _voxelize_normalized_points(normalized_points, dilate_iters=1)
        voxel_grid = _finalize_object_voxel_grid(
            voxel_grid,
            scan_id=scan_id,
            obj_id=obj_id,
            variant="fallback_lifted",
        )
        if args.visualize or os.getenv("OBJECTX_VOXEL_WRITE_LIFTED_PLY", "0") == "1":
            _save_point_cloud(
                lifted_points, f"vis/{scan_id}_{obj_id}_lifted_points_world.ply"
            )
            _save_point_cloud(
                normalized_points,
                f"vis/{scan_id}_{obj_id}_lifted_points_normalized.ply",
            )
        return voxel_grid, mean, scale

    tsdf_resolution = int(os.getenv("OBJECTX_VOXEL_TSDF_RESOLUTION", "96"))
    tsdf_sdf_trunc = float(
        os.getenv("OBJECTX_VOXEL_TSDF_SDF_TRUNC", str(3.0 / tsdf_resolution))
    )
    tsdf_depth_trunc = float(os.getenv("OBJECTX_VOXEL_TSDF_DEPTH_TRUNC", "10.0"))
    world_from_object = np.eye(4, dtype=np.float32)
    world_from_object[:3, :3] *= 2.0 * scale
    world_from_object[:3, 3] = mean.astype(np.float32)

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=1.0,
        resolution=tsdf_resolution,
        sdf_trunc=tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        origin=np.array([-0.5, -0.5, -0.5], dtype=np.float64),
    )
    intrinsic = depth_intrinsics["intrinsic_mat"].astype(np.float64)
    depth_width = int(depth_intrinsics["width"])
    depth_height = int(depth_intrinsics["height"])
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        depth_width,
        depth_height,
        intrinsic[0, 0],
        intrinsic[1, 1],
        intrinsic[0, 2],
        intrinsic[1, 2],
    )
    zero_color = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)

    integrated_frames = 0
    for obj_mask, depth_map, camera_to_world in zip(
        selected_masks, selected_depths, pose_camera_to_world
    ):
        mask_depth = np.array(
            Image.fromarray((obj_mask > 0).astype(np.uint8)).resize(
                (depth_width, depth_height), resample=Image.NEAREST
            ),
            dtype=bool,
        )
        if not np.any(mask_depth):
            continue
        masked_depth = np.where(mask_depth, depth_map.astype(np.float32), 0.0)
        if float(masked_depth.max()) <= 0.0:
            continue

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(zero_color),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0,
            depth_trunc=tsdf_depth_trunc,
            convert_rgb_to_intensity=False,
        )
        object_to_camera = np.linalg.inv(camera_to_world).astype(np.float64) @ world_from_object.astype(np.float64)
        volume.integrate(rgbd, intrinsic_o3d, object_to_camera)
        integrated_frames += 1

    _LOGGER.info(
        "[2.5] object %s/%s integrated_frames=%s in TSDF",
        scan_id,
        obj_id,
        integrated_frames,
    )
    if integrated_frames == 0:
        if fallback_to_lifted:
            return _fallback_to_lifted("No valid RGBD frames could be integrated into TSDF")
        raise ValueError("No valid RGBD frames could be integrated into TSDF")

    tsdf_mesh = volume.extract_triangle_mesh()
    tsdf_mesh.compute_vertex_normals()
    tsdf_mesh.remove_duplicated_vertices()
    tsdf_mesh.remove_duplicated_triangles()
    tsdf_mesh.remove_unreferenced_vertices()
    tsdf_mesh.remove_degenerate_triangles()
    tsdf_mesh.remove_non_manifold_edges()

    if len(tsdf_mesh.vertices) == 0 or len(tsdf_mesh.triangles) == 0:
        if not fallback_to_lifted:
            raise ValueError("TSDF fusion produced an empty mesh")
        return _fallback_to_lifted("TSDF fusion produced an empty mesh")

    if args.visualize or os.getenv("OBJECTX_VOXEL_WRITE_LIFTED_PLY", "0") == "1":
        _save_point_cloud(lifted_points, f"vis/{scan_id}_{obj_id}_lifted_points_world.ply")
        _save_point_cloud(
            normalized_points,
            f"vis/{scan_id}_{obj_id}_lifted_points_normalized.ply",
        )
        _save_triangle_mesh(tsdf_mesh, f"vis/{scan_id}_{obj_id}_tsdf_mesh.ply")

    voxel_grid_o3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        tsdf_mesh,
        1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    tsdf_dilate_iters = int(os.getenv("OBJECTX_VOXEL_TSDF_DILATE_ITERS", "0"))
    voxel_grid = np.array([voxel.grid_index for voxel in voxel_grid_o3d.get_voxels()])
    for _ in range(max(0, tsdf_dilate_iters)):
        voxel_centers = ((voxel_grid.astype(np.float32) + 0.5) / 64.0) - 0.5
        tmp_point_cloud = o3d.geometry.PointCloud()
        tmp_point_cloud.points = o3d.utility.Vector3dVector(
            voxel_centers.astype(np.float64)
        )
        tmp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            tmp_point_cloud,
            1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        voxel_grid = _dilate_voxels(tmp_voxel_grid)
    try:
        voxel_grid = _finalize_object_voxel_grid(
            voxel_grid,
            scan_id=scan_id,
            obj_id=obj_id,
            variant="tsdf",
        )
    except ValueError as exc:
        if not fallback_to_lifted:
            raise
        return _fallback_to_lifted(str(exc))

    return voxel_grid, mean, scale


def _build_hybrid_object_voxel_grid(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    lifted_voxel_grid, mean, scale = _build_lifted_object_voxel_grid(
        scan_id=scan_id,
        obj_id=obj_id,
        selected_masks=selected_masks,
        selected_depths=selected_depths,
        pose_camera_to_world=pose_camera_to_world,
        depth_intrinsics=depth_intrinsics,
    )
    tsdf_voxel_grid, _, _ = _build_tsdf_object_voxel_grid(
        scan_id=scan_id,
        obj_id=obj_id,
        selected_masks=selected_masks,
        selected_depths=selected_depths,
        pose_camera_to_world=pose_camera_to_world,
        depth_intrinsics=depth_intrinsics,
    )
    merged = np.concatenate([lifted_voxel_grid, tsdf_voxel_grid], axis=0)
    merged = _finalize_object_voxel_grid(
        merged,
        scan_id=scan_id,
        obj_id=obj_id,
        variant="hybrid",
    )
    _LOGGER.info(
        "[2.5] object %s/%s hybrid_union lifted=%s tsdf=%s merged=%s",
        scan_id,
        obj_id,
        int(lifted_voxel_grid.shape[0]),
        int(tsdf_voxel_grid.shape[0]),
        int(merged.shape[0]),
    )
    return merged, mean, scale


@torch.no_grad()
def voxelise_features(
    obj_data: Dict[str, str],
    scan_id: str,
    mode: str = "gs_annotations",
) -> None:
    """
    Voxelise features for scan.

    Args:
        obj_data (Dict[str, str]): Object data.
        scan_id (str): Scan ID.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
    """

    scenes_dir = osp.join(root_dir, "scenes")
    frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
    extrinsics = scan3r.load_frame_poses(
        data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs
    )
    intrinsics = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)
    filter_unobserved = os.getenv("OBJECTX_VOXEL_FILTER_UNOBSERVED", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    object_source = _resolve_object_source()
    depth_intrinsics = None
    depth_shift = None
    depth_abs_tol = float(os.getenv("OBJECTX_VOXEL_DEPTH_ABS_TOL", "0.05"))
    depth_rel_tol = float(os.getenv("OBJECTX_VOXEL_DEPTH_REL_TOL", "0.02"))
    depth_map_cache = {}
    requires_depth = filter_unobserved or object_source in {
        "lifted_masks",
        "tsdf_masks",
        "hybrid_masks",
    }
    if requires_depth:
        depth_intrinsics = scan3r.load_intrinsics(
            data_dir=scenes_dir, scan_id=scan_id, type="depth"
        )
        depth_shift = _load_depth_shift(scenes_dir, scan_id)
    mask_source = scan3r.resolve_mask_source(getattr(args, "mask_source", None))
    _LOGGER.info("[2.5] using mask source: %s", mask_source)
    _LOGGER.info("[2.5] using object source: %s", object_source)
    mask = scan3r.load_masks(
        data_dir=root_dir, scan_id=scan_id, mask_source=mask_source
    )
    mesh = None
    annos = None
    if object_source == "gt_mesh":
        mesh = scan3r.load_ply_mesh(
            data_dir=scenes_dir,
            scan_id=scan_id,
            label_file_name="labels.instances.annotated.v2.ply",
        )
        annos = scan3r.load_ply_data(
            data_dir=scenes_dir,
            scan_id=scan_id,
            label_file_name="labels.instances.annotated.v2.ply",
        )["vertex"]["objectId"]
    max_views = int(os.getenv("OBJECTX_VOXEL_MAX_VIEWS", "150"))
    _log_rss(f"[2.5] loaded scan {scan_id} with {len(frame_idxs)} frames")

    for obj in obj_data["objects"]:
        try:
            voxel_path = osp.join(
                args.model_dir,
                "files",
                mode,
                scan_id,
                str(obj["id"]),
                "voxel_output_dense.npz",
            )
            mean_scale_path = osp.join(
                args.model_dir,
                "files",
                mode,
                scan_id,
                str(obj["id"]),
                "mean_scale_dense.npz",
            )

            os.makedirs(osp.dirname(voxel_path), exist_ok=True)
            os.makedirs(osp.dirname(mean_scale_path), exist_ok=True)

            if (
                osp.exists(mean_scale_path)
                and osp.exists(voxel_path)
                and "arr_0" in np.load(voxel_path)
                and not args.override
            ):
                _LOGGER.info(f"Skipping {scan_id} ({obj['id']})")
                continue

            obj_id = int(obj["id"])
            # STEP 1: Select frames where the current object is visible according to the
            # current mask source.
            selected_frame_ids = []
            selected_masks = []
            for frame_id in frame_idxs:
                frame_mask = _lookup_mask_frame(mask, frame_id)
                if frame_mask is None:
                    continue
                obj_mask = np.where(np.asarray(frame_mask) == int(obj_id), 1, 0)
                if obj_mask.sum() > 0:
                    selected_frame_ids.append(frame_id)
                    selected_masks.append(obj_mask)

            if len(selected_frame_ids) == 0:
                _LOGGER.info(f"Skipping {scan_id} ({obj['id']}) because object is not visible in frames")
                continue

            selected_frame_ids, selected_masks = _filter_selected_masks(
                selected_frame_ids, selected_masks, object_source=object_source
            )
            if len(selected_frame_ids) == 0:
                _LOGGER.info(
                    "Skipping %s (%s) because cleaned masks are empty in all frames",
                    scan_id,
                    obj_id,
                )
                continue
            selected_frame_ids, selected_masks = _select_object_frames(
                selected_frame_ids,
                selected_masks,
                extrinsics=extrinsics,
                max_views=max_views,
            )
            min_selected_frames = int(os.getenv("OBJECTX_VOXEL_MIN_SELECTED_FRAMES", "0"))
            if len(selected_frame_ids) < max(0, min_selected_frames):
                _LOGGER.info(
                    "Skipping %s (%s) because selected_frames=%s < min_selected_frames=%s",
                    scan_id,
                    obj_id,
                    len(selected_frame_ids),
                    min_selected_frames,
                )
                continue

            rendered_obj = []
            selected_depths = []
            for frame_id, obj_mask in zip(selected_frame_ids, selected_masks):
                image = Image.open(
                    f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
                ).convert("RGB")
                image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                rendered_obj.append(image_t * torch.from_numpy(obj_mask[None, :, :]))
                if requires_depth:
                    if frame_id not in depth_map_cache:
                        depth_map_cache[frame_id] = scan3r.load_depth_map(
                            osp.join(
                                root_dir,
                                "scenes",
                                scan_id,
                                "sequence",
                                f"frame-{frame_id}.depth.pgm",
                            ),
                            depth_shift,
                        )
                    selected_depths.append(depth_map_cache[frame_id])

            rendered_obj = torch.stack(rendered_obj).float()
            pose_camera_to_world = _resolve_pose_camera_to_world(
                extrinsics, selected_frame_ids
            )
            _log_rss(
                f"[2.5] object {scan_id}/{obj_id} selected_frames={len(selected_frame_ids)}"
            )

            # STEP 2/3: Build object geometry and voxel grid from the selected source.
            if object_source == "lifted_masks":
                voxel_grid, mean, scale = _build_lifted_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            elif object_source == "tsdf_masks":
                voxel_grid, mean, scale = _build_tsdf_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            elif object_source == "hybrid_masks":
                voxel_grid, mean, scale = _build_hybrid_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                )
            else:
                segmented_mesh = _segment_mesh(mesh, annos, obj_id, scan_id)
                mean, scale = _normalize_segmented_mesh(segmented_mesh)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                    segmented_mesh,
                    1 / 64,
                    min_bound=(-0.5, -0.5, -0.5),
                    max_bound=(0.5, 0.5, 0.5),
                )
                voxel_grid = _dilate_voxels(voxel_grid)

            # STEP 4: Save mean and scale (Scene composition)
            if not args.dry_run:
                np.savez(mean_scale_path, mean=mean, scale=scale)
                _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

            if (
                os.path.exists(voxel_path) and "arr_0" in np.load(voxel_path)
            ) and not args.override:
                _LOGGER.info(f"Skipping {scan_id} ({obj['id']})")
                continue

            # STEP 5: Project the voxel to the image
            pose_world_to_camera = _invert_pose_list(pose_camera_to_world)

            projection_color, linear_depth = _project_to_image(
                torch.Tensor(voxel_grid),
                torch.Tensor(mean),
                torch.Tensor([scale]),
                torch.from_numpy(np.stack(pose_world_to_camera)),
                torch.from_numpy(intrinsics["intrinsic_mat"]),
            )  # Shape: (Nimages, Npoints, 2)
            observed_views = None
            observed_voxels = None
            if filter_unobserved:
                projection_depth, _ = _project_to_image(
                    torch.Tensor(voxel_grid),
                    torch.Tensor(mean),
                    torch.Tensor([scale]),
                    torch.from_numpy(np.stack(pose_world_to_camera)),
                    torch.from_numpy(depth_intrinsics["intrinsic_mat"]),
                )
                observed_views = _compute_voxel_observations(
                    projection_color=projection_color,
                    projection_depth=projection_depth,
                    linear_depth=linear_depth,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    color_size=(intrinsics["width"], intrinsics["height"]),
                    depth_size=(
                        depth_intrinsics["width"],
                        depth_intrinsics["height"],
                    ),
                    depth_abs_tol=depth_abs_tol,
                    depth_rel_tol=depth_rel_tol,
                )
                observed_voxels = observed_views.any(axis=0)
                _LOGGER.info(
                    "[2.5] object %s/%s observed_voxels=%s/%s",
                    scan_id,
                    obj_id,
                    int(observed_voxels.sum()),
                    int(observed_voxels.shape[0]),
                )
                if not np.any(observed_voxels):
                    _LOGGER.info(
                        "Skipping %s (%s) because no voxels passed visibility checks",
                        scan_id,
                        obj_id,
                        )
                    continue

            # STEP 6: Normalize the projection to [-1, 1]
            projection = (
                projection_color
                / torch.Tensor([intrinsics["width"], intrinsics["height"]]).float()
            ) * 2.0 - 1.0

            # STEP 7: Get the DINO embeddings
            patch_embeddings = _get_dino_embedding(
                rendered_obj
            )  # Shape: (Nimages, 1024, 64, 64)

            # STEP 8: Match the embeddings to the projection
            patchtokens = (
                F.grid_sample(
                    patch_embeddings,
                    projection.cuda().unsqueeze(1),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(2)
                .permute(0, 2, 1)
                .cpu()
                .numpy()
            )  # Shape: (Nimages, Npoints, 1024)

            if filter_unobserved:
                observation_weights = observed_views.astype(np.float32)[..., None]
                observation_count = observation_weights.sum(axis=0)
                patchtokens = np.divide(
                    (patchtokens * observation_weights).sum(axis=0),
                    np.clip(observation_count, 1.0, None),
                )
                patchtokens = patchtokens[observed_voxels]
                voxel_grid = voxel_grid[observed_voxels]
            else:
                patchtokens = np.mean(patchtokens, axis=0)

            patchtokens = patchtokens.astype(np.float16)  # Shape: (Npoints, 1024)
            assert patchtokens.shape[0] == voxel_grid.shape[0]
            assert patchtokens.shape[1] == 1024
            assert voxel_grid.shape[1] == 3
            voxel_grid = torch.concatenate(
                [torch.Tensor(voxel_grid), torch.Tensor(patchtokens)], dim=1
            )
            if args.visualize:
                vis.save_voxel_as_ply(
                    voxel_grid.cpu().numpy(),
                    f"vis/{scan_id}_{obj_id}_voxel.ply",
                    show_color=True,
                )
            assert voxel_grid.shape[-1] == 1027
            if not args.dry_run:
                _save_featured_voxel(
                    voxel_grid,
                    output_file=voxel_path,
                )
                _log_rss(f"[2.5] saved voxel {scan_id}/{obj_id}")
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            _LOGGER.exception(f"Error processing {scan_id} ({obj_id}): {e}")
        finally:
            for name in [
                "segmented_mesh",
                "voxel_grid",
                "rendered_obj",
                "pose_camera_to_world",
                "projection",
                "projection_color",
                "projection_depth",
                "linear_depth",
                "observed_views",
                "observed_voxels",
                "patch_embeddings",
                "patchtokens",
                "selected_masks",
                "selected_frame_ids",
                "selected_depths",
            ]:
                if name in locals():
                    del locals()[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def process_data(
    cfg: Config, mode: str = "gs_annotations", split: str = "train"
) -> np.ndarray:
    """
    Process scans to get featured voxel representation.

    Args:
        cfg: Configuration object.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations".
        split (str, optional): Split to run subscan generation on. Defaults to "train".

    Returns:
        np.ndarray: processed subscan IDs.
    """

    scan_type = cfg.autoencoder.encoder.scan_type
    resplit = "resplit_" if cfg.data.resplit else ""
    scan_ids_filename = (
        f"{split}_{resplit}scans.txt"
        if scan_type == "scan"
        else f"{split}_scans_subscenes.txt"
    )
    objects_info_file = osp.join(root_dir, "files", "objects.json")
    all_obj_info = common.load_json(objects_info_file)

    subscan_ids_generated = np.genfromtxt(
        osp.join(root_dir, "files", scan_ids_filename), dtype=str
    )
    subscan_ids_processed = []

    subRescan_ids_generated = {}
    scans_dir = cfg.data.root_dir
    scans_files_dir = osp.join(scans_dir, "files")

    all_scan_data = common.load_json(osp.join(scans_files_dir, "3RScan.json"))

    for scan_data in all_scan_data:
        ref_scan_id = scan_data["reference"]
        if ref_scan_id in subscan_ids_generated:
            rescan_ids = [scan["reference"] for scan in scan_data["scans"]]
            subRescan_ids_generated[ref_scan_id] = [ref_scan_id] + rescan_ids

    subscan_ids_generated = subRescan_ids_generated

    all_subscan_ids = [
        subscan_id
        for scan_id in subscan_ids_generated
        for subscan_id in subscan_ids_generated[scan_id]
    ]

    for subscan_id in tqdm(all_subscan_ids):
        obj_data = next(
            obj_data
            for obj_data in all_obj_info["scans"]
            if obj_data["scan"] == subscan_id
        )

        voxelise_features(
            mode=mode,
            obj_data=obj_data,
            scan_id=subscan_id,
        )

        subscan_ids_processed.append(subscan_id)

    subscan_ids = np.array(subscan_ids_processed)
    return subscan_ids


def parse_args() -> Tuple[Namespace, list]:
    """
    Parse command line arguments.

    Returns:
        Tuple[argparse.Namespace, list]: Parsed arguments and unknown arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="train",
        type=str,
    )
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="vis")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--mask-source", type=str, default=None)
    parser.add_argument("--object-source", type=str, default=None)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    common.init_log(level=logging.INFO)
    _LOGGER.info("**** Starting feature voxelisation for 3RScan ****")
    args, unknown = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    cfg = update_configs(args.config, unknown, do_ensure_dir=False)
    root_dir = cfg.data.root_dir

    model = _load_dino_model(args.model)
    model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    scan_ids = process_data(cfg, mode="gs_annotations", split=args.split)
