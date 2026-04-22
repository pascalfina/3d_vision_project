import itertools
import logging
import os
import os.path as osp
import gc
import resource
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from tqdm import tqdm

from configs import Config, update_configs
from utils import common, scan3r
from utils import visualisation as vis

_LOGGER = logging.getLogger(__name__)


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
    points_world = voxel * scale + mean  # (N, 3)

    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics with shape (V,4,4), got {tuple(extrinsics.shape)}")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(extrinsics.shape[0], -1, -1)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (V,3,3) or (3,3), got {tuple(intrinsics.shape)}")

    n_views = extrinsics.shape[0]
    n_points = points_world.shape[0]
    points_h = torch.cat(
        [points_world.float(), torch.ones((n_points, 1), device=points_world.device)],
        dim=1,
    )  # (N, 4)
    points_h = points_h.unsqueeze(0).expand(n_views, -1, -1)  # (V, N, 4)
    cam_h = torch.bmm(extrinsics.float(), points_h.transpose(1, 2))  # (V, 4, N)
    cam_xyz = cam_h[:, :3, :]  # (V, 3, N)

    x = cam_xyz[:, 0, :]
    y = cam_xyz[:, 1, :]
    z = cam_xyz[:, 2, :]
    z_safe = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)

    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    uv = torch.stack([u, v], dim=-1)  # (V, N, 2)
    linear_depth = z  # (V, N)
    return uv, linear_depth


def _load_depth_shift(data_dir: str, scan_id: str) -> float:
    info_path = osp.join(data_dir, scan_id, "sequence", "_info.txt")
    with open(info_path) as f:
        for line in f:
            if line.startswith("m_depthShift"):
                return float(line.split("=", 1)[1].strip())
    return 1000.0


def _normalize_frame_id(frame_id) -> str:
    if isinstance(frame_id, (int, np.integer)):
        return f"{int(frame_id):06d}"
    text = str(frame_id).strip()
    if text.startswith("frame-"):
        text = text.split("frame-", 1)[1]
    text = text.split(".", 1)[0]
    if text.isdigit():
        return f"{int(text):06d}"
    return text


def _normalize_mask_frame_keys(mask_by_frame: dict) -> dict:
    if not isinstance(mask_by_frame, dict):
        raise ValueError("Expected mask dictionary keyed by frame id")
    normalized = {}
    for raw_key, mask in mask_by_frame.items():
        norm_key = _normalize_frame_id(raw_key)
        if norm_key in normalized and not np.array_equal(normalized[norm_key], mask):
            raise ValueError(
                f"Mask key collision after normalization for frame {norm_key!r}"
            )
        normalized[norm_key] = mask
    return normalized


def _load_predicted_frame_poses(
    predicted_scenes_dir: str, scan_id: str, frame_idxs: list[str]
) -> dict[str, np.ndarray]:
    sequence_dir = Path(predicted_scenes_dir) / scan_id / "sequence"
    if not sequence_dir.exists():
        raise FileNotFoundError(
            f"Predicted sequence directory does not exist: {sequence_dir}"
        )

    frame_poses: dict[str, np.ndarray] = {}
    missing = []
    for frame_idx in frame_idxs:
        pose_path = sequence_dir / f"frame-{frame_idx}.pose.txt"
        if not pose_path.exists():
            missing.append(frame_idx)
            continue
        frame_poses[frame_idx] = np.loadtxt(pose_path).astype(np.float32).reshape(4, 4)

    if missing:
        preview = ",".join(missing[:5])
        raise FileNotFoundError(
            f"Missing predicted poses for {len(missing)} frames in {sequence_dir}. "
            f"Examples: {preview}"
        )
    return frame_poses


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
    if normalized_points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Open3D voxelization may segfault on this cluster build; keep this path pure numpy.
    points = normalized_points.astype(np.float32, copy=False)
    voxel_indices = np.floor((points + 0.5) * 64.0).astype(np.int32)
    valid = np.all((voxel_indices >= 0) & (voxel_indices < 64), axis=1)
    voxel_indices = voxel_indices[valid]
    if voxel_indices.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    occupancy = np.zeros((64, 64, 64), dtype=np.uint8)
    occupancy[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2],
    ] = 1
    if dilate_iters > 0:
        occupancy = ndimage.binary_dilation(
            occupancy.astype(bool),
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(dilate_iters),
        ).astype(np.uint8)
    return np.argwhere(occupancy > 0).astype(np.int32)


def _clean_object_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.astype(bool)
    if cleaned.sum() == 0:
        return cleaned.astype(np.uint8)

    # Keep only the largest connected component to reduce mask noise.
    labeled, num_labels = ndimage.label(cleaned)
    if num_labels > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        cleaned = labeled == counts.argmax()

    if cleaned.sum() < 16:
        return np.zeros_like(mask, dtype=np.uint8)
    return cleaned.astype(np.uint8)


def _filter_selected_masks(
    selected_frame_ids: list[str], selected_masks: list[np.ndarray]
) -> tuple[list[str], list[np.ndarray]]:
    filtered_ids = []
    filtered_masks = []
    for frame_id, obj_mask in zip(selected_frame_ids, selected_masks):
        cleaned = _clean_object_mask(obj_mask)
        if cleaned.sum() > 0:
            filtered_ids.append(frame_id)
            filtered_masks.append(cleaned)
    return filtered_ids, filtered_masks


def _select_object_frames(
    frame_ids: list[str],
    masks: list[np.ndarray],
    extrinsics: dict[str, np.ndarray],
    max_views: int,
) -> tuple[list[str], list[np.ndarray]]:
    if max_views <= 0 or len(frame_ids) <= max_views:
        return frame_ids, masks
    del extrinsics  # kept in signature for compatibility with existing callers
    areas = np.array([mask.sum() for mask in masks], dtype=np.float32)
    keep = np.argsort(-areas, kind="stable")[:max_views]
    keep = np.sort(keep.astype(np.int64))
    selected_frame_ids = [frame_ids[i] for i in keep.tolist()]
    selected_masks = [masks[i] for i in keep.tolist()]
    selected_areas = [int(masks[i].sum()) for i in keep.tolist()]
    preview = ",".join(selected_frame_ids[: min(5, len(selected_frame_ids))])
    _LOGGER.info(
        "[2.5] frame_selection top_area kept=%s/%s area[min=%s mean=%.1f max=%s] ids=%s",
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


def _invert_pose_list(poses: list[np.ndarray]) -> list[np.ndarray]:
    return [np.linalg.inv(pose).astype(np.float32) for pose in poses]


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] >= 32:
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
    lower_q = 1.0
    upper_q = 99.0
    if 0.0 <= lower_q < upper_q <= 100.0:
        lower = np.percentile(filtered, lower_q, axis=0)
        upper = np.percentile(filtered, upper_q, axis=0)
        keep = np.all((filtered >= lower) & (filtered <= upper), axis=1)
        if keep.sum() >= max(32, int(0.2 * filtered.shape[0])):
            filtered = filtered[keep]

    if filtered.shape[0] >= 64:
        center = np.median(filtered, axis=0)
        distances = np.linalg.norm(filtered - center[None, :], axis=1)
        keep_q = 99.0
        keep_radius = np.percentile(distances, keep_q)
        keep = distances <= keep_radius
        if keep.sum() >= max(32, int(0.2 * filtered.shape[0])):
            filtered = filtered[keep]

    return filtered.astype(np.float32, copy=False)


def _save_point_cloud(points: np.ndarray, output_file: str) -> None:
    if points.size == 0:
        return
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(output_file, point_cloud)


def _save_point_cloud_with_colors(
    points: np.ndarray, colors: np.ndarray, output_file: str
) -> None:
    if points.size == 0:
        return
    if colors.shape[0] != points.shape[0] or colors.shape[1] != 3:
        raise ValueError(
            f"Invalid colors shape {colors.shape} for points shape {points.shape}"
        )
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.clip(colors.astype(np.float32), 0.0, 1.0).astype(np.float64)
    )
    o3d.io.write_point_cloud(output_file, point_cloud)


def _save_triangle_mesh(mesh: o3d.geometry.TriangleMesh, output_file: str) -> None:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return
    o3d.io.write_triangle_mesh(output_file, mesh)


def _color_from_obj_id(obj_id: int) -> np.ndarray:
    # Deterministic color from object id using golden-ratio hue spacing.
    hue = (obj_id * 0.6180339887498949) % 1.0
    sat = 0.65
    val = 0.95
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    i = i % 6
    if i == 0:
        rgb = (val, t, p)
    elif i == 1:
        rgb = (q, val, p)
    elif i == 2:
        rgb = (p, val, t)
    elif i == 3:
        rgb = (p, q, val)
    elif i == 4:
        rgb = (t, p, val)
    else:
        rgb = (val, p, q)
    return np.array(rgb, dtype=np.float32)


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

        pixels = np.stack(
            [
                x.astype(np.float32),
                y.astype(np.float32),
                np.ones_like(x, dtype=np.float32),
            ],
            axis=0,
        )
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
        "[2.5] lifted points filtered %s -> %s",
        int(lifted_points.shape[0]),
        int(filtered_points.shape[0]),
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
    min_lifted_points = 64
    if lifted_points.shape[0] < max(0, min_lifted_points):
        raise ValueError(
            f"Too few lifted points ({lifted_points.shape[0]}) for object quality threshold"
        )
    normalized_points, mean, scale = _normalize_points(lifted_points)
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
    min_voxels = 32
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


def _build_tsdf_object_voxel_grid(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
    pointcloud_output_file: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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

    tsdf_resolution = 96
    tsdf_sdf_trunc = 3.0 / tsdf_resolution
    tsdf_depth_trunc = 10.0
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
        raise ValueError("No valid RGBD frames could be integrated into TSDF")

    tsdf_mesh = volume.extract_triangle_mesh()
    tsdf_mesh.compute_vertex_normals()
    tsdf_mesh.remove_duplicated_vertices()
    tsdf_mesh.remove_duplicated_triangles()
    tsdf_mesh.remove_unreferenced_vertices()
    tsdf_mesh.remove_degenerate_triangles()
    tsdf_mesh.remove_non_manifold_edges()

    fallback_to_lifted = True
    if len(tsdf_mesh.vertices) == 0 or len(tsdf_mesh.triangles) == 0:
        if not fallback_to_lifted:
            raise ValueError("TSDF fusion produced an empty mesh")
        _LOGGER.warning(
            "[2.5] object %s/%s TSDF mesh empty, falling back to lifted-point voxelization",
            scan_id,
            obj_id,
        )
        voxel_grid = _voxelize_normalized_points(normalized_points, dilate_iters=1)
        voxel_grid = _finalize_object_voxel_grid(
            voxel_grid,
            scan_id=scan_id,
            obj_id=obj_id,
            variant="fallback_lifted",
        )
        if pointcloud_output_file is not None:
            _save_point_cloud(lifted_points, pointcloud_output_file)
        if args.visualize:
            _save_point_cloud(
                lifted_points, f"vis/{scan_id}_{obj_id}_lifted_points_world.ply"
            )
            _save_point_cloud(
                normalized_points,
                f"vis/{scan_id}_{obj_id}_lifted_points_normalized.ply",
            )
        return voxel_grid, mean, scale, lifted_points

    if pointcloud_output_file is not None:
        _save_point_cloud(lifted_points, pointcloud_output_file)
    if args.visualize:
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
    tsdf_dilate_iters = 0
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
    voxel_grid = _finalize_object_voxel_grid(
        voxel_grid,
        scan_id=scan_id,
        obj_id=obj_id,
        variant="tsdf",
    )

    return voxel_grid, mean, scale, lifted_points


def _build_lifted_object_voxel_grid(
    scan_id: str,
    obj_id: int,
    selected_masks: list[np.ndarray],
    selected_depths: list[np.ndarray],
    pose_camera_to_world: list[np.ndarray],
    depth_intrinsics: dict,
    pointcloud_output_file: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
    if pointcloud_output_file is not None:
        _save_point_cloud(lifted_points, pointcloud_output_file)
    if args.visualize:
        _save_point_cloud(
            lifted_points, f"vis/{scan_id}_{obj_id}_lifted_points_world.ply"
        )
        _save_point_cloud(
            normalized_points,
            f"vis/{scan_id}_{obj_id}_lifted_points_normalized.ply",
        )
    return voxel_grid, mean, scale, lifted_points


@torch.no_grad()
def voxelise_features(
    obj_data: Dict[str, str],
    scan_id: str,
    mode: str = "gs_annotations_predicted",
) -> None:
    """
    Voxelise features for scan.

    Args:
        obj_data (Dict[str, str]): Object data.
        scan_id (str): Scan ID.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations_predicted".
    """

    rgb_scenes_dir = osp.join(root_dir, "scenes")
    predicted_scenes_dir = osp.join(root_dir, "scenes_predicted")
    frame_idxs = [
        _normalize_frame_id(frame_idx)
        for frame_idx in scan3r.load_frame_idxs(data_dir=rgb_scenes_dir, scan_id=scan_id)
    ]
    extrinsics = _load_predicted_frame_poses(
        predicted_scenes_dir=predicted_scenes_dir,
        scan_id=scan_id,
        frame_idxs=frame_idxs,
    )
    intrinsics = scan3r.load_intrinsics(data_dir=predicted_scenes_dir, scan_id=scan_id)
    depth_intrinsics = scan3r.load_intrinsics(
        data_dir=predicted_scenes_dir, scan_id=scan_id, type="depth"
    )
    depth_shift = _load_depth_shift(predicted_scenes_dir, scan_id)
    depth_map_cache = {}

    mask_source = scan3r.resolve_mask_source(
        getattr(args, "mask_source", None) or "gt_projection_predicted"
    )
    object_source = (getattr(args, "object_source", None) or "lifted_masks").strip().lower()
    if object_source not in {"lifted_masks", "tsdf_masks"}:
        raise ValueError(
            f"Unsupported --object-source '{object_source}'. "
            "Expected one of: lifted_masks, tsdf_masks"
        )
    _LOGGER.info("[2.5] using mask source: %s", mask_source)
    _LOGGER.info("[2.5] object geometry source: %s", object_source)
    mask = _normalize_mask_frame_keys(
        scan3r.load_masks(data_dir=root_dir, scan_id=scan_id, mask_source=mask_source)
    )
    max_views = 150
    _log_rss(f"[2.5] loaded scan {scan_id} with {len(frame_idxs)} frames")
    scene_lifted_points: list[np.ndarray] = []
    scene_lifted_colors: list[np.ndarray] = []

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

            has_existing_outputs = (
                osp.exists(mean_scale_path)
                and osp.exists(voxel_path)
                and "arr_0" in np.load(voxel_path)
            )

            obj_id = int(obj["id"])
            # STEP 1: Select frames where the current object is visible according to the
            # current mask source.
            selected_frame_ids = []
            selected_masks = []
            missing_mask_frames = 0
            for frame_id in frame_idxs:
                frame_mask = mask.get(frame_id)
                if frame_mask is None:
                    missing_mask_frames += 1
                    continue
                obj_mask = np.where(frame_mask == int(obj_id), 1, 0)
                if obj_mask.sum() > 0:
                    selected_frame_ids.append(frame_id)
                    selected_masks.append(obj_mask)
            if missing_mask_frames > 0:
                _LOGGER.warning(
                    "Scan %s object %s skipped %d frames missing mask key",
                    scan_id,
                    obj_id,
                    missing_mask_frames,
                )

            if len(selected_frame_ids) == 0:
                _LOGGER.info(f"Skipping {scan_id} ({obj['id']}) because object is not visible in frames")
                continue

            selected_frame_ids, selected_masks = _filter_selected_masks(
                selected_frame_ids, selected_masks
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
            min_selected_frames = 1
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
                    f"{rgb_scenes_dir}/{scan_id}/sequence/frame-{frame_id}.color.jpg"
                ).convert("RGB")
                image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                rendered_obj.append(image_t * torch.from_numpy(obj_mask[None, :, :]))
                if frame_id not in depth_map_cache:
                    depth_map_cache[frame_id] = scan3r.load_depth_map(
                        osp.join(
                            predicted_scenes_dir,
                            scan_id,
                            "sequence",
                            f"frame-{frame_id}.depth.pgm",
                        ),
                        depth_shift,
                    )
                selected_depths.append(depth_map_cache[frame_id])

            rendered_obj = torch.stack(rendered_obj).float()
            pose_camera_to_world = [
                np.array(extrinsics[frame_id], copy=True) for frame_id in selected_frame_ids
            ]
            _log_rss(
                f"[2.5] object {scan_id}/{obj_id} selected_frames={len(selected_frame_ids)}"
            )

            object_out_dir = osp.dirname(voxel_path)
            pointcloud_path = (
                osp.join(object_out_dir, "pointcloud_world.ply")
                if (not args.dry_run and args.save_object_pointclouds)
                else None
            )
            if object_source == "lifted_masks":
                voxel_grid, mean, scale, lifted_points = _build_lifted_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                    pointcloud_output_file=pointcloud_path,
                )
            else:
                voxel_grid, mean, scale, lifted_points = _build_tsdf_object_voxel_grid(
                    scan_id=scan_id,
                    obj_id=obj_id,
                    selected_masks=selected_masks,
                    selected_depths=selected_depths,
                    pose_camera_to_world=pose_camera_to_world,
                    depth_intrinsics=depth_intrinsics,
                    pointcloud_output_file=pointcloud_path,
                )
            scene_lifted_points.append(lifted_points)
            obj_color = _color_from_obj_id(obj_id)
            scene_lifted_colors.append(
                np.repeat(obj_color[None, :], lifted_points.shape[0], axis=0)
            )

            # STEP 4: Save mean and scale (Scene composition)
            if not args.dry_run and (args.override or not osp.exists(mean_scale_path)):
                np.savez(mean_scale_path, mean=mean, scale=scale)
                _LOGGER.info(f"Saved mean and scale to {mean_scale_path}")

            #if has_existing_outputs and not args.override:
            #    _LOGGER.info(
            #        "Skipping voxel feature extraction for %s (%s) because outputs exist",
            #        scan_id,
            #        obj["id"],
            #    )
            #    continue

            # STEP 5: Project the voxel to the image
            pose_world_to_camera = _invert_pose_list(pose_camera_to_world)

            projection_color, linear_depth = _project_to_image(
                torch.Tensor(voxel_grid),
                torch.Tensor(mean),
                torch.Tensor([scale]),
                torch.from_numpy(np.stack(pose_world_to_camera)),
                torch.from_numpy(intrinsics["intrinsic_mat"]),
            )  # Shape: (Nimages, Npoints, 2)
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
                "voxel_grid",
                "rendered_obj",
                "pose_camera_to_world",
                "projection",
                "projection_color",
                "linear_depth",
                "patch_embeddings",
                "patchtokens",
                "selected_masks",
                "selected_frame_ids",
                "selected_depths",
                "lifted_points",
            ]:
                if name in locals():
                    del locals()[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.dry_run:
        scene_out_dir = osp.join(args.model_dir, "files", mode, scan_id)
        os.makedirs(scene_out_dir, exist_ok=True)
        scene_pointcloud_path = osp.join(scene_out_dir, "pointcloud_world.ply")
        if scene_lifted_points:
            merged_scene_points = np.concatenate(scene_lifted_points, axis=0)
            merged_scene_colors = np.concatenate(scene_lifted_colors, axis=0)
            _save_point_cloud_with_colors(
                merged_scene_points, merged_scene_colors, scene_pointcloud_path
            )
            _LOGGER.info(
                "[2.5] saved scene pointcloud: %s (%d points)",
                scene_pointcloud_path,
                int(merged_scene_points.shape[0]),
            )
        else:
            _LOGGER.warning(
                "[2.5] no object pointclouds generated for %s; scene pointcloud not saved",
                scan_id,
            )


def process_data(
    cfg: Config, mode: str = "gs_annotations_predicted", split: str = "train"
) -> np.ndarray:
    """
    Process scans to get featured voxel representation.

    Args:
        cfg: Configuration object.
        mode (str, optional): Mode to run subscan generation on. Defaults to "gs_annotations_predicted".
        split (str, optional): Split to run subscan generation on. Defaults to "train".

    Returns:
        np.ndarray: processed subscan IDs.
    """
    objects_info_file = osp.join(cfg.data.root_dir, "files", args.objects_file)
    if not osp.exists(objects_info_file):
        raise FileNotFoundError(
            f"Predicted objects registry not found: {objects_info_file}"
        )
    all_obj_info = common.load_json(objects_info_file)
    if not isinstance(all_obj_info, dict) or not isinstance(all_obj_info.get("scans"), list):
        raise ValueError(
            f"Invalid registry format in {objects_info_file}; expected {{'scans': [...]}}"
        )
    obj_lookup = {
        entry["scan"]: entry
        for entry in all_obj_info["scans"]
        if isinstance(entry, dict)
        and isinstance(entry.get("scan"), str)
        and isinstance(entry.get("objects"), list)
    }
    subscan_ids_generated = sorted(obj_lookup.keys())
    if args.scene:
        if args.scene not in obj_lookup:
            raise ValueError(
                f"Scene {args.scene} not found in {objects_info_file}"
            )
        subscan_ids_generated = [args.scene]

    subscan_ids_processed = []
    for subscan_id in tqdm(subscan_ids_generated):
        obj_data = obj_lookup.get(subscan_id)
        if obj_data is None:
            _LOGGER.warning(
                "Skipping %s because it is missing in %s",
                subscan_id,
                args.objects_file,
            )
            continue
        if not obj_data["objects"]:
            _LOGGER.warning(
                "Skipping %s because predicted object list is empty",
                subscan_id,
            )
            continue

        voxelise_features(
            mode=mode,
            obj_data=obj_data,
            scan_id=subscan_id,
        )
        subscan_ids_processed.append(subscan_id)

    return np.array(subscan_ids_processed)


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
    parser.add_argument("--save-object-pointclouds", action="store_true")
    parser.add_argument("--object-source", type=str, default="lifted_masks")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--mask-source", type=str, default=None)
    parser.add_argument("--objects-file", type=str, default="objects_predicted.json")
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
    scan_ids = process_data(cfg, mode="gs_annotations_predicted", split=args.split)
