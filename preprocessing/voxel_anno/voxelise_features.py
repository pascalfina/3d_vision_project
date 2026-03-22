import itertools
import logging
import os
import os.path as osp
import gc
import resource
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
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
        else:
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


def _normalize_segmented_mesh(segmented_mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(segmented_mesh.vertices)
    mean = vertices.mean(axis=0)
    vertices -= mean
    scale = np.max(np.abs(vertices))
    vertices *= 1.0 / (2 * scale)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    segmented_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mean, scale


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
    depth_intrinsics = None
    depth_shift = None
    depth_abs_tol = float(os.getenv("OBJECTX_VOXEL_DEPTH_ABS_TOL", "0.05"))
    depth_rel_tol = float(os.getenv("OBJECTX_VOXEL_DEPTH_REL_TOL", "0.02"))
    depth_map_cache = {}
    if filter_unobserved:
        depth_intrinsics = scan3r.load_intrinsics(
            data_dir=scenes_dir, scan_id=scan_id, type="depth"
        )
        depth_shift = _load_depth_shift(scenes_dir, scan_id)
    mask = scan3r.load_masks(data_dir=root_dir, scan_id=scan_id)
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
            # STEP 1: Segment the mesh
            segmented_mesh = _segment_mesh(mesh, annos, obj_id, scan_id)
            # STEP 2: Normalize to unit cube (-0.5, 0.5)
            mean, scale = _normalize_segmented_mesh(segmented_mesh)
            # STEP 3: Voxelise the mesh
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

            # STEP 5: Render the object
            selected_frame_ids = []
            selected_masks = []
            for frame_id in frame_idxs:
                obj_mask = np.where(mask[frame_id] == int(obj_id), 1, 0)
                if obj_mask.sum() > 0:
                    selected_frame_ids.append(frame_id)
                    selected_masks.append(obj_mask)
                if len(selected_frame_ids) >= max_views:
                    break

            if len(selected_frame_ids) == 0:
                _LOGGER.info(f"Skipping {scan_id} ({obj['id']}) because object is not visible in frames")
                continue

            rendered_obj = []
            selected_depths = []
            for frame_id, obj_mask in zip(selected_frame_ids, selected_masks):
                image = Image.open(
                    f"{root_dir}/scenes/{scan_id}/sequence/frame-{frame_id}.color.jpg"
                ).convert("RGB")
                image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                rendered_obj.append(image_t * torch.from_numpy(obj_mask[None, :, :]))
                if filter_unobserved:
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
            pose_camera_to_world = [
                np.linalg.inv(extrinsics[frame_id]) for frame_id in selected_frame_ids
            ]
            _log_rss(
                f"[2.5] object {scan_id}/{obj_id} selected_frames={len(selected_frame_ids)}"
            )

            # STEP 6: Project the voxel to the image
            projection_color, linear_depth = _project_to_image(
                torch.Tensor(voxel_grid),
                torch.Tensor(mean),
                torch.Tensor([scale]),
                torch.from_numpy(np.stack(pose_camera_to_world)),
                torch.from_numpy(intrinsics["intrinsic_mat"]),
            )  # Shape: (Nimages, Npoints, 2)
            observed_views = None
            observed_voxels = None
            if filter_unobserved:
                projection_depth, _ = _project_to_image(
                    torch.Tensor(voxel_grid),
                    torch.Tensor(mean),
                    torch.Tensor([scale]),
                    torch.from_numpy(np.stack(pose_camera_to_world)),
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

            # STEP 7: Normalize the projection to [-1, 1]
            projection = (
                projection_color
                / torch.Tensor([intrinsics["width"], intrinsics["height"]]).float()
            ) * 2.0 - 1.0

            # STEP 8: Get the DINO embeddings
            patch_embeddings = _get_dino_embedding(
                rendered_obj
            )  # Shape: (Nimages, 1024, 64, 64)

            # STEP 9: Match the embeddings to the projection
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
