import argparse
import copy
import logging
import os
import os.path as osp

import imageio
import numpy as np
import open3d as o3d
import torch
from gaussian_renderer import render
from PIL import Image
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RSceneGraphDataset
from src.models.autoencoder import AutoEncoder
from src.models.latent_autoencoder import LatentAutoencoder
from src.modules.sparse.basic import SparseTensor, sparse_cat
from src.representations import Gaussian
from utils import common, mesh, torch_util
from utils.gaussian_camera import build_minicam
from utils.gaussian_splatting import GaussianSplat
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

_REPRESENTATION_CONFIG = {
    "perturb_offset": True,
    "voxel_size": 1.5,
    "num_gaussians": 32,
    "2d_filter_kernel_size": 0.1,
    "3d_filter_kernel_size": 9e-4,
    "scaling_bias": 4e-3,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}


_LOGGER = logging.getLogger(__name__)


class SceneGraph2UnstructuredLatentPipeline:
    def __init__(
        self,
        cfg: Config,
        visualize=False,
        split="train",
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.dataset = Scan3RSceneGraphDataset(cfg, split)
        self.vis = visualize
        self.rep_config = _REPRESENTATION_CONFIG
        self.output_dir = osp.join(cfg.data.root_dir, cfg.inference.output_dir)
        self.max_objects_per_chunk = int(
            os.environ.get("OBJECTX_SLAT_MAX_OBJECTS_PER_CHUNK", "16")
        )

    def load_model(self):
        self.latent_autoencoder = LatentAutoencoder(
            cfg=self.cfg.autoencoder, device=self.device
        )
        self.model = AutoEncoder(cfg=self.cfg.autoencoder, device=self.device)
        self.latent_autoencoder.load_state_dict(
            torch.load(
                self.cfg.inference.slat_model_path,
                map_location="cpu" if not torch.cuda.is_available() else "cuda",
            )["model"],
            strict=False,
        )
        self.model.load_state_dict(
            torch.load(
                self.cfg.inference.ulat_model_path,
                map_location="cpu" if not torch.cuda.is_available() else "cuda",
            )["model"],
            strict=False,
        )
        self.rep_config = self.latent_autoencoder.decoder.rep_config
        return self.model

    def _densify(self, sparse_splat, fill=0.0):
        shape = sparse_splat.dense().shape
        dense_splat = torch.full(
            (shape[0], 1, shape[2], shape[3], shape[4]),
            fill,
            device=sparse_splat.device,
        )
        dense_splat[
            sparse_splat.coords[:, 0],
            :,
            sparse_splat.coords[:, 1],
            sparse_splat.coords[:, 2],
            sparse_splat.coords[:, 3],
        ] = 1.0
        return torch.cat((sparse_splat.dense(), dense_splat), dim=1)

    def _sparsify(self, dense_splat, occupancy_threshold=None, max_voxels=None):
        if occupancy_threshold is None:
            occupancy_threshold = float(
                os.environ.get("OBJECTX_INFER_OCC_THRESHOLD", "0.1")
            )
        if max_voxels is None:
            max_voxels = int(os.environ.get("OBJECTX_INFER_MAX_VOXELS", "0"))
        occupancy = dense_splat[:, -1]
        coords = torch.nonzero(occupancy > occupancy_threshold, as_tuple=False)
        if max_voxels > 0 and coords.shape[0] > max_voxels:
            scores = occupancy[
                coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            ]
            keep = torch.topk(scores, k=max_voxels, largest=True, sorted=False).indices
            coords = coords[keep]
        feats = dense_splat[coords[:, 0], :-1, coords[:, 1], coords[:, 2], coords[:, 3]]
        return SparseTensor(
            coords=coords.int(),
            feats=feats,
        )

    def _get_env_float_list(self, name, default=""):
        raw = os.environ.get(name, default)
        if not raw:
            return []
        values = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))
        return values

    def _get_env_int_list(self, name, default=""):
        raw = os.environ.get(name, default)
        if not raw:
            return []
        values = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            values.append(int(item))
        return values

    def _decode_gaussians_with_retry(self, reconstruction_dense):
        occ_candidates = [
            float(os.environ.get("OBJECTX_INFER_OCC_THRESHOLD", "0.1"))
        ] + self._get_env_float_list(
            "OBJECTX_INFER_OCC_THRESHOLD_FALLBACKS", "0.15,0.2,0.25,0.3"
        )
        max_candidates = [int(os.environ.get("OBJECTX_INFER_MAX_VOXELS", "0"))] + (
            self._get_env_int_list(
                "OBJECTX_INFER_MAX_VOXELS_FALLBACKS", "200000,150000,120000,80000"
            )
        )

        tried = set()
        last_exc = None
        for occ in occ_candidates:
            for max_voxels in max_candidates:
                key = (round(float(occ), 6), int(max_voxels))
                if key in tried:
                    continue
                tried.add(key)
                reconstruction_sparse = self._sparsify(
                    reconstruction_dense,
                    occupancy_threshold=occ,
                    max_voxels=max_voxels,
                )
                try:
                    _LOGGER.info(
                        "Decoding gaussians with occ_threshold=%s max_voxels=%s sparse_coords=%s",
                        occ,
                        max_voxels,
                        reconstruction_sparse.coords.shape[0],
                    )
                    return self.latent_autoencoder.decode(reconstruction_sparse)
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    last_exc = exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _LOGGER.warning(
                        "Gaussian decode retry failed with occ_threshold=%s max_voxels=%s: %s",
                        occ,
                        max_voxels,
                        exc,
                    )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No gaussian decode attempt was made.")

    def inference(self, idx, object_id=None):
        with torch.no_grad():
            _LOGGER.info(f"Getting item: {idx}")
            data_dict = self.dataset.collate_fn([self.dataset[idx]])
            scene_graphs = data_dict["scene_graphs"]
            tot_obj_splat = (
                scene_graphs["tot_obj_splat"].to("cuda")
                if torch.cuda.is_available()
                else scene_graphs["tot_obj_splat"]
            )
            # Stage 1.
            if (
                torch.cuda.is_available()
                and self.max_objects_per_chunk > 0
                and tot_obj_splat.shape[0] > self.max_objects_per_chunk
            ):
                sparse_chunks = []
                total_objects = tot_obj_splat.shape[0]
                for start in range(0, total_objects, self.max_objects_per_chunk):
                    end = min(start + self.max_objects_per_chunk, total_objects)
                    chunk = tot_obj_splat[start:end]
                    _LOGGER.info(
                        "Encoding object chunk %s:%s/%s for %s",
                        start,
                        end,
                        total_objects,
                        scene_graphs["scene_ids"][0],
                    )
                    sparse_chunks.append(
                        self.latent_autoencoder.encode(
                            {"scene_graphs": {"tot_obj_splat": chunk}}
                        )
                    )
                    torch.cuda.empty_cache()
                sparse_splat = sparse_cat(sparse_chunks, dim=0)
            else:
                sparse_splat = self.latent_autoencoder.encode(
                    {"scene_graphs": {"tot_obj_splat": tot_obj_splat}}
                )

            # Stage 2.
            stage2_scene_graphs = {
                "batch_size": scene_graphs["batch_size"],
                "tot_obj_dense_splat": self._densify(sparse_splat),
            }
            embedding = self.model.encode({"scene_graphs": stage2_scene_graphs})
            # Stage 2.
            if self.vis:
                reconstruction_dense = self.model.decode(embedding)
                reconstruction = self._decode_gaussians_with_retry(
                    reconstruction_dense
                )
            else:
                reconstruction = None

            means, scales, obj_ids, scan_id = (
                scene_graphs["mean_obj_splat"],
                scene_graphs["scale_obj_splat"],
                scene_graphs["obj_ids"],
                scene_graphs["scene_ids"][0],
            )

        self.save_embedding(embedding, means, scales, obj_ids, scan_id)

        if self.vis:
            if object_id is not None:
                object_id = torch.tensor(object_id)
                idx = torch.where(
                    torch.isin(
                        torch.from_numpy(data_dict["scene_graphs"]["obj_ids"]),
                        object_id,
                    )
                )[0]
                if len(idx) == 0:
                    _LOGGER.warning(f"Object {object_id} not found in the scene.")
                    return
            else:
                _LOGGER.info(f"Saving all objects in the scene.")
                idx = torch.arange(len(reconstruction))
            self.save_scene(
                [copy.deepcopy(reconstruction[i]) for i in idx],
                scan_id,
                means[idx],
                scales[idx],
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                self.save_render_orbit(
                    [copy.deepcopy(reconstruction[i]) for i in idx],
                    scan_id,
                    means[idx],
                    scales[idx],
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                _LOGGER.warning(
                    f"Skipping orbit rendering for {scan_id[0]} due to GPU error: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if self._get_env_bool("OBJECTX_VIS_SKIP_GS", False):
                _LOGGER.info(
                    f"Skipping rendered_gs output for {scan_id[0]} "
                    "because OBJECTX_VIS_SKIP_GS=1"
                )
            else:
                try:
                    self.save_render_orbit_gs(
                        scan_id, obj_ids[idx], means[idx], scales[idx]
                    )
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    _LOGGER.warning(
                        f"Skipping rendered_gs output for {scan_id[0]} due to GPU error: {exc}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _get_env_bool(self, name, default=False):
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() not in {"0", "false", "no", "off", ""}

    def _get_bg_color(self) -> torch.Tensor:
        raw = (os.environ.get("OBJECTX_VIS_BG_COLOR") or "0,0,0").strip()
        try:
            parts = [float(x) for x in raw.split(",")]
            if len(parts) != 3:
                raise ValueError
        except ValueError:
            _LOGGER.warning(
                "Invalid OBJECTX_VIS_BG_COLOR=%s, falling back to black", raw
            )
            parts = [0.0, 0.0, 0.0]
        return torch.tensor(parts, device="cuda")

    def _get_env_float(self, name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            _LOGGER.warning("Invalid %s=%s, using default %s", name, raw, default)
            return default

    def _get_env_int(self, name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            _LOGGER.warning("Invalid %s=%s, using default %s", name, raw, default)
            return default

    def _prune_representation(self, representation, label: str):
        opacity_min = self._get_env_float("OBJECTX_VIS_PRUNE_OPACITY_MIN", 0.0)
        opacity_quantile = self._get_env_float(
            "OBJECTX_VIS_PRUNE_OPACITY_QUANTILE", 0.0
        )
        scale_max = self._get_env_float("OBJECTX_VIS_PRUNE_SCALE_MAX", 0.0)
        scale_quantile = self._get_env_float("OBJECTX_VIS_PRUNE_SCALE_QUANTILE", 1.0)
        max_points = self._get_env_int("OBJECTX_VIS_PRUNE_MAX_POINTS", 0)

        is_gaussian_model = hasattr(representation, "_xyz")
        xyz = representation._xyz if is_gaussian_model else representation.xyz
        if xyz.numel() == 0:
            return representation

        opacity = (
            representation.get_opacity.reshape(-1)
            if is_gaussian_model
            else representation.get_opacity.reshape(-1)
        )
        scaling = (
            representation.get_scaling.max(dim=1).values
            if is_gaussian_model
            else representation.get_scaling.max(dim=1).values
        )
        mask = torch.ones_like(opacity, dtype=torch.bool)

        if opacity_min > 0.0:
            mask &= opacity >= opacity_min
        if 0.0 < opacity_quantile < 1.0 and opacity.numel() > 1:
            opacity_thr = torch.quantile(opacity, opacity_quantile)
            mask &= opacity >= opacity_thr
        if scale_max > 0.0:
            mask &= scaling <= scale_max
        if 0.0 < scale_quantile < 1.0 and scaling.numel() > 1:
            scale_thr = torch.quantile(scaling, scale_quantile)
            mask &= scaling <= scale_thr

        keep = int(mask.sum().item())
        total = int(mask.numel())
        if keep == 0:
            _LOGGER.warning(
                "Gaussian prune for %s would remove everything; skipping prune", label
            )
            return representation

        if max_points > 0 and keep > max_points:
            scores = opacity[mask]
            topk = torch.topk(scores, k=max_points, largest=True, sorted=False).indices
            full_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)[topk]
            new_mask = torch.zeros_like(mask)
            new_mask[full_idx] = True
            mask = new_mask
            keep = int(mask.sum().item())

        if keep == total:
            return representation

        _LOGGER.info(
            "Pruned gaussians for %s: kept %s/%s (opacity_min=%.3f opacity_quantile=%.3f scale_max=%.4f scale_quantile=%.3f max_points=%s)",
            label,
            keep,
            total,
            opacity_min,
            opacity_quantile,
            scale_max,
            scale_quantile,
            max_points,
        )

        if is_gaussian_model:
            representation._xyz = representation._xyz[mask]
            representation._features_dc = representation._features_dc[mask]
            representation._opacity = representation._opacity[mask]
            representation._scaling = representation._scaling[mask]
            representation._rotation = representation._rotation[mask]
        else:
            representation.xyz = representation.xyz[mask]
            representation.features_dc = representation.features_dc[mask]
            representation.features_rest = representation.features_rest[mask]
            representation.opacity = representation.opacity[mask]
            representation.scaling = representation.scaling[mask]
            representation.rotation = representation.rotation[mask]

        return representation

    def _postprocess_render(self, rendered_image: torch.Tensor) -> torch.Tensor:
        exposure = float(os.environ.get("OBJECTX_VIS_EXPOSURE", "1.0"))
        gamma = float(os.environ.get("OBJECTX_VIS_GAMMA", "1.0"))
        black_floor = float(os.environ.get("OBJECTX_VIS_BLACK_FLOOR", "0.0"))
        image = rendered_image.clamp(0.0, 1.0)
        if exposure != 1.0:
            image = (image * exposure).clamp(0.0, 1.0)
        if black_floor > 0.0:
            image = torch.maximum(
                image,
                torch.full_like(image, min(max(black_floor, 0.0), 1.0)),
            )
        if gamma != 1.0:
            gamma = max(gamma, 1e-4)
            image = image.pow(gamma).clamp(0.0, 1.0)
        return image

    def _get_orbit_mode(self, num_objects: int) -> str:
        mode = (os.environ.get("OBJECTX_VIS_ORBIT_MODE") or "auto").strip().lower()
        if mode in {"legacy", "object", "auto"}:
            if mode != "auto":
                return mode
        return "object" if num_objects <= 5 else "legacy"

    def _compute_orbit_camera_params(
        self,
        object_positions: np.ndarray,
        *,
        num_objects: int,
        fovx: float,
        fovy: float,
    ) -> tuple[np.ndarray, float, float]:
        bbox_min = object_positions.min(axis=0)
        bbox_max = object_positions.max(axis=0)
        scene_center = (bbox_min + bbox_max) / 2.0
        bbox_extent = np.maximum(bbox_max - bbox_min, 1e-3)
        bbox_diag = float(np.linalg.norm(bbox_extent))
        mode = self._get_orbit_mode(num_objects)

        if mode == "legacy":
            radius = float(os.environ.get("OBJECTX_VIS_RADIUS", "2.0"))
            height = float(os.environ.get("OBJECTX_VIS_HEIGHT", "1.0"))
            return scene_center, radius, height

        fit_margin = float(os.environ.get("OBJECTX_VIS_FIT_MARGIN", "1.8"))
        min_radius = float(os.environ.get("OBJECTX_VIS_MIN_RADIUS", "0.35"))
        min_height = float(os.environ.get("OBJECTX_VIS_MIN_HEIGHT", "0.1"))
        vertical_lift = float(os.environ.get("OBJECTX_VIS_VERTICAL_LIFT", "0.15"))

        # Fit the orbit distance to the object extent and field of view.
        safe_half_fov = max(0.15, min(float(fovx), float(fovy)) / 2.0)
        max_half_extent = float(np.max(bbox_extent)) / 2.0
        radius = max(
            min_radius,
            fit_margin * max_half_extent / np.tan(safe_half_fov),
            0.6 * bbox_diag,
        )
        height = max(min_height, vertical_lift * bbox_extent[1] + 0.1 * bbox_diag)
        return scene_center, radius, height

    def save_embedding(self, embedding, means, scales, obj_ids, scan_id):
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = osp.join(self.output_dir, f"{scan_id[0]}_ulat.npz")
        _LOGGER.info(f"Saving to {output_path}")
        np.savez(
            output_path,
            embedding=embedding.cpu(),
            mean=means.cpu(),
            scale=scales.cpu(),
            obj_id=obj_ids,
        )

    def save_scene(self, reconstruction, scan_id, means, scales):
        os.makedirs("vis", exist_ok=True)
        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                return None
            device = reconstruction[i].get_xyz.device
            dtype = reconstruction[i].get_xyz.dtype
            scale = torch.as_tensor(scales[i], device=device, dtype=dtype)
            mean = torch.as_tensor(means[i], device=device, dtype=dtype)
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 1 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(
                torch.tensor([2, 2, 2], device=device, dtype=dtype)
            )
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 2.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.translate(
                -torch.tensor([1, 1, 1], device=device, dtype=dtype)
            )
            assert (
                splat._xyz.min() >= -1.0 - 1e-2 and splat._xyz.max() <= 1.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(scale)
            splat.translate(mean)
            return splat

        reconstruction = list(map(lambda x: _scale(x), enumerate(reconstruction)))
        representation = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )

        representation._xyz = torch.concatenate(
            [splat._xyz for splat in reconstruction if splat is not None]
        )
        representation._features_dc = torch.concatenate(
            [splat._features_dc for splat in reconstruction if splat is not None]
        )
        representation._opacity = torch.concatenate(
            [splat._opacity for splat in reconstruction if splat is not None]
        )
        representation._scaling = torch.concatenate(
            [splat._scaling for splat in reconstruction if splat is not None]
        )
        representation._rotation = torch.concatenate(
            [splat._rotation for splat in reconstruction if splat is not None]
        )
        representation = self._prune_representation(
            representation, f"{scan_id[0]}_scene"
        )
        representation.save_ply(f"vis/{scan_id[0]}_joint.ply")

    def save_render(self, reconstruction, scene_ids, means, scales):
        if not osp.exists("vis/rendered"):
            os.makedirs("vis/rendered")

        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                return splat
            device = reconstruction[i].get_xyz.device
            dtype = reconstruction[i].get_xyz.dtype
            scale = torch.as_tensor(scales[i], device=device, dtype=dtype)
            mean = torch.as_tensor(means[i], device=device, dtype=dtype)
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 1 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(
                torch.tensor([2, 2, 2], device=device, dtype=dtype)
            )
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 2.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.translate(
                -torch.tensor([1, 1, 1], device=device, dtype=dtype)
            )
            assert (
                splat._xyz.min() >= -1.0 - 1e-2 and splat._xyz.max() <= 1.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(scale)
            splat.translate(mean)
            return splat

        reconstruction = list(map(lambda x: _scale(x), enumerate(reconstruction)))
        representation = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )
        representation._xyz = torch.concatenate(
            [reconstruction[i]._xyz for i in range(len(reconstruction))]
        )
        representation._features_dc = torch.concatenate(
            [reconstruction[i]._features_dc for i in range(len(reconstruction))]
        )
        representation._opacity = torch.concatenate(
            [reconstruction[i]._opacity for i in range(len(reconstruction))]
        )
        representation._scaling = torch.concatenate(
            [reconstruction[i]._scaling for i in range(len(reconstruction))]
        )
        representation._rotation = torch.concatenate(
            [reconstruction[i]._rotation for i in range(len(reconstruction))]
        )
        predicted_images = []
        ground_truth_images = []
        rendered_frames = []
        bg_color = self._get_bg_color()
        scene_id = scene_ids[0]
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = self.dataset.image_poses[scene_id]

        gs_mesh = mesh.splat_to_mesh(
            splat=copy.deepcopy(representation).to_pt(),
            Ks=intrinsics["intrinsic_mat"],
            world_to_cams=poses,
            width=intrinsics["width"],
            height=intrinsics["height"],
            sh_degree_to_use=0,
            near_plane=0.01,
            far_plane=100.0,
        )

        for i, frame_id in enumerate(poses):
            extrinsics = poses[frame_id]
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            image = (
                torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # SHape: (3, H, W)

            pose_camera_to_world = np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))
            viewpoint_camera = build_minicam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                znear=0.01,
                zfar=100.0,
                R=pose_camera_to_world[:3, :3].T,
                T=pose_camera_to_world[:3, 3],
            )

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=bg_color,
            )["render"]
            rendered_image = self._postprocess_render(rendered_image)

            predicted_images.append(rendered_image)
            ground_truth_images.append(image)

            frame_path = f"vis/rendered/{scene_id}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            del image
            torch.cuda.empty_cache()

        video_path = f"vis/rendered/{scene_id}_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)

    def save_render_orbit(
        self, reconstruction, scene_ids, means, scales, output_dir="vis/rendered"
    ):
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene_ids[0]

        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                print(f"Splat {i} is empty.")
                return splat
            device = reconstruction[i].get_xyz.device
            dtype = reconstruction[i].get_xyz.dtype
            scale = torch.as_tensor(scales[i], device=device, dtype=dtype)
            mean = torch.as_tensor(means[i], device=device, dtype=dtype)
            splat.rescale(
                torch.tensor([2, 2, 2], device=device, dtype=dtype)
            )
            splat.translate(
                -torch.tensor([1, 1, 1], device=device, dtype=dtype)
            )
            splat.rescale(scale)
            splat.translate(mean)
            return splat

        # Apply transformation to all objects
        reconstruction = list(map(lambda x: _scale(x), enumerate(reconstruction)))

        # Create the representation
        representation = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )

        representation._xyz = torch.cat(
            [reconstruction[i]._xyz for i in range(len(reconstruction))]
        )
        representation._features_dc = torch.cat(
            [reconstruction[i]._features_dc for i in range(len(reconstruction))]
        )
        representation._opacity = torch.cat(
            [reconstruction[i]._opacity for i in range(len(reconstruction))]
        )
        representation._scaling = torch.cat(
            [reconstruction[i]._scaling for i in range(len(reconstruction))]
        )
        representation._rotation = torch.cat(
            [reconstruction[i]._rotation for i in range(len(reconstruction))]
        )
        representation = self._prune_representation(
            representation, f"{scene_id}_render"
        )

        object_positions = representation._xyz.detach().cpu().numpy()

        num_frames = int(os.environ.get("OBJECTX_VIS_NUM_FRAMES", "120"))
        angle_step = 2 * np.pi / num_frames  # Step size for rotation

        rendered_frames = []
        bg_color = self._get_bg_color()

        # Get intrinsics for frustum visualization
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = np.stack(list(self.dataset.image_poses[scene_id].values()))
        render_scale = float(os.environ.get("OBJECTX_VIS_RENDER_SCALE", "1.0"))
        render_width = max(64, int(round(intrinsics["width"] * render_scale)))
        render_height = max(64, int(round(intrinsics["height"] * render_scale)))
        fovy = focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"])
        fovx = focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"])
        scene_center, radius, height = self._compute_orbit_camera_params(
            object_positions,
            num_objects=len(reconstruction),
            fovx=fovx,
            fovy=fovy,
        )

        if self._get_env_bool("OBJECTX_VIS_EXPORT_MESH", True):
            gs_mesh = mesh.splat_to_mesh(
                splat=copy.deepcopy(representation).to_pt(),
                Ks=intrinsics["intrinsic_mat"],
                world_to_cams=poses,
                width=render_width,
                height=render_height,
                sh_degree_to_use=0,
                near_plane=0.01,
                far_plane=100.0,
            )
            o3d.io.write_triangle_mesh(f"{output_dir}/{scene_id}_mesh.ply", gs_mesh)

        for i in range(num_frames):
            theta = i * angle_step
            # Camera position: Move in a circular orbit around the Y-axis
            cam_x = scene_center[0] + radius * np.cos(theta)  # Orbit on XZ plane
            cam_y = scene_center[2] + radius * np.sin(theta)  # Orbit on XZ plane
            cam_z = scene_center[1] + height  # Keep at a fixed height

            # Camera should always look at the scene center
            cam_position = np.array([cam_x, cam_y, cam_z])  # Camera position
            forward = cam_position - scene_center  # Look at scene center
            forward /= np.linalg.norm(forward)  # Normalize forward vector

            # Define world UP direction
            up = np.array([0, 1, 0])  # Fixed Y-up

            # Compute RIGHT vector (perpendicular to FORWARD and UP)
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)  # Normalize right vector

            # Recompute UP to maintain perfect orthogonality
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)  # Normalize up vector

            # Construct the camera rotation matrix
            R = np.stack([right, up, -forward], axis=1)  # Rotation matrix
            T = cam_position  # Camera position

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            viewmat = np.linalg.inv(viewmat)

            # Create camera instance
            viewpoint_camera = build_minicam(
                width=render_width,
                height=render_height,
                fovy=fovy,
                fovx=fovx,
                znear=0.01,
                zfar=100.0,
                R=viewmat[:3, :3].T,
                T=viewmat[:3, 3],
            )

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=bg_color,
            )["render"]
            rendered_image = self._postprocess_render(rendered_image)

            frame_path = f"{output_dir}/{scene_ids[0]}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            torch.cuda.empty_cache()

        # Save video
        video_path = f"{output_dir}/{scene_ids[0]}_orbit_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)
        print(f"Video saved at {video_path}")

    def save_render_orbit_gs(
        self, scene_ids, obj_ids, means, scales, output_dir="vis/rendered_gs"
    ):
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene_ids[0]
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.reshape(-1).tolist()
        elif not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        obj_ids = [int(obj_id) for obj_id in obj_ids]

        # load gaussian splat representation
        reconstruction = []
        for obj_id in obj_ids:
            ply_path = (
                f"{self.cfg.data.root_dir}/files/gs_annotations/{scene_id}/{obj_id}/"
                "point_cloud/iteration_7000/point_cloud.ply"
            )
            if not osp.exists(ply_path):
                _LOGGER.warning(f"Missing gs annotation ply: {ply_path}")
                continue
            reconstruction.append(GaussianSplat.load_ply(ply_path).to_torch())

        if len(reconstruction) == 0:
            _LOGGER.warning(
                f"No gs_annotations found for scene {scene_id}; skipping rendered_gs output."
            )
            return

        # Create the representation
        representation = GaussianSplat(
            xyz=torch.cat([splat.xyz for splat in reconstruction]),
            features_dc=torch.cat([splat.features_dc for splat in reconstruction]),
            opacity=torch.cat([splat.opacity for splat in reconstruction]),
            scaling=torch.cat([splat.scaling for splat in reconstruction]),
            rotation=torch.cat([splat.rotation for splat in reconstruction]),
        )
        representation = self._prune_representation(
            representation, f"{scene_id}_rendered_gs"
        )

        object_positions = representation.xyz.detach().cpu().numpy()

        num_frames = int(os.environ.get("OBJECTX_VIS_NUM_FRAMES", "120"))
        angle_step = 2 * np.pi / num_frames  # Step size for rotation

        rendered_frames = []
        bg_color = self._get_bg_color()

        # Get intrinsics for frustum visualization
        intrinsics = self.dataset.image_intrinsics[scene_id]
        poses = np.stack(list(self.dataset.image_poses[scene_id].values()))

        gs_mesh = mesh.splat_to_mesh(
            splat=copy.deepcopy(representation).to_pt(),
            Ks=intrinsics["intrinsic_mat"],
            world_to_cams=poses,
            width=intrinsics["width"],
            height=intrinsics["height"],
            sh_degree_to_use=0,
            near_plane=0.01,
            far_plane=100.0,
        )
        o3d.io.write_triangle_mesh(f"{output_dir}/{scene_id}_mesh.ply", gs_mesh)
        fovy = focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"])
        fovx = focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"])
        scene_center, radius, height = self._compute_orbit_camera_params(
            object_positions,
            num_objects=len(reconstruction),
            fovx=fovx,
            fovy=fovy,
        )

        for i in range(num_frames):
            theta = i * angle_step
            # Camera position: Move in a circular orbit around the Y-axis
            cam_x = scene_center[0] + radius * np.cos(theta)  # Orbit on XZ plane
            cam_y = scene_center[2] + radius * np.sin(theta)  # Orbit on XZ plane
            cam_z = scene_center[1] + height  # Keep at a fixed height

            # Camera should always look at the scene center
            cam_position = np.array([cam_x, cam_y, cam_z])  # Camera position
            forward = cam_position - scene_center  # Look at scene center
            forward /= np.linalg.norm(forward)  # Normalize forward vector

            # Define world UP direction
            up = np.array([0, 1, 0])  # Fixed Y-up

            # Compute RIGHT vector (perpendicular to FORWARD and UP)
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)  # Normalize right vector

            # Recompute UP to maintain perfect orthogonality
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)  # Normalize up vector

            # Construct the camera rotation matrix
            R = np.stack([right, up, -forward], axis=1)  # Rotation matrix
            T = cam_position  # Camera position

            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            viewmat = np.linalg.inv(viewmat)

            # Create camera instance
            viewpoint_camera = build_minicam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                znear=0.01,
                zfar=100.0,
                R=viewmat[:3, :3].T,
                T=viewmat[:3, 3],
            )

            rendered_image = render(
                viewpoint_camera,
                representation,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=bg_color,
            )["render"]
            rendered_image = self._postprocess_render(rendered_image)

            frame_path = f"{output_dir}/{scene_ids[0]}_frame_{i:03d}.png"
            save_image(rendered_image, frame_path)
            rendered_frames.append(
                (rendered_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    np.uint8
                )
            )

            torch.cuda.empty_cache()

        # Save video
        video_path = f"{output_dir}/{scene_ids[0]}_orbit_rendered.mp4"
        imageio.mimsave(video_path, rendered_frames, fps=30)
        print(f"Video saved at {video_path}")

    def run(self, scene_id=None, object_id=None):
        if scene_id is not None:
            self.inference(scene_id, object_id)
        else:
            for idx in range(len(self.dataset)):
                self.inference(idx)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train structured latent inference model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results."
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for inference."
    )
    parser.add_argument("--scene_id", type=str, default=None, help="Specific scene.")
    parser.add_argument(
        "--objects_id", type=int, nargs="+", default=None, help="Specific object."
    )
    return parser.parse_known_args()


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args, do_ensure_dir=False)
    pipeline = SceneGraph2UnstructuredLatentPipeline(
        cfg, visualize=args.visualize, split=args.split
    )
    pipeline.run(args.scene_id, args.objects_id)


if __name__ == "__main__":
    main()
