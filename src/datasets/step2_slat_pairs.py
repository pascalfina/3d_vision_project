from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils import data


class Step2SLATPairDataset(data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        grid_size: int = 64,
        require_missing_voxels: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.grid_size = grid_size
        self.require_missing_voxels = require_missing_voxels
        manifest = json.loads(self.manifest_path.read_text())
        self.samples = self._build_samples(manifest)

    def _build_samples(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for entry in manifest["entries"]:
            if entry["split"] != self.split:
                continue
            for obj_pair in entry["object_pairs"]:
                if self.require_missing_voxels and (
                    obj_pair["observed_voxels"] >= obj_pair["full_voxels"]
                ):
                    continue
                samples.append(
                    {
                        "scene_id": entry["scene_id"],
                        "split": entry["split"],
                        "obj_id": obj_pair["obj_id"],
                        "path": obj_pair["path"],
                        "full_voxels": obj_pair["full_voxels"],
                        "observed_voxels": obj_pair["observed_voxels"],
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _scatter_features(
        self, coords: np.ndarray, feats: np.ndarray, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        channels = feats.shape[1]
        dense = torch.zeros((channels, *shape), dtype=torch.float32)
        if coords.shape[0] == 0:
            return dense
        x = torch.from_numpy(coords[:, 0]).long()
        y = torch.from_numpy(coords[:, 1]).long()
        z = torch.from_numpy(coords[:, 2]).long()
        values = torch.from_numpy(feats).float().t().contiguous()
        dense[:, x, y, z] = values
        return dense

    def _scatter_mask(
        self, coords: np.ndarray, values: np.ndarray, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        dense = torch.zeros((1, *shape), dtype=torch.float32)
        if coords.shape[0] == 0:
            return dense
        x = torch.from_numpy(coords[:, 0]).long()
        y = torch.from_numpy(coords[:, 1]).long()
        z = torch.from_numpy(coords[:, 2]).long()
        dense[0, x, y, z] = torch.from_numpy(values).float()
        return dense

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pair = np.load(sample["path"], allow_pickle=True)
        shape = (self.grid_size, self.grid_size, self.grid_size)

        full_coords = np.asarray(pair["full_coords"], dtype=np.int16)
        full_feats = np.asarray(pair["full_feats"], dtype=np.float32)
        sparse_feats_on_full = np.asarray(pair["sparse_feats_on_full"], dtype=np.float32)
        observed_mask = np.asarray(pair["observed_mask"], dtype=np.uint8)

        target_feats = self._scatter_features(full_coords, full_feats, shape)
        sparse_feats = self._scatter_features(full_coords, sparse_feats_on_full, shape)
        observed_mask_dense = self._scatter_mask(full_coords, observed_mask, shape)
        full_support_dense = self._scatter_mask(
            full_coords, np.ones(len(full_coords), dtype=np.uint8), shape
        )
        missing_mask_dense = full_support_dense * (1.0 - observed_mask_dense)

        return {
            "scene_id": sample["scene_id"],
            "split": sample["split"],
            "obj_id": torch.tensor(sample["obj_id"], dtype=torch.long),
            "input_feats": sparse_feats,
            "observed_mask": observed_mask_dense,
            "target_feats": target_feats,
            "full_support_mask": full_support_dense,
            "missing_mask": missing_mask_dense,
            "full_mean": torch.from_numpy(np.asarray(pair["full_mean"], dtype=np.float32)),
            "full_scale": torch.as_tensor(pair["full_scale"], dtype=torch.float32),
            "sparse_mean": torch.from_numpy(
                np.asarray(pair["sparse_mean"], dtype=np.float32)
            ),
            "sparse_scale": torch.as_tensor(pair["sparse_scale"], dtype=torch.float32),
            "path": sample["path"],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys_to_stack = [
            "obj_id",
            "input_feats",
            "observed_mask",
            "target_feats",
            "full_support_mask",
            "missing_mask",
            "full_mean",
            "full_scale",
            "sparse_mean",
            "sparse_scale",
        ]
        out = {
            "scene_id": [sample["scene_id"] for sample in batch],
            "split": [sample["split"] for sample in batch],
            "path": [sample["path"] for sample in batch],
        }
        for key in keys_to_stack:
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
        return out
