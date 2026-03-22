from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.step2_slat_pairs import Step2SLATPairDataset
from src.models.slat_completion_cnn import SmallSLATCompletionCNN
from utils import common

_LOGGER = logging.getLogger(__name__)


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text())
    raise ValueError(
        f"Unsupported config format for {config_path}. Use a .json config."
    )


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument(
        "--manifest",
        default="/work/scratch/pafina/objectx-data-sparse10-keep75pct/files/step2_slat_pairs_keep75pct_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.set_defaults(**config)
    args = parser.parse_args(remaining)
    if args.output_dir is None:
        parser.error("--output-dir is required (either directly or via --config).")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = ((prediction - target) ** 2) * mask
    denom = mask.sum() * prediction.shape[1]
    if denom.item() <= 0:
        return prediction.new_zeros(())
    return loss.sum() / denom


def run_epoch(
    model: SmallSLATCompletionCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_missing_voxels = 0.0
    total_samples = 0

    autocast_enabled = amp and device.type == "cuda"
    for batch in loader:
        input_feats = batch["input_feats"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        target_feats = batch["target_feats"].to(device)
        missing_mask = batch["missing_mask"].to(device)

        model_input = torch.cat([input_feats, observed_mask], dim=1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            predicted_unknown = model(model_input)
            final_prediction = (
                observed_mask * input_feats + (1.0 - observed_mask) * predicted_unknown
            )
            loss = masked_mse_loss(final_prediction, target_feats, missing_mask)

        if is_train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_missing_voxels += float(missing_mask.sum().detach().cpu())
        total_samples += input_feats.shape[0]

    return {
        "loss": total_loss / max(total_samples, 1),
        "missing_voxels": total_missing_voxels / max(total_samples, 1),
    }


def save_checkpoint(
    path: Path,
    model: SmallSLATCompletionCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    common.init_log(level=logging.INFO)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Step2SLATPairDataset(
        manifest_path=args.manifest,
        split="train",
        grid_size=args.grid_size,
        require_missing_voxels=True,
    )
    val_dataset = Step2SLATPairDataset(
        manifest_path=args.manifest,
        split="val",
        grid_size=args.grid_size,
        require_missing_voxels=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = SmallSLATCompletionCNN(
        in_channels=9,
        hidden_channels=args.hidden_channels,
        out_channels=8,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )

    _LOGGER.info("Train samples: %s", len(train_dataset))
    _LOGGER.info("Val samples: %s", len(val_dataset))
    _LOGGER.info(
        "Model parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                scaler=None,
                amp=args.amp,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_missing_voxels": train_metrics["missing_voxels"],
            "val_loss": val_metrics["loss"],
            "val_missing_voxels": val_metrics["missing_voxels"],
        }
        history.append(record)
        _LOGGER.info(
            "Epoch %s | train_loss=%.6f | val_loss=%.6f | train_missing_voxels=%.1f | val_missing_voxels=%.1f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["missing_voxels"],
            val_metrics["missing_voxels"],
        )

        save_checkpoint(
            output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    _LOGGER.info("Saved training history to %s", output_dir / "history.json")


if __name__ == "__main__":
    main()
