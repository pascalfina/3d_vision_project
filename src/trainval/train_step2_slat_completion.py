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
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accuracy-tol-factor", type=float, default=0.25)
    parser.add_argument("--train-augment", action="store_true")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--rot90-prob", type=float, default=0.5)
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--observed-dropout-prob", type=float, default=0.05)
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


def _summarize_epoch_metrics(
    *,
    sq_err_sum: float,
    target_sq_sum: float,
    within_tol_sum: float,
    feature_count: float,
    sample_count: int,
    missing_voxel_sum: float,
    accuracy_tol_factor: float,
) -> dict[str, float]:
    if feature_count <= 0:
        return {
            "loss": 0.0,
            "rmse": 0.0,
            "nrmse_pct": 0.0,
            "baseline_loss": 0.0,
            "improvement_pct": 0.0,
            "voxel_acc_pct": 0.0,
            "missing_voxels": 0.0,
            "feature_count": 0.0,
            "accuracy_tol_factor": accuracy_tol_factor,
        }

    mse = sq_err_sum / feature_count
    rmse = math.sqrt(mse)
    baseline_mse = target_sq_sum / feature_count
    baseline_rmse = math.sqrt(baseline_mse)
    improvement_pct = 100.0 * (baseline_mse - mse) / max(baseline_mse, 1e-8)
    voxel_acc_pct = 100.0 * within_tol_sum / feature_count

    return {
        "loss": mse,
        "rmse": rmse,
        "nrmse_pct": 100.0 * rmse / max(baseline_rmse, 1e-8),
        "baseline_loss": baseline_mse,
        "improvement_pct": improvement_pct,
        "voxel_acc_pct": voxel_acc_pct,
        "missing_voxels": missing_voxel_sum / max(sample_count, 1),
        "feature_count": feature_count / max(sample_count, 1),
        "accuracy_tol_factor": accuracy_tol_factor,
    }


def run_epoch(
    model: SmallSLATCompletionCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    accuracy_tol_factor: float,
    grad_accum_steps: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_sq_err = 0.0
    total_target_sq = 0.0
    total_within_tol = 0.0
    total_feature_count = 0.0
    total_missing_voxels = 0.0
    total_samples = 0

    autocast_enabled = amp and device.type == "cuda"
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader, start=1):
        input_feats = batch["input_feats"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        target_feats = batch["target_feats"].to(device)
        missing_mask = batch["missing_mask"].to(device)

        model_input = torch.cat([input_feats, observed_mask], dim=1)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            predicted_unknown = model(model_input)
            final_prediction = (
                observed_mask * input_feats + (1.0 - observed_mask) * predicted_unknown
            )
            loss = masked_mse_loss(final_prediction, target_feats, missing_mask)

        if is_train:
            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(loader))
            if scaler is not None and autocast_enabled:
                scaler.scale(loss / grad_accum_steps).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                (loss / grad_accum_steps).backward()
                if should_step:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        missing_feature_count = float(
            (missing_mask.sum() * final_prediction.shape[1]).detach().cpu()
        )
        if missing_feature_count > 0:
            sq_err = (((final_prediction - target_feats) ** 2) * missing_mask).sum()
            target_sq = ((target_feats**2) * missing_mask).sum()
            target_rms = torch.sqrt(target_sq / missing_feature_count)
            tol = accuracy_tol_factor * target_rms
            within_tol = (
                (((final_prediction - target_feats).abs() <= tol).float() * missing_mask)
                .sum()
            )

            total_sq_err += float(sq_err.detach().cpu())
            total_target_sq += float(target_sq.detach().cpu())
            total_within_tol += float(within_tol.detach().cpu())
            total_feature_count += missing_feature_count

        total_missing_voxels += float(missing_mask.sum().detach().cpu())
        total_samples += input_feats.shape[0]

    return _summarize_epoch_metrics(
        sq_err_sum=total_sq_err,
        target_sq_sum=total_target_sq,
        within_tol_sum=total_within_tol,
        feature_count=total_feature_count,
        sample_count=total_samples,
        missing_voxel_sum=total_missing_voxels,
        accuracy_tol_factor=accuracy_tol_factor,
    )


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


def save_history_plots(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _LOGGER.warning("Could not save plots because matplotlib is unavailable: %s", exc)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [record["epoch"] for record in history]

    def _plot_series(
        series: list[tuple[str, list[float]]],
        title: str,
        ylabel: str,
        path: Path,
    ) -> None:
        plt.figure(figsize=(8, 5))
        for label, values in series:
            plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if len(series) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    _plot_series(
        [
            ("train_loss", [record["train_loss"] for record in history]),
            ("val_loss", [record["val_loss"] for record in history]),
        ],
        title="Train vs Val Loss",
        ylabel="Masked MSE",
        path=plots_dir / "loss_curve.png",
    )

    if "train_rmse" in history[0] and "val_rmse" in history[0]:
        _plot_series(
            [
                ("train_rmse", [record["train_rmse"] for record in history]),
                ("val_rmse", [record["val_rmse"] for record in history]),
            ],
            title="Train vs Val RMSE",
            ylabel="RMSE",
            path=plots_dir / "rmse_curve.png",
        )

    percent_keys = [
        ("val_nrmse_pct", "Val NRMSE (%)"),
        ("val_improvement_pct", "Val Improvement (%)"),
        ("val_voxel_acc_pct", "Val Voxel Accuracy (%)"),
    ]
    available_percent_keys = [
        (label, [record[key] for record in history])
        for key, label in percent_keys
        if key in history[0]
    ]
    if available_percent_keys:
        _plot_series(
            available_percent_keys,
            title="Validation Percent Metrics",
            ylabel="Percent",
            path=plots_dir / "val_percent_metrics.png",
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
        augment=args.train_augment,
        flip_prob=args.flip_prob,
        rot90_prob=args.rot90_prob,
        feature_noise_std=args.feature_noise_std,
        observed_dropout_prob=args.observed_dropout_prob,
    )
    val_dataset = Step2SLATPairDataset(
        manifest_path=args.manifest,
        split="val",
        grid_size=args.grid_size,
        require_missing_voxels=True,
    )
    test_dataset = Step2SLATPairDataset(
        manifest_path=args.manifest,
        split="test",
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
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
    _LOGGER.info("Test samples: %s", len(test_dataset))
    _LOGGER.info(
        "Augment train=%s flip_prob=%.2f rot90_prob=%.2f feature_noise_std=%.4f observed_dropout_prob=%.4f",
        args.train_augment,
        args.flip_prob,
        args.rot90_prob,
        args.feature_noise_std,
        args.observed_dropout_prob,
    )
    _LOGGER.info(
        "Optimization lr=%.6f weight_decay=%.6f grad_accum_steps=%s grad_clip_norm=%.2f",
        args.lr,
        args.weight_decay,
        args.grad_accum_steps,
        args.grad_clip_norm,
    )
    _LOGGER.info(
        "Model parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=args.amp,
            accuracy_tol_factor=args.accuracy_tol_factor,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip_norm=args.grad_clip_norm,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                scaler=None,
                amp=args.amp,
                accuracy_tol_factor=args.accuracy_tol_factor,
                grad_accum_steps=1,
                grad_clip_norm=0.0,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_nrmse_pct": train_metrics["nrmse_pct"],
            "train_improvement_pct": train_metrics["improvement_pct"],
            "train_voxel_acc_pct": train_metrics["voxel_acc_pct"],
            "train_missing_voxels": train_metrics["missing_voxels"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_nrmse_pct": val_metrics["nrmse_pct"],
            "val_improvement_pct": val_metrics["improvement_pct"],
            "val_voxel_acc_pct": val_metrics["voxel_acc_pct"],
            "val_missing_voxels": val_metrics["missing_voxels"],
        }
        history.append(record)
        _LOGGER.info(
            "Epoch %s | train_loss=%.6f | val_loss=%.6f | val_nrmse=%.2f%% | val_improvement=%.2f%% | val_voxel_acc=%.2f%% | train_missing_voxels=%.1f | val_missing_voxels=%.1f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["nrmse_pct"],
            val_metrics["improvement_pct"],
            val_metrics["voxel_acc_pct"],
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
            best_epoch = epoch
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
    save_history_plots(history, output_dir)
    _LOGGER.info("Saved training plots to %s", output_dir / "plots")

    best_checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
            scaler=None,
            amp=args.amp,
            accuracy_tol_factor=args.accuracy_tol_factor,
            grad_accum_steps=1,
            grad_clip_norm=0.0,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": history[-1]["epoch"] if history else 0,
        "last_train_loss": history[-1]["train_loss"] if history else 0.0,
        "last_val_loss": history[-1]["val_loss"] if history else 0.0,
        "test_metrics": test_metrics,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _LOGGER.info(
        "Best checkpoint test | test_loss=%.6f | test_nrmse=%.2f%% | test_improvement=%.2f%% | test_voxel_acc=%.2f%%",
        test_metrics["loss"],
        test_metrics["nrmse_pct"],
        test_metrics["improvement_pct"],
        test_metrics["voxel_acc_pct"],
    )
    _LOGGER.info("Saved test metrics to %s", output_dir / "test_metrics.json")
    _LOGGER.info("Saved summary to %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
