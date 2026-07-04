from dataset import DepthDataset
from meter import MeterArchitecture
import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import globals
import logger
from eval import evaluate
from loss import balanced_loss_function
from hardware_acceleration import enable_hardware_acceleration, Config
from meter import GaussianMeter
from dataset import NormalizedNyuDataset, DepthTrainDataset

RUNS_DIR = Path("runs")

BATCH_SIZE = 128
EPOCHS = 60

LEARNING_RATE = 1e-3
LR_DECAY_EVERY = 20
LR_DECAY_GAMMA = 0.1
WEIGHT_DECAY = 1e-2


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train = DataLoader(
        DepthTrainDataset("train", augment=True, augmentation="meter", normalization="imagenet"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
        prefetch_factor=2,
        persistent_workers=True,
    )
    val = DataLoader(
        NormalizedNyuDataset("val", normalization="imagenet"),
        batch_size=16,
        shuffle=False,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
    )
    test = DataLoader(
        NormalizedNyuDataset("test", normalization="imagenet"),
        batch_size=16,
        shuffle=False,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
    )
    return train, val, test


def train_supervised(
    name: str,
    dataset: DepthDataset = "nyu",
    arch: MeterArchitecture = "xxs",
    enable_derf: bool = False,
    enable_gelu: bool = False,
    resume: bool = True,
):
    output_dir = RUNS_DIR / f"{dataset}_{arch}_{name}{'_derf' if enable_derf else '_ln'}{'_gelu' if enable_gelu else '_relu'}"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(Config.DEFAULT)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run directory: {output_dir}")

    train, val, test = get_dataloaders()
    raw_model = GaussianMeter(device, arch, enable_derf=enable_derf, enable_gelu=enable_gelu).to(device)

    param_groups = [{"params": raw_model.parameters(), "lr": LEARNING_RATE}]
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=LR_DECAY_EVERY, gamma=LR_DECAY_GAMMA)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    loss_fn = balanced_loss_function(device)

    history = {
        "train_loss": [],
        "loss_depth": [],
        "loss_ssim": [],
        "loss_normal": [],
        "loss_grad": [],
        "rmse": [],
        "rel": [],
        "delta1": [],
        "lr": [],
    }
    best_val_rmse = math.inf
    # epochs are 1-based: epoch is the ordinal number, not a zero index
    start_epoch = 1

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        history = {**history, **ckpt.get("history", {})}
        best_val_rmse = ckpt.get("best_val_rmse", best_val_rmse)
        start_epoch = int(ckpt["epoch"]) + 1

        rng = ckpt["rng_state"]
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
        if rng.get("mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng["mps"].cpu())

        logger.warn(f"resumed from {checkpoint_path} at epoch #{start_epoch}")

    if device.type == "cuda":
        logger.info("Enabling torch.compile for CUDA")
        model = torch.compile(raw_model)
    else:
        logger.warn(f"torch.compile is not supported/stable on {device.type}, skipping")
        model = raw_model

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        epoch_components = {"loss_depth": 0.0, "loss_ssim": 0.0, "loss_normal": 0.0, "loss_grad": 0.0}
        for batch in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["depth"].to(device, non_blocking=True)  # centimeters
                z = model(x)

            # loss in fp32, outside autocast: ssim computes E[x^2] - mu^2 at
            # centimeter scale (~1e5), which cancels catastrophically in bf16
            # and can hit a zero denominator -> (1 - inf) * 100 = -inf
            loss_depth, loss_ssim, loss_normal, loss_grad = loss_fn(z.float(), y)
            loss = loss_depth + loss_ssim + loss_normal + loss_grad
            if not torch.isfinite(loss):
                msg = f"Invalid loss {loss.item()} at epoch {epoch}"
                logger.error(msg)
                raise RuntimeError(msg)

            epoch_loss += loss.item()
            epoch_components["loss_depth"] += loss_depth.item()
            epoch_components["loss_ssim"] += loss_ssim.item()
            epoch_components["loss_normal"] += loss_normal.item()
            epoch_components["loss_grad"] += loss_grad.item()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # spike protection only: the METER loss works in centimeters, so
            # healthy gradient norms are ~50, far above lejepa.py's scale
            trainable = [p for p in raw_model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=500.0)
            scaler.step(opt)
            scaler.update()

        # paper decays the LR per epoch, not per step
        scheduler.step()

        train_loss = epoch_loss / len(train)
        # select on val only; the test set is held out for the final report
        metrics = evaluate(raw_model, val, device)
        logger.info(f"[{epoch}/{EPOCHS}] train/loss {train_loss:.4f}")
        logger.info(f"[{epoch}/{EPOCHS}] val/RMSE  {metrics['rmse']:.3f}m")
        logger.info(f"[{epoch}/{EPOCHS}] val/REL {metrics['rel']:.3f}")
        logger.info(f"[{epoch}/{EPOCHS}] val/d1 {metrics['delta1']:.3f}")

        history["train_loss"].append(train_loss)
        for component, total in epoch_components.items():
            history[component].append(total / len(train))
        history["rmse"].append(metrics["rmse"])
        history["rel"].append(metrics["rel"])
        history["delta1"].append(metrics["delta1"])
        history["lr"].append(scheduler.get_last_lr()[0])

        is_best = metrics["rmse"] < best_val_rmse
        best_val_rmse = min(best_val_rmse, metrics["rmse"])

        checkpoint = {
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "history": history,
            "best_val_rmse": best_val_rmse,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            },
        }

        # write to a temp file first so a crash never corrupts the last checkpoint
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"[{epoch}/{EPOCHS}] new best val/RMSE {best_val_rmse:.3f}m")

        # dumped every epoch so a crash never loses the chart data
        with open(output_dir / "losses.json", "w") as losses_file:
            json.dump(history, losses_file)

    # final, one-shot evaluation on the untouched test set, using the
    # val-selected best checkpoint (falls back to the last weights in memory)
    if best_checkpoint_path.exists():
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        raw_model.load_state_dict(best_ckpt["model_state"])
        logger.info(f"loaded best checkpoint (epoch {best_ckpt['epoch']}) for test eval")

    test_metrics = evaluate(raw_model, test, device)
    logger.info(f"[final] test/RMSE  {test_metrics['rmse']:.3f}m")
    logger.info(f"[final] test/REL {test_metrics['rel']:.3f}")
    logger.info(f"[final] test/d1 {test_metrics['delta1']:.3f}")

    with open(output_dir / "test_metrics.json", "w") as test_file:
        json.dump(test_metrics, test_file)


def main(name: str, enable_derf: bool = False, enable_gelu: bool = False, resume: bool = True):
    train_supervised(name, enable_derf=enable_derf, enable_gelu=enable_gelu, resume=resume)
    logger.info(
        f"Finished training {name} with {'DERF' if enable_derf else 'Layer Normalization'} and {'GELU' if enable_gelu else 'ReLU'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--derf", action="store_true")
    parser.add_argument("--gelu", action="store_true")
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="ignore any existing checkpoint and start fresh")
    args = parser.parse_args()
    raise SystemExit(main(args.name, args.derf, args.gelu, args.resume))
