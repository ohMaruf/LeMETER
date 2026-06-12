"""Downstream MDE training on NYU, configured by the consts below.

ENCODER_SOURCE selects the initialization (LeJEPA-pretrained vs random
control); FREEZE_ENCODER selects the protocol: frozen encoder = controlled
representation probing (runs/decoder/<src>), unfrozen = practical fine-tuning
with the encoder at 10x lower LR (runs/finetune/<src>). Tracks
RMSE / AbsRel / d1 on the test split every epoch (goals.txt: convergence
speed); compare against the published METER xxs numbers
(RMSE 0.560 m, REL 0.163, d1 0.763 with eval.py on this machine).
"""

from dataset import DepthTrainDataset
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import globals
import logger
from dataset import NormalizedNyuDataset
from eval import delta1, rel, rmse, run_inference, valid_depth_mask
from hardware_acceleration import Config, enable_hardware_acceleration
from loss import balanced_loss_function
from meter import Meter

# which encoder the decoder is trained on; run once per value:
#   "lejepa" — the pretrained encoder under test
#   "random" — control: what the decoder achieves with no pretraining at all
ENCODER_SOURCE = "lejepa"

# frozen = representation probing (the controlled comparison between encoders);
# unfrozen = practical fine-tuning protocol (goals.txt), encoder at 10x lower
# LR so the pretrained features are adapted rather than overwritten
FREEZE_ENCODER = False
ENCODER_LR = globals.LEARNING_RATE / 10

# epoch 20, not the last checkpoint: snapshot probing showed depth-probe R2
# peaks at epoch 10-20 (0.30) and slowly degrades to 0.26 by epoch 60, while
# SIGReg keeps inflating rank with depth-irrelevant directions
ENCODER_CHECKPOINT_PATH = Path("runs/pretrain_encoder/encoder_epoch_020.pt")
OUTPUT_DIR = Path("runs/decoder" if FREEZE_ENCODER else "runs/finetune") / ENCODER_SOURCE
CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pt"

WEIGHT_DECAY = 1e-4


def build_frozen_encoder_model(device: torch.device) -> Meter:
    model = Meter(device, "xxs")  # fresh random encoder + decoder

    if ENCODER_SOURCE == "lejepa":
        ckpt = torch.load(ENCODER_CHECKPOINT_PATH, map_location="cpu")
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in ckpt["model_state"].items()
            if key.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder_state)
        logger.info(f"encoder: LeJEPA weights from {ENCODER_CHECKPOINT_PATH} (epoch {ckpt.get('epoch')})")
    else:
        logger.info("encoder: random initialization (no-pretraining control)")

    if FREEZE_ENCODER:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model.to(device)
    return model


@torch.no_grad()
def evaluate(model: Meter, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"rmse": 0.0, "rel": 0.0, "delta1": 0.0}
    count = 0
    for item in loader:
        x = item["image"].to(device)
        y = item["depth"].to(device).float()  # test labels: millimeters, full res
        z = run_inference(model, x)
        for index in range(x.shape[0]):
            yi, zi = y[index], z[index]
            mask = valid_depth_mask(yi, zi)
            totals["rmse"] += rmse(yi, zi, mask) / 1000.0  # millimeters -> meters
            totals["rel"] += rel(yi, zi, mask)
            totals["delta1"] += delta1(yi, zi, mask)
            count += 1
    return {key: value / count for key, value in totals.items()}


def train_decoder():
    RESUME = True
    torch.manual_seed(globals.SEED)
    DEVICE = enable_hardware_acceleration(Config.DEFAULT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_frozen_encoder_model(DEVICE)

    train = DataLoader(
        DepthTrainDataset("train", augment=True),
        batch_size=globals.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
        prefetch_factor=2,
        persistent_workers=True,
    )
    test = DataLoader(
        NormalizedNyuDataset("test"),
        batch_size=16,
        shuffle=False,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
    )

    param_groups = [{"params": model.decoder.parameters(), "lr": globals.LEARNING_RATE}]
    if not FREEZE_ENCODER:
        param_groups.append({"params": model.encoder.parameters(), "lr": ENCODER_LR})
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    warmup_steps = len(train)
    total_steps = len(train) * globals.DECODER_EPOCHS
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=globals.LEARNING_RATE * 1e-2)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    loss_fn = balanced_loss_function(DEVICE)

    history = {
        "train_loss": [],
        "loss_depth": [],
        "loss_ssim": [],
        "loss_normal": [],
        "loss_grad": [],
        "rmse": [],
        "rel": [],
        "delta1": [],
    }
    best_rmse = math.inf
    start_epoch = 0

    if RESUME and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        history = {**history, **ckpt.get("history", {})}
        best_rmse = ckpt.get("best_rmse", best_rmse)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        rng = ckpt["rng_state"]
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
        if rng.get("mps") and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(rng["mps"].cpu())

        logger.warn(f"resumed from {CHECKPOINT_PATH} at epoch #{start_epoch}")

    if DEVICE.type == "cuda":
        logger.info("Enabling torch.compile for CUDA")
        compiled_model = torch.compile(model)
    else:
        logger.warn(f"torch.compile is not supported/stable on {DEVICE.type}, skipping")
        compiled_model = model

    for epoch in range(start_epoch, globals.DECODER_EPOCHS):
        compiled_model.train()
        if FREEZE_ENCODER:
            # the encoder must stay in eval mode: BatchNorm running stats are
            # part of the frozen representation under evaluation
            model.encoder.eval()

        epoch_loss = 0.0
        epoch_components = {"loss_depth": 0.0, "loss_ssim": 0.0, "loss_normal": 0.0, "loss_grad": 0.0}
        for batch in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                x = batch["image"].to(DEVICE, non_blocking=True)
                y = batch["depth"].to(DEVICE, non_blocking=True)
                z = compiled_model(x)
                loss_depth, loss_ssim, loss_normal, loss_grad = loss_fn(z, y)
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
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=500.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        train_loss = epoch_loss / len(train)
        metrics = evaluate(model, test, DEVICE)
        logger.info(
            f"[{epoch}/{globals.DECODER_EPOCHS}] train/loss {train_loss:.4f} "
            f"test/RMSE {metrics['rmse']:.3f}m test/REL {metrics['rel']:.3f} "
            f"test/d1 {metrics['delta1']:.3f}"
        )

        history["train_loss"].append(train_loss)
        for component, total in epoch_components.items():
            history[component].append(total / len(train))
        history["rmse"].append(metrics["rmse"])
        history["rel"].append(metrics["rel"])
        history["delta1"].append(metrics["delta1"])

        is_best = metrics["rmse"] < best_rmse
        best_rmse = min(best_rmse, metrics["rmse"])

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "history": history,
            "best_rmse": best_rmse,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            },
        }

        # to avoid risk of corrupting the previous checkpoint file
        temp_path = CHECKPOINT_PATH.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(CHECKPOINT_PATH)
        if is_best:
            torch.save(checkpoint, BEST_CHECKPOINT_PATH)

        with open(OUTPUT_DIR / "history.json", "w") as history_file:
            json.dump(history, history_file)
        plot_history(history, OUTPUT_DIR / "history.png")


def plot_history(history: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(history["train_loss"], label="total")
    for component in ("loss_depth", "loss_ssim", "loss_normal", "loss_grad"):
        axes[0].plot(history[component], label=component.removeprefix("loss_"), lw=0.9)
    axes[0].set_title("train loss"), axes[0].legend(fontsize=8)
    axes[1].plot(history["rmse"], label="RMSE (m)")
    axes[1].plot(history["rel"], label="AbsRel")
    axes[1].set_title("test error"), axes[1].legend()
    axes[2].plot(history["delta1"])
    axes[2].set_title("test δ1")
    for ax in axes:
        ax.set_xlabel("epoch"), ax.grid(alpha=0.3)
    fig.suptitle("frozen-encoder decoder training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(train_decoder())
