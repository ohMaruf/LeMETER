from typing import Literal
from eval import evaluate
import json
from dataset import DepthDataset, DepthTrainDataset
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
from dataset import NormalizedNyuDataset
from hardware_acceleration import Config, enable_hardware_acceleration
from loss import balanced_loss_function
from meter import Meter, MeterArchitecture

RUNS_DIR = Path("runs")
DECODER_BATCH_SIZE = 128
WEIGHT_DECAY = 1e-2
LR_DECAY_STEP = 20
LR_DECAY_GAMMA = 0.1

SCHEDULE_EPOCHS = {
    "freeze_encoder": 60,
    "finetune": 60,
    "warm_start": 65,
}

# Initial learning rates for each schedule (encoder, decoder)
INITIAL_LRS = {
    "freeze_encoder": (0.0, 1e-3),
    "finetune": (1e-3, 1e-3),
    "warm_start": (1e-3, 1e-3),  # encoder frozen for first 5 epochs
}

DecoderSchedule = Literal["warm_start", "freeze_encoder", "finetune"]

def build_encoder_model(
    device: torch.device,
    arch: MeterArchitecture,
    encoder_checkpoint_path: Path,
    train_encoder: bool,
) -> Meter:
    model = Meter(device, arch)

    ckpt = torch.load(encoder_checkpoint_path, map_location="cpu")
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in ckpt["model_state"].items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state)
    logger.info(f"loaded encoder weights from {encoder_checkpoint_path} (epoch {ckpt.get('epoch') + 1})")

    for param in model.encoder.parameters():
        param.requires_grad = train_encoder

    logger.info(f"encoder is {'trainable' if train_encoder else 'frozen'}")
    model.to(device)
    return model

def get_dataloaders():
    train = DataLoader(
        DepthTrainDataset("train", augment=True),
        batch_size=DECODER_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
        prefetch_factor=2,
        persistent_workers=True,
    )
    val = DataLoader(
        NormalizedNyuDataset("val"),
        batch_size=16,
        shuffle=False,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
    )
    return train, val

def train_decoder(
        run_name: str,
        config: Config = Config.DEFAULT,
        resume: bool = True,
        arch: MeterArchitecture = "xxs",
        dataset: DepthDataset = "nyu",
        schedule: DecoderSchedule = "warm_start",
        checkpoint_epoch: int = globals.PRETRAIN_EPOCHS,
):

    total_epochs = SCHEDULE_EPOCHS[schedule]
    init_enc_lr, init_dec_lr = INITIAL_LRS[schedule]
    train_encoder = (schedule != "freeze_encoder")  # may be temporarily frozen in warm_start

    output_dir = RUNS_DIR / f"{dataset}_{arch}_{run_name}_decoder_{schedule}"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    input_dir = RUNS_DIR / f"{dataset}_{arch}_{run_name}_encoder"
    encoder_checkpoint_path = input_dir / f"encoder_epoch_{checkpoint_epoch:03d}.pt"

    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run directory: {output_dir}")
    logger.info(f"schedule: {schedule}, epochs: {total_epochs}")

    raw_model = build_encoder_model(device, arch, encoder_checkpoint_path, train_encoder)
    train, val = get_dataloaders()

    # Optimizer with separate groups and initial LRs
    param_groups = [
        {"params": raw_model.decoder.parameters(), "lr": init_dec_lr, "name": "decoder"},
    ]
    if train_encoder:
        param_groups.append(
            {"params": raw_model.encoder.parameters(), "lr": init_enc_lr, "name": "encoder"}
        )

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
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
    }
    best_rmse = math.inf
    start_epoch = 0

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        history = {**history, **ckpt.get("history", {})}
        best_rmse = ckpt.get("best_rmse", best_rmse)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

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

    for epoch in range(start_epoch, total_epochs):
        # Handle encoder freezing for warm_start (first 5 epochs)
        if schedule == "warm_start" and epoch < 5:
            raw_model.encoder.eval()
            for param in raw_model.encoder.parameters():
                param.requires_grad = False
        elif schedule == "warm_start" and epoch == 5:
            # Enable encoder training after warm-up
            for param in raw_model.encoder.parameters():
                param.requires_grad = True
            # Optionally, if you want to reset the encoder's LR to its initial value
            # after the warm‑up, you can adjust the scheduler's internal counter,
            # but we keep it decaying from the start.

        if schedule == "freeze_encoder" or (schedule == "warm_start" and epoch < 5):
            raw_model.encoder.eval()

        model.train()
        epoch_loss = 0.0
        epoch_components = {"loss_depth": 0.0, "loss_ssim": 0.0, "loss_normal": 0.0, "loss_grad": 0.0}

        for batch in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["depth"].to(device, non_blocking=True)
                z = model(x)

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
            trainable = [p for p in raw_model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=500.0)
            scaler.step(opt)
            scaler.update()

        scheduler.step()

        train_loss = epoch_loss / len(train)
        metrics = evaluate(raw_model, val, device)
        logger.info(f"[{epoch}/{total_epochs}] train/loss {train_loss:.4f}")
        logger.info(f"[{epoch}/{total_epochs}] val/RMSE  {metrics['rmse']:.3f}m")
        logger.info(f"[{epoch}/{total_epochs}] val/REL {metrics['rel']:.3f}")
        logger.info(f"[{epoch}/{total_epochs}] val/d1 {metrics['delta1']:.3f}")

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
            "model_state": raw_model.state_dict(),
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

        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)

        with open(output_dir / "losses.json", "w") as losses_file:
            json.dump(history, losses_file)

def main():
    from cli import parse_cli_args

    args = parse_cli_args()
    train_decoder(
        run_name=args.name,
        config=args.config,
        resume=args.resume,
        arch=args.arch,
        dataset=args.dataset,
        checkpoint_epoch=args.checkpoint_epoch,
        schedule=args.decoder_schedule,
    )

if __name__ == "__main__":
    raise SystemExit(main())