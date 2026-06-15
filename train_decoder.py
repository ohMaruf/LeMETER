from cli import parse_cli_args
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
from eval import delta1, rel, rmse, run_inference, valid_depth_mask
from hardware_acceleration import Config, enable_hardware_acceleration
from loss import balanced_loss_function
from meter import Meter, MeterArchitecture

RUNS_DIR = Path("runs")

# when fine-tuning, the encoder is adapted at 10x lower LR than the decoder so
# the pretrained features are refined rather than overwritten
ENCODER_LR = globals.LEARNING_RATE / 10
DECODER_BATCH_SIZE = 128
DECODER_EPOCHS = 60
# METER paper: AdamW with weight decay 0.01, LR decayed x0.1 every 20 epochs
WEIGHT_DECAY = 1e-2
LR_DECAY_EVERY = 20
LR_DECAY_GAMMA = 0.1


def build_encoder_model(
    device: torch.device,
    arch: MeterArchitecture,
    encoder_checkpoint_path: Path,
    freeze_encoder: bool,
) -> Meter:
    model = Meter(device, arch)

    ckpt = torch.load(encoder_checkpoint_path, map_location="cpu")
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in ckpt["model_state"].items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state)
    logger.info(f"encoder: LeJEPA weights from {encoder_checkpoint_path} (epoch {ckpt.get('epoch')})")

    if freeze_encoder:
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


def train_decoder(
        run_name: str,
        config: Config = Config.DEFAULT,
        resume: bool = True,
        arch: MeterArchitecture = "xxs",
        dataset: DepthDataset = "nyu",
        freeze_encoder: bool = True,
        checkpoint_epoch: int = globals.PRETRAIN_EPOCHS,
):
    output_dir = RUNS_DIR / f"{dataset}_{arch}_{run_name}_decoder"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    input_dir = RUNS_DIR / f"{dataset}_{arch}_{run_name}_encoder"
    encoder_checkpoint_path = input_dir / f"encoder_epoch_{checkpoint_epoch:03d}.pt"
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run directory: {output_dir}")

    raw_model = build_encoder_model(device, arch, encoder_checkpoint_path, freeze_encoder)

    train = DataLoader(
        DepthTrainDataset("train", augment=True),
        batch_size=DECODER_BATCH_SIZE,
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

    param_groups = [{"params": raw_model.decoder.parameters(), "lr": globals.LEARNING_RATE}]
    if not freeze_encoder:
        param_groups.append({"params": raw_model.encoder.parameters(), "lr": ENCODER_LR})
    # METER paper: AdamW (beta1=0.9, beta2=0.999), step LR decay x0.1 every 20
    # epochs (so it steps once per epoch, not per batch)
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

    for epoch in range(start_epoch, DECODER_EPOCHS):
        model.train()
        if freeze_encoder:
            # the encoder must stay in eval mode: BatchNorm running stats are
            # part of the frozen representation under evaluation
            raw_model.encoder.eval()

        epoch_loss = 0.0
        epoch_components = {"loss_depth": 0.0, "loss_ssim": 0.0, "loss_normal": 0.0, "loss_grad": 0.0}
        for batch in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["depth"].to(device, non_blocking=True)
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
        metrics = evaluate(raw_model, test, device)
        logger.info(
            f"[{epoch}/{DECODER_EPOCHS}] train/loss {train_loss:.4f} "
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

        # to avoid risk of corrupting the previous checkpoint file
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)


def main():
    args = parse_cli_args()
    train_decoder(
        run_name=args.name,
        config=args.config,
        resume=args.resume,
        arch=args.arch,
        dataset=args.dataset,
        freeze_encoder=args.freeze_encoder,
        checkpoint_epoch=args.checkpoint_epoch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
