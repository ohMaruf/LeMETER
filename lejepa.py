import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

import globals
import logger
from dataset import AugmentedNyuDataset
from hardware_acceleration import Config, enable_hardware_acceleration
from meter import LeMeterEncoder, Meter
from sigreg import SigReg

OUTPUT_DIR = Path("runs/lemeter")
CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pt"


def pretrain_lejepa_encoder():
    RESUME = True
    DEVICE = enable_hardware_acceleration(Config.DEFAULT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meter = Meter.load(DEVICE, "nyu", "xxs")
    meter.train()

    raw_encoder = LeMeterEncoder(DEVICE, meter.encoder).to(DEVICE)
    train_ds = AugmentedNyuDataset("train", globals.VIEWS)
    train = DataLoader(
        train_ds,
        batch_size=globals.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    # probe = nn.Sequential(nn.LayerNorm(EMBEDDING_DIM), nn.Linear(EMBEDDING_DIM, 1)).to(DEVICE)
    sigreg = SigReg().to(DEVICE)

    g1 = {
        "params": raw_encoder.parameters(),
        "lr": globals.LEARNING_RATE,
        "weight_decay": 5e-2,
    }
    # g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    # opt = torch.optim.AdamW([g1, g2])
    opt = torch.optim.AdamW([g1])
    warmup_steps = len(train)
    total_steps = len(train) * globals.NUM_EPOCHS
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(
        opt,
        schedulers=[s1, s2],
        milestones=[warmup_steps],
    )

    history = {
        "sigreg_loss": [],
        "lejepa_loss": [],
    }

    start_epoch = 0
    if RESUME and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        raw_encoder.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.warn(f"resumed from {CHECKPOINT_PATH} at epoch #{start_epoch}")

    # optimization strategies
    if DEVICE.type == "cuda":
        logger.info("Enabling torch.compile for CUDA")
        encoder = torch.compile(raw_encoder)
    else:
        logger.warn(f"torch.compile is not supported/stable on {DEVICE.type}, skipping")
        encoder = raw_encoder

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(start_epoch, globals.NUM_EPOCHS):
        encoder.train()  # , probe.train()

        epoch_sigreg = 0.0
        epoch_lejepa = 0.0
        for views, y in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(DEVICE.type, dtype=torch.bfloat16): # bfloat16 is numerically more stable than float16
                views = views.to(DEVICE, non_blocking=True)
                # y = y.to(DEVICE, non_blocking=True)
                # _, proj = encoder(views)
                _, proj, _ = encoder(views)

                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (1 - globals.LAMBDA)
                # y_rep, yhat = y.repeat_interleave(VIEWS), probe(emb.detach())
                # probe_loss = TF.cross_entropy(yhat, y_rep)
                loss = lejepa_loss  # + probe_loss
                if not torch.isfinite(loss):
                    msg = f"Invalid loss {loss.item()} at epoch {epoch}"
                    logger.error(msg)
                    raise RuntimeError(msg)

            epoch_sigreg += sigreg_loss.item()
            epoch_lejepa += lejepa_loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw_encoder.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()


        logger.info(f"[{epoch}/{globals.NUM_EPOCHS}] pretrain/lejepa {epoch_lejepa / len(train)}")
        logger.info(f"[{epoch}/{globals.NUM_EPOCHS}] pretrain/sigreg {epoch_sigreg / len(train)}")

        history["sigreg_loss"].append(epoch_sigreg / len(train))
        history["lejepa_loss"].append(epoch_lejepa / len(train))

        checkpoint = {
            "epoch": epoch,
            "model_state": raw_encoder.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

    with open(OUTPUT_DIR / "losses.json", "w") as losses_file:
        json.dump(history, losses_file)


def main():
    pretrain_lejepa_encoder()

    # todo list pretraining
    # [x] add checkpoints
    # [ ] track loss across epochs to build charts loss vs epochs

    # todo list training
    # [ ] track loss across epochs to build charts loss vs epochs
    # [ ] track accuracy on test set over epochs to build chart accuracy vs epochs


if __name__ == "__main__":
    raise SystemExit(main())
