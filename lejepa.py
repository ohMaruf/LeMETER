import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

OUTPUT_DIR = Path("runs/pretrain_encoder")
CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pt"

# periodic encoder snapshots, so we can later chart how the latent space (PCA
# probing) and the downstream decoder performance evolve with pretraining length
SNAPSHOT_EVERY = 5


def pretrain_lejepa_encoder():
    RESUME = True
    torch.manual_seed(globals.SEED)
    DEVICE = enable_hardware_acceleration(Config.DEFAULT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # random initialization: starting from the depth-supervised METER weights
    # would contaminate the SSL-vs-supervised comparison, since the encoder
    # would already contain depth-task information before LeJEPA runs
    meter = Meter(DEVICE, "xxs")

    raw_encoder = LeMeterEncoder(DEVICE, meter.encoder).to(DEVICE)
    train_ds = AugmentedNyuDataset("train", globals.VIEWS)
    # shuffle is required: the manifest is grouped by scene (~178 frames per
    # scene), so without it every batch holds near-duplicate frames and the
    # SIGReg batch statistic degenerates
    train = DataLoader(
        train_ds,
        batch_size=globals.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
        # RAM in queues = workers x prefetch x ~290 MiB per batch; 24 x 2 is
        # ~13.5 GiB, comfortable on the 32 GiB box
        prefetch_factor=2,
        pin_memory=False,
        # disabled because it performs worse on our training machine when set to True
        # pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    # nonlinear diagnostic probe: regresses the (normalized) mean scene depth
    # from the detached embedding. It never influences the encoder; its
    # per-epoch R2 tracks how much depth information the embedding carries.
    probe = nn.Sequential(
        nn.LayerNorm(globals.EMBEDDING_DIM),
        nn.Linear(globals.EMBEDDING_DIM, globals.EMBEDDING_DIM),
        nn.GELU(),
        nn.Linear(globals.EMBEDDING_DIM, 1),
    ).to(DEVICE)
    sigreg = SigReg().to(DEVICE)

    g1 = {
        "params": raw_encoder.parameters(),
        "lr": globals.LEARNING_RATE,
        "weight_decay": 5e-2,
    }
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * globals.PRETRAIN_EPOCHS
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=globals.LEARNING_RATE * 1e-2)
    scheduler = SequentialLR(
        opt,
        schedulers=[s1, s2],
        milestones=[warmup_steps],
    )

    history = {
        "sigreg_loss": [],
        "inv_loss": [],
        "lejepa_loss": [],
        "probe_r2": [],
        "lr": [],
        "grad_norm": [],
        "epoch_seconds": [],
    }

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 0
    if RESUME and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        raw_encoder.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        sigreg.load_state_dict(ckpt["sigreg_state"])
        probe.load_state_dict(ckpt["probe_state"])
        # keep defaults for series that older checkpoints did not track
        history = {**history, **ckpt.get("history", {})}
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        rng = ckpt["rng_state"]
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
        if rng.get("mps") and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(rng["mps"].cpu())


        logger.warn(f"resumed from {CHECKPOINT_PATH} at epoch #{start_epoch}")

    # optimization strategies
    if DEVICE.type == "cuda":
        logger.info("Enabling torch.compile for CUDA")
        encoder = torch.compile(raw_encoder)
    else:
        logger.warn(f"torch.compile is not supported/stable on {DEVICE.type}, skipping")
        encoder = raw_encoder

    for epoch in range(start_epoch, globals.PRETRAIN_EPOCHS):
        encoder.train(), probe.train()

        epoch_start = time.time()
        epoch_sigreg = 0.0
        epoch_inv = 0.0
        epoch_lejepa = 0.0
        epoch_grad_norm = 0.0
        probe_ss_res = 0.0
        probe_y_sum = 0.0
        probe_y_sq = 0.0
        probe_count = 0
        for views, y in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(DEVICE.type, dtype=torch.bfloat16): # bfloat16 is numerically more stable than float16
                views = views.to(DEVICE, non_blocking=True)
                emb, proj, _ = encoder(views)

                proj_mean = proj.mean(0)
                inv_loss = (proj_mean - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (1 - globals.LAMBDA)

            # probe in fp32, on detached embeddings: gradients flow only into
            # the probe head, and the target is mean depth normalized by the
            # 10 m (1000 cm) range
            y_rep = y.to(DEVICE, non_blocking=True).float().repeat_interleave(globals.VIEWS) / 1000.0
            yhat = probe(emb.detach().float()).squeeze(-1)
            probe_loss = F.mse_loss(yhat, y_rep)

            loss = lejepa_loss + probe_loss
            if not torch.isfinite(loss):
                msg = f"Invalid loss {loss.item()} at epoch {epoch}"
                logger.error(msg)
                raise RuntimeError(msg)

            epoch_sigreg += sigreg_loss.item()
            epoch_inv += inv_loss.item()
            epoch_lejepa += lejepa_loss.item()
            probe_ss_res += (yhat - y_rep).detach().square().sum().item()
            probe_y_sum += y_rep.sum().item()
            probe_y_sq += y_rep.square().sum().item()
            probe_count += y_rep.numel()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_encoder.parameters(), max_norm=1.0)
            epoch_grad_norm += grad_norm.item()
            scaler.step(opt)
            scaler.update()
            scheduler.step()


        # streaming R2 over the epoch: 1 - SS_res / SS_tot
        probe_ss_tot = probe_y_sq - probe_y_sum**2 / max(probe_count, 1)
        probe_r2 = 1.0 - probe_ss_res / max(probe_ss_tot, 1e-12)

        logger.info(f"[{epoch}/{globals.PRETRAIN_EPOCHS}] pretrain/lejepa {epoch_lejepa / len(train)}")
        logger.info(f"[{epoch}/{globals.PRETRAIN_EPOCHS}] pretrain/sigreg {epoch_sigreg / len(train)}")
        logger.info(f"[{epoch}/{globals.PRETRAIN_EPOCHS}] pretrain/probe_r2 {probe_r2:.4f}")

        history["sigreg_loss"].append(epoch_sigreg / len(train))
        history["inv_loss"].append(epoch_inv / len(train))
        history["lejepa_loss"].append(epoch_lejepa / len(train))
        history["probe_r2"].append(probe_r2)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["grad_norm"].append(epoch_grad_norm / len(train))
        history["epoch_seconds"].append(time.time() - epoch_start)

        checkpoint = {
            "epoch": epoch,
            "model_state": raw_encoder.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "sigreg_state": sigreg.state_dict(),
            "probe_state": probe.state_dict(),
            "history": history,
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

        if (epoch + 1) % SNAPSHOT_EVERY == 0 or epoch + 1 == globals.PRETRAIN_EPOCHS:
            snapshot_path = OUTPUT_DIR / f"encoder_epoch_{epoch + 1:03d}.pt"
            torch.save({"epoch": epoch, "model_state": raw_encoder.state_dict()}, snapshot_path)
            logger.info(f"saved encoder snapshot to {snapshot_path}")

        # dumped every epoch so a crash never loses the chart data
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
