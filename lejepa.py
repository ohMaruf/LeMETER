from meter import get_embedding_dim
from meter import MeterArchitecture
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
from meter import LeMeterEncoder
from sigreg import SigReg
from dataset import DepthDataset

RUNS_DIR = Path("runs")

# periodic encoder snapshots, so we can later chart how the latent space (PCA
# probing) and the downstream decoder performance evolve with pretraining length
SNAPSHOT_EVERY = 5


def pretrain_lejepa_encoder(
        run_name: str,
        config: Config = Config.DEFAULT,
        resume: bool = True,
        arch: MeterArchitecture = "xxs",
        dataset: DepthDataset = "nyu",
):
    output_dir = RUNS_DIR / f"{dataset}_{arch}_{run_name}_encoder"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(config)
    embedding_dim = get_embedding_dim(arch)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run directory: {output_dir}")

    raw_encoder = LeMeterEncoder(device, arch).to(device)
    train_ds = AugmentedNyuDataset("train", globals.VIEWS, augmentation="lejepa_multi_view", with_depth=True, normalization="imagenet")
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

    # dense diagnostic probe: a 1x1 conv (per-pixel linear map) from the detached
    # bottleneck features to depth. It never influences the encoder; its per-epoch
    # R2 tracks how much *spatially structured* depth the features carry. The
    # affine-free BatchNorm only standardizes the frozen features (no learnable
    # depth-specific parameters).
    probe = nn.Sequential(
        nn.BatchNorm2d(embedding_dim, affine=False),
        nn.Conv2d(embedding_dim, 1, kernel_size=1),
    ).to(device)
    sigreg = SigReg().to(device)

    g1 = {
        "params": raw_encoder.parameters(),
        "lr": globals.LEARNING_RATE,
        "weight_decay": 1e-2,
    }
    g2 = {
        "params": probe.parameters(),
        "lr": globals.LEARNING_RATE,
        "weight_decay": 1e-7,
    }
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * globals.PRETRAIN_EPOCHS
    s1 = LinearLR(opt, start_factor=1e-2, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=globals.LEARNING_RATE * 5e-2)
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
        # std of a random 1-D projection of the SIGReg output, averaged over the
        # epoch; SIGReg targets N(0,1), so this should climb to ~1.0. A plateau
        # well below 1 means the term is too weak — raise globals.LAMBDA.
        "proj_sigma": [],
        "lr": [],
        "grad_norm": [],
        "epoch_seconds": [],
    }

    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
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


        logger.warn(f"resumed from {checkpoint_path} at epoch #{start_epoch + 1}")

    # optimization strategies
    if device.type == "cuda":
        logger.info("Enabling torch.compile for CUDA")
        encoder = torch.compile(raw_encoder)
    else:
        logger.warn(f"torch.compile is not supported/stable on {device.type}, skipping")
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
        # streaming per-coordinate stats of the projection, to recover the
        # std of a random 1-D slice: E[Var(proj @ a)] = mean_j Var(proj_j)
        proj_sum = torch.zeros(embedding_dim, device=device)
        proj_sq = torch.zeros(embedding_dim, device=device)
        proj_count = 0
        for views, y in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=torch.bfloat16): # bfloat16 is numerically more stable than float16
                views = views.to(device, non_blocking=True)
                emb, proj, _, feat_map = encoder(views)

                Vg = globals.GLOBAL_VIEWS
                proj_g = proj[:Vg]
                proj_l = proj

                proj_g_mean = proj_g.mean(0)
                inv_loss_l = (
                    (proj_g_mean - proj_l).square().mean()
                    if proj_l.numel() > 0 else 0.0
                )
                # inv_loss_g = (proj_g_mean - proj_g).square().mean()
                # inv_loss = inv_loss_l + inv_loss_g
                inv_loss = inv_loss_l

                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (1 - globals.LAMBDA)

            # dense probe in fp32, on detached features: gradients flow only into
            # the 1x1-conv head, never the encoder. each view's depth map is
            # spatially aligned (with_depth=True) and normalized by the 10 m
            # (1000 cm) sensor range.
            depth = y.to(device, non_blocking=True).flatten(0, 1).float() / 1000.0
            pred = probe(feat_map.detach().float())
            # score on the probe's native (feature-map) grid: pool the GT down to
            # the prediction rather than upsampling the coarse prediction up, so R2
            # reflects depth decodable at the bottleneck resolution (and we skip a
            # per-step interpolation).
            depth = F.adaptive_avg_pool2d(depth, pred.shape[-2:])
            valid = depth > 0  # zero == no sensor return, excluded from loss/metrics
            probe_loss = F.mse_loss(pred[valid], depth[valid])

            loss = lejepa_loss + probe_loss
            if not torch.isfinite(loss):
                msg = f"Invalid loss {loss.item()} at epoch {epoch}"
                logger.error(msg)
                raise RuntimeError(msg)

            flat_proj = proj.detach().float().reshape(-1, proj.shape[-1])
            proj_sum += flat_proj.sum(0)
            proj_sq += flat_proj.square().sum(0)
            proj_count += flat_proj.shape[0]

            v_pred = pred[valid].detach()
            v_depth = depth[valid]
            epoch_sigreg += sigreg_loss.item()
            epoch_inv += inv_loss.item()
            epoch_lejepa += lejepa_loss.item()
            probe_ss_res += (v_pred - v_depth).square().sum().item()
            probe_y_sum += v_depth.sum().item()
            probe_y_sq += v_depth.square().sum().item()
            probe_count += v_depth.numel()

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

        # E[std of a random unit-direction slice] = sqrt(mean_j Var(proj_j))
        proj_var = (proj_sq / proj_count - (proj_sum / proj_count).square()).clamp_min(0.0)
        proj_sigma = proj_var.mean().sqrt().item()

        logger.info(f"[{epoch + 1}/{globals.PRETRAIN_EPOCHS}] pretrain/lejepa {epoch_lejepa / len(train)}")
        logger.info(f"[{epoch + 1}/{globals.PRETRAIN_EPOCHS}] pretrain/invariance {epoch_inv / len(train)}")
        logger.info(f"[{epoch + 1}/{globals.PRETRAIN_EPOCHS}] pretrain/sigreg {epoch_sigreg / len(train)}")
        logger.info(f"[{epoch + 1}/{globals.PRETRAIN_EPOCHS}] pretrain/probe_r2 {probe_r2:.4f}")
        logger.info(f"[{epoch + 1}/{globals.PRETRAIN_EPOCHS}] pretrain/proj_sigma {proj_sigma:.4f} (target 1.0)")

        history["sigreg_loss"].append(epoch_sigreg / len(train))
        history["inv_loss"].append(epoch_inv / len(train))
        history["lejepa_loss"].append(epoch_lejepa / len(train))
        history["probe_r2"].append(probe_r2)
        history["proj_sigma"].append(proj_sigma)
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
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)

        if (epoch + 1) % SNAPSHOT_EVERY == 0 or epoch + 1 == globals.PRETRAIN_EPOCHS:
            snapshot_path = output_dir / f"encoder_epoch_{epoch + 1:03d}.pt"
            torch.save({"epoch": epoch, "model_state": raw_encoder.state_dict()}, snapshot_path)
            logger.info(f"saved encoder snapshot to {snapshot_path}")

        # dumped every epoch so a crash never loses the chart data
        with open(output_dir / "losses.json", "w") as losses_file:
            json.dump(history, losses_file)


def main():
    from cli import parse_cli_args

    args = parse_cli_args()
    pretrain_lejepa_encoder(
        run_name=args.name,
        config=args.config,
        resume=args.resume,
        arch=args.arch,
        dataset=args.dataset
    )

if __name__ == "__main__":
    raise SystemExit(main())
