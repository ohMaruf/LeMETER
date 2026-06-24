from dataclasses import dataclass, asdict
from dataset import DepthTrainDataset
import torchvision
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
from cli import parse_cli_args
from dataset import AugmentedNyuDataset
from hardware_acceleration import Config, enable_hardware_acceleration
from meter import LeMeterEncoder, Meter
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

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run directory: {output_dir}")

    # random initialization: starting from the depth-supervised METER weights
    # would contaminate the SSL-vs-supervised comparison, since the encoder
    # would already contain depth-task information before LeJEPA runs
    meter = Meter(device, arch)

    raw_encoder = LeMeterEncoder(device, meter.encoder).to(device)
    # with_depth: every view also yields a spatially-aligned depth map, so the
    # online dense probe can be supervised per-pixel on the same crop/flip the
    # encoder saw
    train_ds = AugmentedNyuDataset("train", globals.VIEWS, augmentation="lemeter", with_depth=True)
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
        nn.BatchNorm2d(globals.EMBEDDING_DIM, affine=False),
        nn.Conv2d(globals.EMBEDDING_DIM, 1, kernel_size=1),
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
        proj_sum = torch.zeros(globals.PROJ_DIM, device=device)
        proj_sq = torch.zeros(globals.PROJ_DIM, device=device)
        proj_count = 0
        for views, y in tqdm(train, total=len(train), position=0, leave=True):
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=torch.bfloat16): # bfloat16 is numerically more stable than float16
                views = views.to(device, non_blocking=True)
                emb, proj, _, feat_map = encoder(views)

                proj_mean = proj.mean(0)
                inv_loss = (proj_mean - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (1 - globals.LAMBDA)

            # dense probe in fp32, on detached features: gradients flow only into
            # the 1x1-conv head, never the encoder. each view's depth map is
            # spatially aligned (with_depth=True) and normalized by the 10 m
            # (1000 cm) sensor range; the prediction is upsampled to the depth grid.
            depth = y.to(device, non_blocking=True).flatten(0, 1).float() / 1000.0
            pred = probe(feat_map.detach().float())
            pred = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)
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

def _le_encoder_spatial_features(le_encoder: LeMeterEncoder, x: torch.Tensor) -> torch.Tensor:
    """Bottleneck feature map (N, EMBEDDING_DIM, h, w), i.e. the spatial structure
    *before* the global average pool that the scalar diagnostic probe reads. The
    dense probe needs this so it can test whether depth is linearly decodable
    per-pixel, not just on average."""
    feats, _ = le_encoder.encoder(x)
    feats = feats[0] if isinstance(feats, (tuple, list)) else feats
    return feats


@dataclass
class DenseProbeMetrics:
    rmse_norm: float
    rmse_meters: float
    r2: float
    delta1: float
    samples_size: int


@torch.no_grad()
def _evaluate_dense_probe(encoder, probe, loader, device) -> DenseProbeMetrics:
    probe.eval()
    ss_res = ss_tot = y_sum = y_sq = count = delta1_hits = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        target = depth / 1000.0  # cm -> fraction of the 10 m sensor range

        with torch.autocast(device.type, dtype=torch.bfloat16):
            feats = _le_encoder_spatial_features(encoder, image)
        pred = probe(feats.float())
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        valid = target > 0  # zero == no sensor return (invalid), excluded from metrics
        p = pred[valid]
        t = target[valid]
        ss_res += (p - t).square().sum().item()
        y_sum += t.sum().item()
        y_sq += t.square().sum().item()
        count += t.numel()

        ratio = torch.maximum(p.clamp_min(1e-6) / t, t / p.clamp_min(1e-6))
        delta1_hits += (ratio < 1.25).float().sum().item()

    ss_tot = y_sq - y_sum**2 / max(count, 1)
    rmse_norm = (ss_res / max(count, 1)) ** 0.5
    return DenseProbeMetrics(
        rmse_norm=rmse_norm,
        rmse_meters=rmse_norm * 10.0,  # normalized unit spans the 10 m range
        r2=1.0 - ss_res / max(ss_tot, 1e-12),
        delta1=delta1_hits / max(count, 1),
        samples_size=int(count),
    )


def dense_depth_probe(
    checkpoint_path: Path | None,
    output_dir: Path = Path("pca_results"),
    config: Config = Config.DEFAULT,
    arch: MeterArchitecture = "xxs",
    dataset: DepthDataset = "nyu",
    epochs: int = 10,
    lr: float = 1e-3,
    num_vis: int = 10,
):
    """Dense per-pixel linear probe (Objective 2 / geometric probing).

    Freezes the encoder and trains only a 1x1 conv (a per-pixel linear map) from
    the bottleneck features to depth, then reports R2 / RMSE / delta1 and writes
    predicted-vs-ground-truth depth maps. Because the encoder never updates, the
    score measures how much *spatially structured* depth information the frozen
    representation already carries -- the real signal the scalar probe cannot see.

    Pass `checkpoint_path=None` for the random-init control arm.
    """
    assert dataset == "nyu", "kitti dataset not implemented yet"
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(config)

    arm = "random" if checkpoint_path is None else checkpoint_path.parent.name
    out_dir = output_dir / f"dense_probe/{dataset}/{arm}/{arch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meter = Meter(device, arch)
    encoder = LeMeterEncoder(device, meter.encoder).to(device)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(ckpt["model_state"])
        logger.info(f"loaded encoder from {checkpoint_path}")
    else:
        logger.warn("no checkpoint: probing a randomly initialized encoder (control)")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # 1x1 conv == per-pixel linear map; the affine-free BatchNorm just
    # standardizes the frozen features so the linear layer conditions well, and
    # carries no learnable depth-specific parameters of its own
    probe = nn.Sequential(
        nn.BatchNorm2d(globals.EMBEDDING_DIM, affine=False),
        nn.Conv2d(globals.EMBEDDING_DIM, 1, kernel_size=1),
    ).to(device)

    common = dict(
        batch_size=globals.BATCH_SIZE,
        num_workers=min(globals.DATALOADER_WORKERS, os.cpu_count() or 1),
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=True,
    )
    train_ds = DepthTrainDataset(split="train", augment=False)
    test_ds = DepthTrainDataset(split="test", augment=False)
    train = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    test = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        for batch in tqdm(train, total=len(train), position=0, leave=True):
            image = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            target = depth / 1000.0

            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                feats = _le_encoder_spatial_features(encoder, image)
            pred = probe(feats.float())
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

            valid = target > 0
            loss = F.mse_loss(pred[valid], target[valid])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        metrics = _evaluate_dense_probe(encoder, probe, test, device)
        logger.info(
            f"[{epoch + 1}/{epochs}] train_mse {epoch_loss / len(train):.5f} | "
            f"test R2 {metrics.r2:.4f} RMSE {metrics.rmse_meters:.3f} m delta1 {metrics.delta1:.4f}"
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=4)

    # predicted-vs-ground-truth depth maps for a handful of test images
    probe.eval()
    for idx in range(min(num_vis, len(test_ds))):
        sample = test_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            feats = _le_encoder_spatial_features(encoder, image)
        pred = probe(feats.float())
        pred = F.interpolate(pred, size=globals.INPUT_RESOLUTION, mode="bilinear", align_corners=False)
        gt = F.interpolate(depth, size=globals.INPUT_RESOLUTION, mode="bilinear", align_corners=False)

        def _to_rgb(m: torch.Tensor) -> torch.Tensor:
            m = (m - m.amin()) / (m.amax() - m.amin() + 1e-8)
            return m[0].repeat(3, 1, 1)

        img_disp = _to_rgb(image)  # min-max so the z-scored input is viewable
        grid = torchvision.utils.make_grid([img_disp, _to_rgb(pred), _to_rgb(gt)])
        torchvision.utils.save_image(grid, out_dir / f"dense_probe_{idx}.png")

    logger.info(f"wrote dense-probe results to {out_dir}")
    return metrics



def main():
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
