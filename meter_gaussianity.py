import argparse
import json
from pathlib import Path

import matplotlib

import globals

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import logger
from dataset import NormalizedNyuDataset
from hardware_acceleration import Config, enable_hardware_acceleration
from meter import Meter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("pca_meter_gaussianity"))
    parser.add_argument("--limit", type=int, default=0, help="cap on test images (0 = all)")
    parser.add_argument("--num-dirs", type=int, default=4, help="random directions to plot")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def pooled_embeddings(encoder, loader, device, limit=0):
    """Mean-pool the encoder's final feature map -> [N, C] on CPU."""
    pooled = []
    seen = 0
    for batch in loader:
        out, _ = encoder(batch["image"].to(device))
        pooled.append(out.float().mean(dim=[2, 3]).cpu())
        seen += out.shape[0]
        if limit and seen >= limit:
            break
    return torch.cat(pooled)[: limit or None]


def gaussianity_stats(samples: torch.Tensor) -> dict:
    """Skewness, excess kurtosis and KS distance to N(0,1) (shape-only)."""
    x = samples.double()
    x = (x - x.mean()) / x.std().clamp_min(1e-12)
    skew = (x**3).mean().item()
    excess_kurtosis = (x**4).mean().item() - 3.0

    xs = x.sort().values
    n = xs.numel()
    empirical_cdf = torch.arange(1, n + 1, dtype=torch.float64) / n
    normal_cdf = 0.5 * (1 + torch.erf(xs / 2**0.5))
    ks = (empirical_cdf - normal_cdf).abs().max().item()
    return {"skew": skew, "excess_kurtosis": excess_kurtosis, "ks": ks}


def figure_gaussianity(samples: torch.Tensor, path: Path):
    """Two rows: raw projections (top) and standardized projections (bottom)."""
    num_dirs = samples.shape[1]
    fig, axes = plt.subplots(2, num_dirs, figsize=(3.5 * num_dirs, 6))
    grid = torch.linspace(-4, 4, 200)
    normal_pdf = torch.exp(-grid.square() / 2) / (2 * torch.pi) ** 0.5

    per_dir = []
    for k in range(num_dirs):
        col = samples[:, k]
        stats = gaussianity_stats(col)
        per_dir.append(stats)

        # raw projection: overlay a Gaussian matched to its own mean/std
        mu, sigma = col.mean().item(), col.std().item()
        ax = axes[0, k]
        ax.hist(col.numpy(), bins=40, density=True, alpha=0.7)
        matched = torch.exp(-((grid * sigma + mu - mu) ** 2) / (2 * sigma**2)) / (
            sigma * (2 * torch.pi) ** 0.5
        )
        ax.plot(grid * sigma + mu, matched, "r-", lw=1.2, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
        ax.set_title(f"dir {k}: raw projection")
        ax.legend(fontsize=8)

        # standardized: compare shape directly to N(0,1)
        std_col = (col - mu) / max(sigma, 1e-12)
        ax = axes[1, k]
        ax.hist(std_col.numpy(), bins=40, density=True, alpha=0.7)
        ax.plot(grid, normal_pdf, "r-", lw=1.2, label="N(0,1)")
        ax.set_title(
            f"dir {k}: standardized\nskew={stats['skew']:.2f} exkurt={stats['excess_kurtosis']:.2f} KS={stats['ks']:.3f}"
        )
        ax.legend(fontsize=8)

    fig.suptitle("METER xxs pooled embeddings: 1-D random projections vs Gaussian")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return per_dir


def main():
    args = parse_args()
    torch.manual_seed(args.seed or globals.SEED)
    device = enable_hardware_acceleration(Config.DEFAULT)
    args.output.mkdir(parents=True, exist_ok=True)

    encoder = Meter.load(device, "nyu", "xxs").encoder.to(device).eval()
    logger.info("loaded depth-supervised METER xxs encoder")

    test_ds = NormalizedNyuDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=False,
    )

    logger.info("extracting pooled test embeddings")
    pooled = pooled_embeddings(encoder, test_loader, device, args.limit)
    logger.info(f"pooled embeddings: {tuple(pooled.shape)}")

    generator = torch.Generator().manual_seed(args.seed or globals.SEED)
    directions = torch.randn(pooled.shape[1], args.num_dirs, generator=generator)
    directions /= directions.norm(dim=0)
    samples = pooled @ directions

    per_dir = figure_gaussianity(samples, args.output / "fig_meter_gaussianity.png")

    abs_skew = sum(abs(d["skew"]) for d in per_dir) / len(per_dir)
    abs_exkurt = sum(abs(d["excess_kurtosis"]) for d in per_dir) / len(per_dir)
    mean_ks = sum(d["ks"] for d in per_dir) / len(per_dir)
    metrics = {
        "num_images": pooled.shape[0],
        "embedding_dim": pooled.shape[1],
        "per_direction": per_dir,
        "mean_abs_skew": abs_skew,
        "mean_abs_excess_kurtosis": abs_exkurt,
        "mean_ks": mean_ks,
    }
    with open(args.output / "meter_gaussianity.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info(
        f"mean |skew|={abs_skew:.3f}  mean |excess kurtosis|={abs_exkurt:.3f}  mean KS={mean_ks:.3f}"
    )
    logger.info(f"wrote figure and metrics to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())