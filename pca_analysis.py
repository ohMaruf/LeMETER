import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import globals
import logger
import pca_plots
from dataset import DepthDataset, NormalizedNyuDataset
from hardware_acceleration import Config, enable_hardware_acceleration
from meter import LeMeterEncoder, Meter, MeterArchitecture, MobileViT, build_mobilevit

# y2/y3 are encoder skip taps (strides 8/16), final is the bottleneck (stride 32).
# Channel counts are arch-dependent (recorded per level as "channels" in metrics).
LEVELS = ("y2", "y3", "final")
TOP_PCS = 5
MI_BINS = 16
DEPTH_TO_METERS = 1.0 / 100.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--arch", choices=("xxs", "xs", "s"), default="xxs", help="MobileViT backbone size")
    parser.add_argument("--dataset", choices=("nyu", "kitti"), default="nyu", help="checkpoints to load (probing data is the NYU test split)")
    parser.add_argument("--output", type=Path, default=None, help="output dir (default pca/{dataset}_{arch})")
    parser.add_argument("--limit", type=int, default=0, help="cap on test images (0 = all)")
    parser.add_argument("--num-viz", type=int, default=6, help="images shown in the PCA map figures")
    parser.add_argument("--probe-train", type=int, default=1024, help="train images used to fit the depth probe")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("pca") / f"{args.dataset}_{args.arch}"
    return args


# --------------------------------------------------------------------------- #
# Encoders and feature extraction
# --------------------------------------------------------------------------- #

def build_encoders(device: torch.device, dataset: DepthDataset, arch: MeterArchitecture) -> dict[str, MobileViT]:
    encoders = {
        "lejepa": LeMeterEncoder.load(device, dataset, arch).encoder,
        "meter": Meter.load(device, dataset, arch).encoder,
    }
    torch.manual_seed(globals.SEED)  # reproducible control, fit after the loaded ones
    encoders["random"] = build_mobilevit(arch)[0]

    for encoder in encoders.values():
        encoder.to(device).eval()
    logger.info(f"built encoders {list(encoders)} for arch={arch} dataset={dataset}")
    return encoders


@torch.no_grad()
def extract_features(encoder, loader, device, limit=0, desc="extracting features"):
    """Returns per-level feature maps [N, C, h, w] and depths [N, 1, H, W] on CPU."""
    feats = {level: [] for level in LEVELS}
    depths = []
    seen = 0
    total = len(loader) if not limit else min(len(loader), -(-limit // loader.batch_size))
    for batch in tqdm(loader, total=total, desc=desc, leave=False):
        out, skips = encoder(batch["image"].to(device))
        feats["final"].append(out.float().cpu())
        feats["y2"].append(skips[2].float().cpu())
        feats["y3"].append(skips[3].float().cpu())
        depths.append(batch["depth"].float())
        seen += out.shape[0]
        if limit and seen >= limit:
            break
    feats = {level: torch.cat(maps)[: limit or None] for level, maps in feats.items()}
    return feats, torch.cat(depths)[: limit or None]


# --------------------------------------------------------------------------- #
# PCA and metrics
# --------------------------------------------------------------------------- #

def flatten_patches(feature_maps: torch.Tensor) -> torch.Tensor:
    # [N, C, h, w] -> [N * h * w, C]
    return feature_maps.permute(0, 2, 3, 1).reshape(-1, feature_maps.shape[1])


def fit_pca(X: torch.Tensor):
    mean = X.mean(0)
    centered = (X - mean).double()
    cov = centered.T @ centered / (X.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0).float()
    eigenvectors = eigenvectors.flip(1).float()
    return mean, eigenvalues, eigenvectors


def effective_rank(eigenvalues: torch.Tensor) -> float:
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def participation_ratio(eigenvalues: torch.Tensor) -> float:
    return (eigenvalues.sum() ** 2 / eigenvalues.square().sum()).item()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ranks_a = a.argsort().argsort().float()
    ranks_b = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ranks_a, ranks_b]))[0, 1].item()


def median_split_iou(pc_map: torch.Tensor, depth_map: torch.Tensor) -> float:
    """Agreement between a far/near split of depth and a PC1 split (sign-invariant)."""
    far = depth_map > depth_map.median()
    positive = pc_map > pc_map.median()
    iou_a = (far & positive).sum() / (far | positive).sum().clamp_min(1)
    iou_b = (far & ~positive).sum() / (far | ~positive).sum().clamp_min(1)
    return max(iou_a.item(), iou_b.item())


def mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = MI_BINS) -> float:
    """Mutual information (bits) between two flat tensors.

    MI captures *any* dependence, not just the monotonic part Spearman sees, so
    it rewards a PC that segments depth planes even when the relationship is not
    rank-consistent. Both inputs are discretized into equal-frequency (quantile)
    bins, which makes the estimate invariant to their arbitrary scales and keeps
    every marginal bin populated — unlike equal-width binning, which on skewed
    features leaves most cells empty and biases MI upward. Range: [0, log2(bins)].
    """
    x_bin, y_bin = _quantile_bin(x, bins), _quantile_bin(y, bins)
    joint = torch.zeros(bins, bins)
    joint.index_put_((x_bin, y_bin), torch.ones_like(x_bin, dtype=torch.float), accumulate=True)

    p_xy = joint / joint.sum()
    independent = p_xy.sum(1, keepdim=True) * p_xy.sum(0, keepdim=True)
    mask = p_xy > 0
    return (p_xy[mask] * torch.log2(p_xy[mask] / independent[mask])).sum().item()


def _quantile_bin(values: torch.Tensor, bins: int) -> torch.Tensor:
    """Map values to equal-frequency bin indices [0, bins) by rank."""
    ranks = values.argsort().argsort()
    return (ranks * bins // values.numel()).clamp_max_(bins - 1)


def depth_at(depth: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(depth, size)


def ridge_probe(X_train, y_train, X_test, y_test, alpha=1e-1):
    mean, std = X_train.mean(0), X_train.std(0).clamp_min(1e-6)
    X_train = ((X_train - mean) / std).double()
    X_test = ((X_test - mean) / std).double()
    y_mean = y_train.mean().double()
    y_train = y_train.double() - y_mean

    gram = X_train.T @ X_train
    gram += alpha * X_train.shape[0] * torch.eye(gram.shape[0], dtype=torch.float64)
    weights = torch.linalg.solve(gram, X_train.T @ y_train)

    prediction = X_test @ weights + y_mean
    residual = (prediction - y_test).square()
    r2 = 1.0 - residual.sum() / (y_test - y_test.mean()).square().sum()
    return {"r2": r2.item(), "rmse_m": residual.mean().sqrt().item()}


def analyze_level(feats: torch.Tensor, depths: torch.Tensor) -> tuple[dict, dict]:
    """Per-encoder-level PCA metrics + artifacts needed for figures."""
    n, c, h, w = feats.shape
    mean, eigenvalues, eigenvectors = fit_pca(flatten_patches(feats))

    depth_small = depth_at(depths, (h, w))  # [N, 1, h, w]
    projections = torch.einsum("nchw,ck->nkhw", feats - mean[None, :, None, None], eigenvectors[:, :TOP_PCS])

    best_abs_rho, pc1_abs_rho, ious = [], [], []
    for index in range(n):
        depth_flat = depth_small[index, 0].flatten()
        rhos = [abs(spearman(projections[index, k].flatten(), depth_flat)) for k in range(TOP_PCS)]
        best_abs_rho.append(max(rhos))
        pc1_abs_rho.append(rhos[0])
        ious.append(median_split_iou(projections[index, 0], depth_small[index, 0]))

    # MI is estimated once over all patches (the shared basis makes a PC's value
    # comparable across images), so the histogram is densely populated.
    depth_all = depth_small[:, 0].reshape(-1)
    mi_per_pc = [mutual_information(projections[:, k].reshape(-1), depth_all) for k in range(TOP_PCS)]

    best_abs_rho = torch.tensor(best_abs_rho)
    metrics = {
        "channels": c,
        "patch_effective_rank": effective_rank(eigenvalues),
        "patch_participation_ratio": participation_ratio(eigenvalues),
        "var_explained_top3": (eigenvalues[:3].sum() / eigenvalues.sum()).item(),
        "spearman_best_pc_mean": best_abs_rho.mean().item(),
        "spearman_best_pc_std": best_abs_rho.std().item(),
        "spearman_pc1_mean": torch.tensor(pc1_abs_rho).mean().item(),
        "fg_bg_iou_mean": torch.tensor(ious).mean().item(),
        "mi_best_pc": max(mi_per_pc),
        "mi_pc1": mi_per_pc[0],
    }
    artifacts = {"projections": projections, "best_abs_rho": best_abs_rho}
    return metrics, artifacts


def pooled_spectrum(feats_final: torch.Tensor) -> tuple[torch.Tensor, dict]:
    pooled = feats_final.mean(dim=[2, 3])
    _, eigenvalues, _ = fit_pca(pooled)
    return eigenvalues, {
        "pooled_effective_rank": effective_rank(eigenvalues),
        "pooled_participation_ratio": participation_ratio(eigenvalues),
        "pooled_dim": pooled.shape[1],
    }


# --------------------------------------------------------------------------- #
# Visualization helpers (analysis side: prepare render-ready arrays)
# --------------------------------------------------------------------------- #

def robust_rgb(projection: torch.Tensor) -> torch.Tensor:
    """[3, h, w] PC projections -> [h, w, 3] in [0, 1] via per-channel 2-98 pct."""
    channels = []
    for channel in projection:
        lo, hi = torch.quantile(channel.flatten(), torch.tensor([0.02, 0.98]))
        channels.append(((channel - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1))
    return torch.stack(channels, dim=-1)


def orient_pc1(projections: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """PC1 maps [num, h, w] sign-flipped so larger = farther, per image."""
    h, w = projections.shape[-2:]
    depth_small = depth_at(depths, (h, w))[:, 0]
    oriented = []
    for index in range(projections.shape[0]):
        pc1 = projections[index, 0]
        if spearman(pc1.flatten(), depth_small[index].flatten()) < 0:
            pc1 = -pc1
        oriented.append(pc1)
    return torch.stack(oriented)


def load_raw_images(dataset: NormalizedNyuDataset, count: int) -> list:
    images = []
    for index in range(count):
        path = dataset._resolve_sample_path(dataset.samples.iloc[index]["image_path"])
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def save_figures(output: Path, encoders, spectra, artifacts, metrics, raw_images, viz_depths):
    names = list(encoders)
    num = len(raw_images)
    depth_display = viz_depths[:num, 0]

    eranks = {name: metrics["encoders"][name]["pooled_effective_rank"] for name in names}
    pooled_dim = metrics["encoders"][names[0]]["pooled_dim"]
    pca_plots.plot_spectrum(spectra, eranks, pooled_dim, output / "fig_spectrum.png")

    best_rho = {name: artifacts[name]["y3"]["best_abs_rho"] for name in names}
    pca_plots.plot_depth_corr(best_rho, "y3", TOP_PCS, output / "fig_depth_corr.png")

    for level in LEVELS:
        rgb = {
            name: torch.stack([robust_rgb(artifacts[name][level]["projections"][i, :3]) for i in range(num)])
            for name in names
        }
        pca_plots.plot_pca_maps(level, raw_images, depth_display, rgb, output / f"fig_pca_maps_{level}.png")

    # PC1-vs-depth is most meaningful at the bottleneck LeJEPA actually shapes.
    pc1 = {name: orient_pc1(artifacts[name]["final"]["projections"][:num], viz_depths[:num]) for name in names}
    pca_plots.plot_pc1_depth("final", raw_images, depth_display, pc1, output / "fig_pc1_depth.png")


def write_report(metrics: dict, output: Path):
    lines = [
        "# PCA probing report",
        "",
        f"arch=`{metrics['arch']}` dataset=`{metrics['dataset']}` images=`{metrics['num_images']}`",
        "",
        "| encoder | level | erank (patch) | top-3 var | best-PC |rho| | PC1 |rho| | fg/bg IoU | MI best | probe R2 | probe RMSE (m) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, encoder_metrics in metrics["encoders"].items():
        for level in LEVELS:
            m = encoder_metrics[level]
            probe = m.get("probe", {})
            lines.append(
                f"| {name} | {level} | {m['patch_effective_rank']:.1f}/{m['channels']} "
                f"| {m['var_explained_top3']:.2f} | {m['spearman_best_pc_mean']:.3f} "
                f"| {m['spearman_pc1_mean']:.3f} | {m['fg_bg_iou_mean']:.3f} | {m['mi_best_pc']:.3f} "
                f"| {probe.get('r2', float('nan')):.3f} | {probe.get('rmse_m', float('nan')):.3f} |"
            )
    lines += ["", "| encoder | pooled erank | participation ratio |", "|---|---|---|"]
    for name, encoder_metrics in metrics["encoders"].items():
        lines.append(
            f"| {name} | {encoder_metrics['pooled_effective_rank']:.1f} "
            f"| {encoder_metrics['pooled_participation_ratio']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def make_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=globals.DATALOADER_WORKERS, persistent_workers=False,
    )


def main():
    args = parse_args()
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(Config.DEFAULT)
    args.output.mkdir(parents=True, exist_ok=True)

    encoders = build_encoders(device, args.dataset, args.arch)

    test_ds = NormalizedNyuDataset("test")
    test_loader = make_loader(test_ds, args.batch_size)

    train_ds = NormalizedNyuDataset("train")
    generator = torch.Generator().manual_seed(globals.SEED)
    probe_indices = torch.randperm(len(train_ds), generator=generator)[: args.probe_train].tolist()
    probe_loader = make_loader(Subset(train_ds, probe_indices), args.batch_size)

    metrics = {"arch": args.arch, "dataset": args.dataset, "encoders": {}}
    artifacts = {}
    spectra = {}
    test_depths = None

    for name, encoder in encoders.items():
        feats, depths_raw = extract_features(encoder, test_loader, device, args.limit, desc=f"{name}: test features")
        test_depths = depths_raw * DEPTH_TO_METERS

        probe_feats, probe_depths_raw = extract_features(encoder, probe_loader, device, desc=f"{name}: probe-train features")
        probe_depths = probe_depths_raw * DEPTH_TO_METERS

        spectra[name], encoder_metrics = pooled_spectrum(feats["final"])
        artifacts[name] = {}

        for level in LEVELS:
            level_metrics, level_artifacts = analyze_level(feats[level], test_depths)
            size = feats[level].shape[-2:]
            level_metrics["probe"] = ridge_probe(
                flatten_patches(probe_feats[level]),
                depth_at(probe_depths, size).flatten(),
                flatten_patches(feats[level]),
                depth_at(test_depths, size).flatten(),
            )
            encoder_metrics[level] = level_metrics
            artifacts[name][level] = level_artifacts
            logger.info(
                f"[{name}/{level}] erank={level_metrics['patch_effective_rank']:.1f}/{level_metrics['channels']} "
                f"best|rho|={level_metrics['spearman_best_pc_mean']:.3f} "
                f"IoU={level_metrics['fg_bg_iou_mean']:.3f} MI={level_metrics['mi_best_pc']:.3f} "
                f"probe R2={level_metrics['probe']['r2']:.3f}"
            )

        metrics["encoders"][name] = encoder_metrics

    metrics["num_images"] = test_depths.shape[0]

    raw_images = load_raw_images(test_ds, min(args.num_viz, len(test_ds)))
    save_figures(args.output, encoders, spectra, artifacts, metrics, raw_images, test_depths)

    with open(args.output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_report(metrics, args.output)
    logger.info(f"wrote figures, metrics.json and report.md to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
