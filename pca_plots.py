from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_spectrum(spectra: dict[str, torch.Tensor], eranks: dict[str, float], pooled_dim: int, path: Path):
    """Pooled-embedding eigenspectra and cumulative explained variance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, eigenvalues in spectra.items():
        axes[0].semilogy(eigenvalues / eigenvalues[0], label=f"{name} (erank={eranks[name]:.1f})")
        axes[1].plot(eigenvalues.cumsum(0) / eigenvalues.sum(), label=name)
    axes[0].set_title(f"Pooled embedding eigenspectrum (final, {pooled_dim}-d)")
    axes[0].set_xlabel("component"), axes[0].set_ylabel("eigenvalue / max")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("component"), axes[1].set_ylabel("fraction")
    axes[1].axhline(0.95, color="gray", ls=":", lw=0.8)
    for ax in axes:
        ax.legend(), ax.grid(alpha=0.3)
    _save(fig, path)


def plot_pca_maps(level: str, raw_images: list, depth_maps: torch.Tensor, rgb_per_encoder: dict[str, torch.Tensor], path: Path):
    """Top-3 PCs rendered as RGB next to the input image and GT depth."""
    names = list(rgb_per_encoder)
    fig, axes = _grid(len(raw_images), 2 + len(names))
    for row in range(len(raw_images)):
        axes[row, 0].imshow(raw_images[row])
        axes[row, 1].imshow(depth_maps[row], cmap="magma")
        for col, name in enumerate(names, start=2):
            axes[row, col].imshow(rgb_per_encoder[name][row], interpolation="nearest")
    _label_columns(axes, ["RGB", "GT depth"] + [f"{name} PCA" for name in names])
    fig.suptitle(f"Patch-feature PCA (top-3 PCs as RGB) — level {level}")
    _save(fig, path)


def plot_pc1_depth(level: str, raw_images: list, depth_maps: torch.Tensor, pc1_per_encoder: dict[str, torch.Tensor], path: Path):
    """PC1 as a signed heatmap (already sign-aligned to depth upstream): the
    picture behind the fg/bg IoU metric — a good encoder's leading direction
    tracks near/far."""
    names = list(pc1_per_encoder)
    fig, axes = _grid(len(raw_images), 2 + len(names))
    for row in range(len(raw_images)):
        axes[row, 0].imshow(raw_images[row])
        axes[row, 1].imshow(depth_maps[row], cmap="magma")
        for col, name in enumerate(names, start=2):
            pc1 = pc1_per_encoder[name][row]
            limit = pc1.abs().flatten().quantile(0.98).clamp_min(1e-8)
            axes[row, col].imshow(pc1, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    _label_columns(axes, ["RGB", "GT depth"] + [f"{name} PC1" for name in names])
    fig.suptitle(f"PC1 sign-aligned to depth (blue→red = near→far) — level {level}")
    _save(fig, path)


def plot_depth_corr(best_abs_rho_per_encoder: dict[str, torch.Tensor], level: str, top_pcs: int, path: Path):
    """Distribution of per-image |Spearman| between the best PC and GT depth."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rho in best_abs_rho_per_encoder.items():
        ax.hist(rho.numpy(), bins=30, range=(0, 1), alpha=0.5, label=name, density=True)
    ax.set_title(f"Per-image |Spearman| between best of top-{top_pcs} PCs and GT depth ({level})")
    ax.set_xlabel("|rho|"), ax.set_ylabel("density")
    ax.legend(), ax.grid(alpha=0.3)
    _save(fig, path)


def _grid(rows: int, cols: int):
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.9 * rows))
    axes = axes.reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def _label_columns(axes, titles: list[str]):
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
