import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augmentation import augmentation2D
from model import IsoEncoder, IsoPredictor, IsoProjectionHead, LeMETER


DEFAULT_DATA_ROOT = Path("preprocessed_datasets/nyu-depth-v2")
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "data/nyu2_train.csv"
DEFAULT_VAL_CSV = DEFAULT_DATA_ROOT / "data/nyu2_test.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage LeMETER training: "
            "1) LeJEPA pretrain IsoEncoder, "
            "2) freeze encoder and train decoder on NYU-depth-v2."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lemeter"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-slices", type=int, default=128)
    parser.add_argument("--sigreg-points", type=int, default=17)
    parser.add_argument("--sigreg-t-max", type=float, default=3.0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_pairs(csv_path: Path) -> list[tuple[str, str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        return [(row[0], row[1]) for row in reader if row]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array


def load_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    if depth.max() > 255.0:
        depth = depth / 1000.0
    elif depth.max() > 1.0:
        depth = depth / 255.0

    return depth[..., None]


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def depth_to_tensor(depth: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(depth.transpose(2, 0, 1))).float()


class NYUDepthV2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        csv_path: Path,
        *,
        ssl_views: bool = False,
    ):
        self.data_root = data_root
        self.pairs = read_pairs(csv_path)
        self.ssl_views = ssl_views

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_rel_path, depth_rel_path = self.pairs[index]
        image = load_rgb(self.data_root / image_rel_path)

        if self.ssl_views:
            dummy_depth = np.zeros((*image.shape[:2], 1), dtype=np.float32)
            view_a, _ = augmentation2D(image.copy(), dummy_depth.copy(), False)
            view_b, _ = augmentation2D(image.copy(), dummy_depth.copy(), False)
            return {
                "view_a": image_to_tensor(view_a),
                "view_b": image_to_tensor(view_b),
            }

        depth = load_depth(self.data_root / depth_rel_path)
        return {
            "image": image_to_tensor(image),
            "depth": depth_to_tensor(depth),
        }

def build_sigreg_loss(args) -> nn.Module:
    try:
        import lejepa
    except ImportError as exc:
        raise ImportError(
            "LeJEPA SIGReg is now delegated to the upstream `lejepa` package. "
            "Install it from GitHub with "
            "`python3 -m pip install \"git+https://github.com/galilai-group/lejepa.git\"`."
        ) from exc

    univariate_test = lejepa.univariate.EppsPulley(
        t_max=args.sigreg_t_max,
        n_points=args.sigreg_points,
    )
    return lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=args.sigreg_slices,
        reduction="mean",
    )


class IsoEncoderPretrainer(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = IsoEncoder()
        self.projection_head = IsoProjectionHead(
            in_ch=96,
            hidden_ch=latent_dim,
            latent_dim=latent_dim,
        )
        self.predictor = IsoPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        tokens, _ = self.projection_head(features["z"])
        return tokens

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens_a = self.encode_tokens(view_a)
        tokens_b = self.encode_tokens(view_b)
        pred_a = self.predictor(tokens_a)
        pred_b = self.predictor(tokens_b)

        return {
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "pred_a": pred_a,
            "pred_b": pred_b,
        }


def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.reshape(-1, tokens.size(-1))


def lejepa_loss(
    outputs: dict[str, torch.Tensor],
    sigreg: nn.Module,
    sigreg_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_loss = 0.5 * (
        F.mse_loss(outputs["pred_a"], outputs["tokens_b"])
        + F.mse_loss(outputs["pred_b"], outputs["tokens_a"])
    )

    sigreg_loss = 0.5 * (
        sigreg(flatten_tokens(outputs["tokens_a"]))
        + sigreg(flatten_tokens(outputs["tokens_b"]))
    )

    total = pred_loss + sigreg_weight * sigreg_loss
    metrics = {
        "total": float(total.detach().item()),
        "pred": float(pred_loss.detach().item()),
        "sigreg": float(sigreg_loss.detach().item()),
    }
    return total, metrics


def depth_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid_mask = target > 0
    if not valid_mask.any():
        return prediction.new_zeros(())
    return F.l1_loss(prediction[valid_mask], target[valid_mask])


def forward_with_frozen_encoder(model: LeMETER, images: torch.Tensor) -> torch.Tensor:
    model.encoder.eval()
    with torch.no_grad():
        features = model.encoder(images)

    depth_logits = model.decoder(features, output_size=images.shape[-2:])

    if model.use_bounded_depth:
        return model.min_depth + (
            model.max_depth - model.min_depth
        ) * torch.sigmoid(depth_logits)

    return F.softplus(depth_logits) + 1e-6


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def train_ssl_epoch(
    model: IsoEncoderPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sigreg: nn.Module,
    sigreg_weight: float,
    device: torch.device,
    log_every: int,
) -> dict[str, float]:
    model.train()
    running_total = 0.0
    running_pred = 0.0
    running_sigreg = 0.0

    for step, batch in enumerate(loader, start=1):
        view_a = batch["view_a"].to(device, non_blocking=True)
        view_b = batch["view_b"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(view_a, view_b)
        loss, metrics = lejepa_loss(outputs, sigreg, sigreg_weight)
        loss.backward()
        optimizer.step()

        running_total += metrics["total"]
        running_pred += metrics["pred"]
        running_sigreg += metrics["sigreg"]

        if step % log_every == 0 or step == len(loader):
            print(
                f"[pretrain] step {step:04d}/{len(loader):04d} "
                f"loss={running_total / step:.4f} "
                f"pred={running_pred / step:.4f} "
                f"sigreg={running_sigreg / step:.4f}"
            )

    num_steps = max(len(loader), 1)
    return {
        "loss": running_total / num_steps,
        "pred": running_pred / num_steps,
        "sigreg": running_sigreg / num_steps,
    }


def train_decoder_epoch(
    model: LeMETER,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int,
) -> dict[str, float]:
    model.decoder.train()
    running_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        target_depth = batch["depth"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = forward_with_frozen_encoder(model, images)
        loss = depth_loss(prediction, target_depth)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().item())

        if step % log_every == 0 or step == len(loader):
            print(
                f"[decoder] step {step:04d}/{len(loader):04d} "
                f"loss={running_loss / step:.4f}"
            )

    return {"loss": running_loss / max(len(loader), 1)}


@torch.no_grad()
def evaluate_decoder(
    model: LeMETER,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.decoder.eval()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target_depth = batch["depth"].to(device, non_blocking=True)
        prediction = forward_with_frozen_encoder(model, images)
        running_loss += float(depth_loss(prediction, target_depth).item())

    return {"loss": running_loss / max(len(loader), 1)}


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def freeze_encoder(model: LeMETER) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False


def maybe_load_pretrained_encoder(
    pretrainer: IsoEncoderPretrainer,
    checkpoint_path: Path | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_state = checkpoint.get("encoder", checkpoint)
    pretrainer.encoder.load_state_dict(encoder_state)

    projection_state = checkpoint.get("projection_head")
    if projection_state is not None:
        pretrainer.projection_head.load_state_dict(projection_state)

    predictor_state = checkpoint.get("predictor")
    if predictor_state is not None:
        pretrainer.predictor.load_state_dict(predictor_state)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Train manifest: {args.train_csv}")
    print(f"Val manifest:   {args.val_csv}")

    ssl_dataset = NYUDepthV2Dataset(args.data_root, args.train_csv, ssl_views=True)
    train_dataset = NYUDepthV2Dataset(args.data_root, args.train_csv, ssl_views=False)
    val_dataset = NYUDepthV2Dataset(args.data_root, args.val_csv, ssl_views=False)

    ssl_loader = build_loader(
        ssl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pretrainer = IsoEncoderPretrainer(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    maybe_load_pretrained_encoder(pretrainer, args.encoder_checkpoint, device)

    sigreg = build_sigreg_loss(args).to(device)

    metrics_log = {
        "pretrain": [],
        "decoder": [],
        "config": vars(args).copy(),
    }
    metrics_log["config"]["data_root"] = str(metrics_log["config"]["data_root"])
    metrics_log["config"]["train_csv"] = str(metrics_log["config"]["train_csv"])
    metrics_log["config"]["val_csv"] = str(metrics_log["config"]["val_csv"])
    metrics_log["config"]["output_dir"] = str(metrics_log["config"]["output_dir"])
    if metrics_log["config"]["encoder_checkpoint"] is not None:
        metrics_log["config"]["encoder_checkpoint"] = str(
            metrics_log["config"]["encoder_checkpoint"]
        )

    if not args.skip_pretrain:
        ssl_optimizer = torch.optim.AdamW(
            pretrainer.parameters(),
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
        )

        best_pretrain_loss = float("inf")
        for epoch in range(1, args.pretrain_epochs + 1):
            start = time.time()
            epoch_metrics = train_ssl_epoch(
                pretrainer,
                ssl_loader,
                ssl_optimizer,
                sigreg,
                args.sigreg_weight,
                device,
                args.log_every,
            )
            epoch_metrics["epoch"] = epoch
            epoch_metrics["seconds"] = time.time() - start
            metrics_log["pretrain"].append(epoch_metrics)
            print(
                f"[pretrain] epoch {epoch:03d}/{args.pretrain_epochs:03d} "
                f"loss={epoch_metrics['loss']:.4f} "
                f"pred={epoch_metrics['pred']:.4f} "
                f"sigreg={epoch_metrics['sigreg']:.4f} "
                f"time={epoch_metrics['seconds']:.1f}s"
            )

            checkpoint = {
                "epoch": epoch,
                "encoder": pretrainer.encoder.state_dict(),
                "projection_head": pretrainer.projection_head.state_dict(),
                "predictor": pretrainer.predictor.state_dict(),
                "optimizer": ssl_optimizer.state_dict(),
                "metrics": epoch_metrics,
                "config": metrics_log["config"],
            }
            save_checkpoint(args.output_dir / "encoder_last.pt", checkpoint)

            if epoch_metrics["loss"] < best_pretrain_loss:
                best_pretrain_loss = epoch_metrics["loss"]
                save_checkpoint(args.output_dir / "encoder_best.pt", checkpoint)
    elif args.encoder_checkpoint is None:
        raise ValueError(
            "--skip-pretrain requires --encoder-checkpoint because no encoder is trained in this run"
        )

    depth_model = LeMETER(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_bounded_depth=True,
    ).to(device)
    depth_model.encoder.load_state_dict(pretrainer.encoder.state_dict())
    freeze_encoder(depth_model)

    decoder_optimizer = torch.optim.AdamW(
        depth_model.decoder.parameters(),
        lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.decoder_epochs + 1):
        start = time.time()
        train_metrics = train_decoder_epoch(
            depth_model,
            train_loader,
            decoder_optimizer,
            device,
            args.log_every,
        )
        val_metrics = evaluate_decoder(depth_model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "seconds": time.time() - start,
        }
        metrics_log["decoder"].append(epoch_metrics)

        print(
            f"[decoder] epoch {epoch:03d}/{args.decoder_epochs:03d} "
            f"train={epoch_metrics['train_loss']:.4f} "
            f"val={epoch_metrics['val_loss']:.4f} "
            f"time={epoch_metrics['seconds']:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model": depth_model.state_dict(),
            "encoder": depth_model.encoder.state_dict(),
            "decoder": depth_model.decoder.state_dict(),
            "optimizer": decoder_optimizer.state_dict(),
            "metrics": epoch_metrics,
            "config": metrics_log["config"],
        }
        save_checkpoint(args.output_dir / "lemeter_last.pt", checkpoint)

        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            save_checkpoint(args.output_dir / "lemeter_best.pt", checkpoint)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
