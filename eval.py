import torch
import torch.nn as nn
import torch.nn.functional as TF

import logger
import globals
from tqdm import tqdm
from hardware_acceleration import enable_hardware_acceleration, Config
from meter import Meter
from dataset import build_dataset
from torch.utils.data import DataLoader
from torch import Tensor
from dataset import NormalizedNyuDataset

def valid_depth_mask(y: Tensor, z: Tensor) -> Tensor:
    return (y > 0) & torch.isfinite(y) & torch.isfinite(z)


def delta1(y: Tensor, z: Tensor, mask: Tensor, threshold=1.25, eps=1e-8) -> float:
    y = y[mask]
    z = z[mask]
    if y.numel() == 0:
        return float("nan")
    y = y.clamp_min(eps)
    z = z.clamp_min(eps)
    ratio = torch.max(z / (y + eps), y / (z + eps))
    return torch.mean((ratio < threshold).float()).item()


def rel(y: Tensor, z: Tensor, mask: Tensor, eps=1e-8) -> float:
    y = y[mask]
    z = z[mask]
    if y.numel() == 0:
        return float("nan")
    return torch.mean(torch.abs(z - y) / (y + eps)).item()


def rmse(y: Tensor, z: Tensor, mask: Tensor) -> float:
    y = y[mask]
    z = z[mask]
    if y.numel() == 0:
        return float("nan")
    return ((y - z).square().mean()).sqrt().item()


@torch.no_grad()
def eval_model(
    model: nn.Module,
    dataset: NormalizedNyuDataset,
    device: torch.device,
):
    model.to(device)
    model.eval()

    total_delta1 = 0.0
    total_rel = 0.0
    total_rmse = 0.0
    valid_items = 0

    test_dataset = DataLoader(dataset)
    for item in tqdm(test_dataset, total=len(test_dataset)):
        x, y = item["image"].to(device), item["depth"].to(device)

        z = model(x)
        z = z.float()
        z = TF.interpolate(
            z,
            size=globals.NYU_IMAGE_RESOLUTION,
            mode="bilinear",
            align_corners=False,
        )

        y = y.float()
        mask = valid_depth_mask(y, z)
        if not mask.any():
            continue

        total_delta1 += delta1(y, z, mask)
        total_rel += rel(y, z, mask)
        total_rmse += rmse(y, z, mask)
        valid_items += 1

    if valid_items == 0:
        raise ValueError("Evaluation dataset did not contain any valid depth pixels.")

    logger.info(f"RMSE = {total_rmse / (100 * valid_items):.3f}")
    logger.info(f"REL = {total_rel / valid_items:.3f}")
    logger.info(f"δ1 = {total_delta1 / valid_items:.3f}")


def main():
    device = enable_hardware_acceleration(Config.DEFAULT)
    model = Meter(device, "xxs")
    state_dict = torch.load("meter-models/build_model_best_nyu_xxs", map_location=device)
    model.load_state_dict(state_dict)

    dataset = build_dataset("test")
    eval_model(model, dataset, device)


if __name__ == "__main__":
    raise SystemExit(main())
