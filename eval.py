import time

import torch
import torch.nn as nn
import torch.nn.functional as TF

import logger
import globals
from tqdm import tqdm
from hardware_acceleration import enable_hardware_acceleration, Config
from meter import Meter
from torch.utils.data import DataLoader
from torch import Tensor
from dataset import NormalizedNyuDataset

MIN_DEPTH_CM = 0.1
MAX_DEPTH_CM = 1000.0
# Eigen et al. center crop for the 480x640 NYU frame: (top, bottom, left,
# right). The Kinect projection makes the frame borders unreliable, so every
# NYU MDE paper scores metrics only inside this rectangle.
EIGEN_CROP = (45, 471, 41, 601)


def eigen_crop_mask(height: int, width: int, device: torch.device) -> Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    top, bottom, left, right = EIGEN_CROP
    mask[top:bottom, left:right] = True
    return mask


def valid_depth_mask(y: Tensor, z: Tensor) -> Tensor:
    # only score pixels whose GT is inside the sensor's valid range...
    mask = (y > MIN_DEPTH_CM) & (y < MAX_DEPTH_CM) & torch.isfinite(y) & torch.isfinite(z)
    # ...and only inside the Eigen center crop (defined for the full NYU frame)
    height, width = y.shape[-2], y.shape[-1]
    if (height, width) == tuple(globals.NYU_IMAGE_RESOLUTION):
        mask = mask & eigen_crop_mask(height, width, y.device)
    return mask

def depth_metrics(y: Tensor, z: Tensor) -> dict[str, float]:
    z = z.clamp(MIN_DEPTH_CM, MAX_DEPTH_CM)
    mask = valid_depth_mask(y, z)
    return {
        "rmse": rmse(y, z, mask) / 100.0,  # centimeters -> meters
        "rel": rel(y, z, mask),
        "delta1": delta1(y, z, mask),
    }


def delta1(
    y: Tensor,
    z: Tensor,
    mask: Tensor | None = None,
    threshold=1.25,
    eps=1e-8,
) -> float:
    if mask is not None:
        y = y[mask]
        z = z[mask]
    if y.numel() == 0:
        return float("nan")
    y = y.clamp_min(eps)
    z = z.clamp_min(eps)
    ratio = torch.max(z / (y + eps), y / (z + eps))
    return torch.mean((ratio < threshold).float()).item()


def rel(
    y: Tensor,
    z: Tensor,
    mask: Tensor | None = None,
    eps=1e-8,
) -> float:
    if mask is not None:
        y = y[mask]
        z = z[mask]
    if y.numel() == 0:
        return float("nan")
    return torch.mean(torch.abs(z - y) / (y + eps)).item()


def rmse(
    y: Tensor,
    z: Tensor,
    mask: Tensor | None = None,
) -> float:
    if mask is not None:
        y = y[mask]
        z = z[mask]
    if y.numel() == 0:
        return float("nan")
    return ((y - z).square().mean()).sqrt().item()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def run_inference(model: nn.Module, x: Tensor) -> Tensor:
    # model and labels are both in centimeters: no unit conversion needed
    z = model(x).float()
    return TF.interpolate(
        z,
        size=globals.NYU_IMAGE_RESOLUTION,
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def benchmark_accuracy(
    model: nn.Module,
    dataset: NormalizedNyuDataset,
    device: torch.device,
) -> None:
    model.to(device)
    model.eval()

    total_delta1 = 0.0
    total_rel = 0.0
    total_rmse = 0.0

    test_dataset = DataLoader(dataset)
    count = 0
    for item in tqdm(test_dataset, total=len(test_dataset)):
        x, y = item["image"].to(device), item["depth"].to(device).float()
        z = run_inference(model, x)
        for index in range(x.shape[0]):
            metrics = depth_metrics(y[index], z[index])
            total_rmse += metrics["rmse"]  # already in meters
            total_rel += metrics["rel"]
            total_delta1 += metrics["delta1"]
            count += 1

    logger.info(f"test/RMSE = {total_rmse / count:.3f}")
    logger.info(f"test/REL = {total_rel / count:.3f}")
    logger.info(f"test/δ1 = {total_delta1 / count:.3f}")


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    dataset: NormalizedNyuDataset,
    device: torch.device,
    *,
    warmup_steps: int = 5,
) -> float:
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset)

    logger.info(f"Running inference benchmark on {device.type}...")
    for step, item in enumerate(
        tqdm(
            dataloader,
            total=min(warmup_steps, len(dataloader)),
            desc=f"{device.type}-fps",
        )
    ):
        x = item["image"].to(device)
        run_inference(model, x)
        if step + 1 > warmup_steps:
            break

    synchronize_device(device)
    start_time = time.perf_counter()
    measured_items = 0
    for item in tqdm(dataloader, total=len(dataloader), desc=f"{device.type}-fps"):
        x = item["image"].to(device)
        run_inference(model, x)
        measured_items += x.shape[0]
    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    fps = measured_items / elapsed_seconds
    logger.info(f"{device.type.upper()} inference FPS = {fps:.2f}")
    return fps


def main():
    dataset = NormalizedNyuDataset("test")
    device = enable_hardware_acceleration(Config.DEFAULT)

    model = Meter.load(device, "nyu", "xxs")

    benchmark_accuracy(model, dataset, device)
    benchmark_inference(model, dataset, device)
    benchmark_inference(model, dataset, torch.device("cpu"))


if __name__ == "__main__":
    raise SystemExit(main())
