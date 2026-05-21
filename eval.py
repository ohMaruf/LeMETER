import torch
import torch.nn as nn
import torch.nn.functional as TF
import logger
import globals
import tqdm
from hardware_acceleration import enable_hardware_acceleration, Config
from meter import Meter
from dataset import build_dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torchprofile import profile_macs
from torchprofile.ir import Node
from dataset import PreprocessedDepthDataset


def delta1(y, z, threshold=1.25, eps=1e-8) -> float:
    ratio = torch.max(z / (y + eps), y / (z + eps))
    return torch.mean((ratio < threshold).float()).item()


def mac(
    model: nn.Module,
    shape: tuple,
    device: torch.Device,
) -> int | dict[Node, int]:
    inputs = torch.randn(shape).to(device)
    return profile_macs(model, inputs)


def rel(y: Tensor, z: Tensor, eps=1e-8) -> float:
    return torch.mean(torch.abs(z - y) / (y + eps)).item()


def rmse(y: Tensor, z: Tensor) -> float:
    return ((y - z).square().mean()).sqrt().item()


@torch.no_grad()
def eval_model(
    model: nn.Module,
    dataset: PreprocessedDepthDataset,
    device: torch.Device,
):
    model.to(device)
    model.eval()

    total_delta1 = 0.0
    total_rel = 0.0
    total_rmse = 0.0

    test_dataset = DataLoader(dataset)
    for item in tqdm.tqdm(test_dataset, total=len(test_dataset)):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            x, y = item["image"].to(device), item["depth"].to(device)
            z = model(x)

            y = TF.interpolate(
                y,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            ).to(device)

            total_delta1 += delta1(y, z)
            total_rel += rel(y, z)
            total_rmse += rmse(y, z)

    logger.info(f"RMSE = {total_rmse / len(test_dataset):.3f}")
    logger.info(f"REL = {total_rel / len(test_dataset):.3f}")
    logger.info(f"δ1 = {total_delta1 / len(test_dataset):.3f}")

    macs = mac(
        model, (1, 3, globals.INPUT_RESOLUTION[0], globals.INPUT_RESOLUTION[1]), device
    )
    if isinstance(macs, int):
        logger.info(f"GigaMACs: {macs / 1e9:.3f}")


def main():
    device = enable_hardware_acceleration(Config.RX9060XT)
    model = Meter(device, "xxs")
    state_dict = torch.load("meter-models/build_model_best_nyu_xxs")
    model.load_state_dict(state_dict)

    dataset = build_dataset("test")
    eval_model(model, dataset, device)


if __name__ == "__main__":
    raise SystemExit(main())
