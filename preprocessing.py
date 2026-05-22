import json
import shutil
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm

import logger
from globals import INPUT_RESOLUTION, NYU_IMAGE_RESOLUTION

NYU_DATASET_NAME = "nyu-depth-v2"
NYU_DATASET_ROOT = Path("datasets/nyu_data")
NYU_TRAIN_CSV = NYU_DATASET_ROOT / Path("data/nyu2_train.csv")
NYU_TEST_CSV = NYU_DATASET_ROOT / Path("data/nyu2_test.csv")
NYU_PREPROCESSED_ROOT = Path("preprocessed_datasets/nyu-depth-v2")
NYU_COMPLETE_MARKER = Path("preprocessed_datasets/.complete") / NYU_DATASET_NAME
NYU_STATS_PATH = NYU_PREPROCESSED_ROOT / "stats.json"


class ChannelStats:
    def __init__(self) -> None:
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.sum_sq = torch.zeros(3, dtype=torch.float64)
        self.pixel_count = 0

    def update(self, image: torch.Tensor) -> None:
        image = image.detach().to("cpu", dtype=torch.float64) / 255.0
        self.sum += image.sum(dim=(1, 2))
        self.sum_sq += image.square().sum(dim=(1, 2))
        self.pixel_count += image.shape[1] * image.shape[2]

    def as_dict(self) -> dict[str, object]:
        if self.pixel_count == 0:
            raise ValueError("Cannot compute channel stats with zero pixels")

        mean = self.sum / self.pixel_count
        variance = (self.sum_sq / self.pixel_count) - mean.square()
        std = variance.clamp_min(0.0).sqrt()
        return {
            "channels": ["R", "G", "B"],
            "mean_rgb": mean.tolist(),
            "std_rgb": std.tolist(),
            "pixel_count_per_channel": self.pixel_count,
        }


def copy_with_parents(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def save_image_tensor(image: torch.Tensor, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.to("cpu", dtype=torch.uint8)
    if destination_path.suffix.lower() in {".jpg", ".jpeg"}:
        write_jpeg(image, str(destination_path), quality=100)
        return
    if destination_path.suffix.lower() == ".png":
        write_png(image, str(destination_path))
        return
    raise ValueError(f"Unsupported image format: {destination_path.suffix}")


def preprocess_nyu_pair(
    image_rel_path: str,
    depth_rel_path: str,
) -> torch.Tensor:
    source_image_path = NYU_DATASET_ROOT / image_rel_path
    source_depth_path = NYU_DATASET_ROOT / depth_rel_path
    target_image_path = NYU_PREPROCESSED_ROOT / image_rel_path
    target_depth_path = NYU_PREPROCESSED_ROOT / depth_rel_path

    for path, label in (
            (source_image_path, "RGB image"),
            (source_depth_path, "depth map"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    image = read_image(str(source_image_path), mode=ImageReadMode.RGB)
    if tuple(image.shape[-2:]) != NYU_IMAGE_RESOLUTION:
        raise ValueError(
            f"Unexpected input size for {source_image_path}: "
            f"expected {NYU_IMAGE_RESOLUTION}, found {tuple(image.shape[-2:])}"
        )

    resized = resize(
        image.to(dtype=torch.float32),
        list(INPUT_RESOLUTION),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    save_image_tensor(
        resized.clamp_(0, 255).round_(),
        target_image_path,
    )
    copy_with_parents(source_depth_path, target_depth_path)
    return resized


def write_nyu_stats(stats: ChannelStats) -> Path:
    NYU_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": NYU_DATASET_NAME,
        "split": "train",
        "image_resolution": list(INPUT_RESOLUTION),
        **stats.as_dict(),
    }
    NYU_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return NYU_STATS_PATH


def write_nyu_completion_marker() -> Path:
    NYU_COMPLETE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    NYU_COMPLETE_MARKER.write_text("complete\n", encoding="utf-8")
    return NYU_COMPLETE_MARKER


def preprocess_nyu_depth_v2() -> None:
    if NYU_COMPLETE_MARKER.exists():
        logger.info(f"Skipping {NYU_DATASET_NAME}: completion marker exists at {NYU_COMPLETE_MARKER}")
        logger.info(f"Preprocessing of {NYU_DATASET_NAME} is complete")
        return

    train_df = cast(pd.DataFrame, pd.read_csv(NYU_TRAIN_CSV, header=None))
    test_df = cast(pd.DataFrame, pd.read_csv(NYU_TEST_CSV, header=None))
    train_stats = ChannelStats()

    logger.info(f"Preprocessing train split of {NYU_DATASET_NAME}...")
    for row in tqdm(train_df.itertuples(), total=len(train_df)):
        resized = preprocess_nyu_pair(*row[1:])
        train_stats.update(resized)

    logger.info(f"Preprocessing test split of {NYU_DATASET_NAME}...")
    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        preprocess_nyu_pair(*row[1:])

    total_pairs = len(train_df) + len(test_df)
    stats_path = write_nyu_stats(train_stats)
    marker = write_nyu_completion_marker()
    logger.log(f"Preprocessed {total_pairs} pairs into {NYU_PREPROCESSED_ROOT}")
    logger.log(f"Wrote RGB channel stats: {stats_path}")
    logger.log(f"Wrote completion marker: {marker}")
    logger.log(f"Preprocessing of {NYU_DATASET_NAME} is complete")


def preprocess_datasets() -> None:
    preprocess_nyu_depth_v2()


if __name__ == "__main__":
    preprocess_datasets()
