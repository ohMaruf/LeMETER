import shutil
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm

import globals
import logger
from globals import INPUT_RESOLUTION, NYU_IMAGE_RESOLUTION
from hardware_acceleration import enable_hardware_acceleration, Config

NYU_DATASET_NAME = "nyu-depth-v2"
NYU_DATASET_ROOT = Path("datasets/nyu_data")
NYU_TRAIN_CSV = NYU_DATASET_ROOT / Path("data/nyu2_train.csv")
NYU_TEST_CSV = NYU_DATASET_ROOT / Path("data/nyu2_test.csv")
NYU_PREPROCESSED_ROOT = Path("preprocessed_datasets/nyu-depth-v2")
NYU_COMPLETE_MARKER = Path("preprocessed_datasets/.complete") / NYU_DATASET_NAME


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


def preprocess_nyu_pair(image_rel_path: str, depth_rel_path: str, device: torch.device) -> None:
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
        image.to(device, dtype=globals.FLOATING_PRECISION),
        list(INPUT_RESOLUTION),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    save_image_tensor(
        resized.clamp_(0, 255).round_(),
        target_image_path,
    )
    copy_with_parents(source_depth_path, target_depth_path)


def write_nyu_completion_marker() -> Path:
    NYU_COMPLETE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    NYU_COMPLETE_MARKER.write_text("complete\n", encoding="utf-8")
    return NYU_COMPLETE_MARKER


def preprocess_nyu_depth_v2(device: torch.device) -> None:
    if NYU_COMPLETE_MARKER.exists():
        logger.info(f"Skipping {NYU_DATASET_NAME}: completion marker exists at {NYU_COMPLETE_MARKER}")
        logger.info(f"Preprocessing of {NYU_DATASET_NAME} is complete")
        return

    train_df = cast(pd.DataFrame, pd.read_csv(NYU_TRAIN_CSV, header=None))
    test_df = cast(pd.DataFrame, pd.read_csv(NYU_TEST_CSV, header=None))

    logger.info(f"Preprocessing train split of {NYU_DATASET_NAME}...")
    for row in tqdm(train_df.itertuples(), total=len(train_df)):
        preprocess_nyu_pair(*row[1:], device=device)

    logger.info(f"Preprocessing test split of {NYU_DATASET_NAME}...")
    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        preprocess_nyu_pair(*row[1:], device=device)

    total_pairs = len(train_df) + len(test_df)
    marker = write_nyu_completion_marker()
    logger.log(f"Preprocessed {total_pairs} pairs into {NYU_PREPROCESSED_ROOT}")
    logger.log(f"Wrote completion marker: {marker}")
    logger.log(f"Preprocessing of {NYU_DATASET_NAME} is complete")


def preprocess_datasets() -> None:
    device = enable_hardware_acceleration(Config.DEFAULT)
    preprocess_nyu_depth_v2(device)


if __name__ == "__main__":
    preprocess_datasets()
