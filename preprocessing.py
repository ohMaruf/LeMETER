import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
NYU_WORKER_COUNT = min(8, os.cpu_count() or 1)
NYU_CHUNK_SIZE = 256
ManifestRow = tuple[str, str]


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

    def merge(self, other: "ChannelStats") -> None:
        self.sum += other.sum
        self.sum_sq += other.sum_sq
        self.pixel_count += other.pixel_count

    def to_payload(self) -> dict[str, object]:
        return {
            "sum": self.sum.tolist(),
            "sum_sq": self.sum_sq.tolist(),
            "pixel_count": self.pixel_count,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ChannelStats":
        stats = cls()
        stats.sum = torch.tensor(payload["sum"], dtype=torch.float64)
        stats.sum_sq = torch.tensor(payload["sum_sq"], dtype=torch.float64)
        stats.pixel_count = int(payload["pixel_count"])
        return stats


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


def load_manifest_rows(csv_path: Path) -> list[ManifestRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [tuple(row[:2]) for row in csv.reader(handle) if row]


def chunk_rows(rows: list[ManifestRow], chunk_size: int) -> list[list[ManifestRow]]:
    return [rows[index:index + chunk_size] for index in range(0, len(rows), chunk_size)]


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


def preprocess_nyu_chunk(
    rows: list[ManifestRow],
    *,
    collect_stats: bool,
) -> dict[str, object]:
    stats = ChannelStats()
    for image_rel_path, depth_rel_path in rows:
        resized = preprocess_nyu_pair(image_rel_path, depth_rel_path)
        if collect_stats:
            stats.update(resized)

    return {
        "processed_count": len(rows),
        "stats": stats.to_payload() if collect_stats else None,
    }


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


def copy_nyu_manifest(csv_path: Path) -> None:
    copy_with_parents(
        csv_path,
        NYU_PREPROCESSED_ROOT / csv_path.relative_to(NYU_DATASET_ROOT),
    )


def preprocess_split(
    rows: list[ManifestRow],
    *,
    split_name: str,
    collect_stats: bool,
) -> ChannelStats | None:
    if not rows:
        return ChannelStats() if collect_stats else None

    logger.info(f"Preprocessing {split_name} split of {NYU_DATASET_NAME}...")
    stats = ChannelStats() if collect_stats else None
    chunks = chunk_rows(rows, NYU_CHUNK_SIZE)

    with ProcessPoolExecutor(max_workers=NYU_WORKER_COUNT) as executor:
        futures = [
            executor.submit(preprocess_nyu_chunk, chunk, collect_stats=collect_stats)
            for chunk in chunks
        ]
        with tqdm(total=len(rows), desc=split_name) as progress:
            for future in as_completed(futures):
                result = future.result()
                progress.update(int(result["processed_count"]))
                if collect_stats:
                    stats_payload = result["stats"]
                    if stats_payload is not None and stats is not None:
                        stats.merge(ChannelStats.from_payload(stats_payload))

    return stats


def preprocess_nyu_depth_v2() -> None:
    if NYU_COMPLETE_MARKER.exists():
        logger.info(f"Skipping {NYU_DATASET_NAME}: completion marker exists at {NYU_COMPLETE_MARKER}")
        logger.info(f"Preprocessing of {NYU_DATASET_NAME} is complete")
        return

    copy_nyu_manifest(NYU_TRAIN_CSV)
    copy_nyu_manifest(NYU_TEST_CSV)

    train_rows = load_manifest_rows(NYU_TRAIN_CSV)
    test_rows = load_manifest_rows(NYU_TEST_CSV)

    train_stats = preprocess_split(train_rows, split_name="train", collect_stats=True)
    preprocess_split(test_rows, split_name="test", collect_stats=False)

    total_pairs = len(train_rows) + len(test_rows)
    if train_stats is None:
        raise ValueError("Train stats were not collected")

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
