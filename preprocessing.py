import csv
import json
import os
import shutil
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import logger
from globals import INPUT_RESOLUTION, NYU_IMAGE_RESOLUTION

# canonical preprocessed depth: uint16 centimeters for every split, matching the
# unit the Meter model predicts in. The source NYU train maps are uint8
# (255 == 10 m); the test maps are uint16 millimeters.
UINT8_DEPTH_TO_CM = 1000.0 / 255.0
MM_TO_CM = 0.1

NYU_DATASET_NAME = "nyu-depth-v2"
NYU_DATASET_ROOT = Path("datasets/nyu_data")
NYU_TRAIN_CSV = NYU_DATASET_ROOT / Path("data/nyu2_train.csv")
NYU_TEST_CSV = NYU_DATASET_ROOT / Path("data/nyu2_test.csv")
NYU_PREPROCESSED_ROOT = Path("preprocessed_datasets/nyu-depth-v2")
NYU_COMPLETE_MARKER = Path("preprocessed_datasets/.complete") / NYU_DATASET_NAME
NYU_STATS_PATH = NYU_PREPROCESSED_ROOT / "stats.json"
NYU_WORKER_COUNT = os.cpu_count() or 1
NYU_CHUNK_SIZE = 256
ManifestRow = tuple[str, str]

_DIR_CACHE: set[Path] = set()


def _ensure_dir(path: Path) -> None:
    if path not in _DIR_CACHE:
        path.mkdir(parents=True, exist_ok=True)
        _DIR_CACHE.add(path)


class ChannelStats:
    def __init__(self) -> None:
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.sum_sq = torch.zeros(3, dtype=torch.float64)
        self.pixel_count = 0

    def update(self, image: torch.Tensor) -> None:
        image = image.detach()
        self.sum += torch.sum(image, dim=(1, 2), dtype=torch.float64) / 255.0
        self.sum_sq += (
            torch.sum(image.to(torch.float64).square(), dim=(1, 2)) / (255.0**2)
        )
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
    _ensure_dir(destination_path.parent)
    shutil.copy2(source_path, destination_path)


def save_image_tensor(image: torch.Tensor, destination_path: Path) -> None:
    _ensure_dir(destination_path.parent)
    image = image.detach().clamp(0, 255).round().to(torch.uint8)
    pil_image = to_pil_image(image, mode="RGB")
    if destination_path.suffix.lower() in {".jpg", ".jpeg"}:
        pil_image.save(destination_path, quality=100)
        return
    if destination_path.suffix.lower() == ".png":
        pil_image.save(destination_path)
        return
    raise ValueError(f"Unsupported image format: {destination_path.suffix}")


def save_depth_as_cm(source_depth_path: Path, destination_path: Path) -> None:
    """Write the depth map as uint16 centimeters, the single canonical unit (the
    one the Meter model predicts in). Train maps arrive as uint8 (255 == 10 m)
    and test maps as uint16 millimeters; both are rescaled to cm. Resolution is
    preserved (depth is never resized)."""
    _ensure_dir(destination_path.parent)
    with Image.open(source_depth_path) as depth:
        array = np.asarray(depth)

    if array.dtype == np.uint8:
        # train: uint8 with 255 == 10 m -> centimeters
        array = np.rint(array.astype(np.float64) * UINT8_DEPTH_TO_CM)
    elif array.dtype == np.uint16:
        # test: uint16 millimeters -> centimeters
        array = np.rint(array.astype(np.float64) * MM_TO_CM)
    else:
        raise ValueError(
            f"Unexpected depth dtype {array.dtype} for {source_depth_path}; "
            "expected uint8 (train) or uint16 (test)"
        )

    Image.fromarray(array.astype(np.uint16)).save(destination_path)


def load_manifest_rows(csv_path: Path) -> list[ManifestRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [tuple(row[:2]) for row in csv.reader(handle) if row]


def chunk_rows(rows: list[ManifestRow], chunk_size: int) -> list[list[ManifestRow]]:
    return [
        rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)
    ]


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

    with Image.open(source_image_path) as image:
        if image.mode != "RGB":
            rgb_image = image.convert("RGB")
        else:
            rgb_image = image

        actual_resolution = (rgb_image.height, rgb_image.width)
        if actual_resolution != NYU_IMAGE_RESOLUTION:
            raise ValueError(
                f"Unexpected input size for {source_image_path}: "
                f"expected {NYU_IMAGE_RESOLUTION}, found {actual_resolution}"
            )

        resized = rgb_image.resize(
            (INPUT_RESOLUTION[1], INPUT_RESOLUTION[0]),
            resample=Image.Resampling.BICUBIC,
        )

        _ensure_dir(target_image_path.parent)
        if target_image_path.suffix.lower() in {".jpg", ".jpeg"}:
            resized.save(target_image_path, quality=100)
        elif target_image_path.suffix.lower() == ".png":
            resized.save(target_image_path)
        else:
            raise ValueError(f"Unsupported image format: {target_image_path.suffix}")

        resized_tensor = to_tensor(resized) * 255
    save_depth_as_cm(source_depth_path, target_depth_path)
    return resized_tensor


def preprocess_nyu_chunk(
    rows: list[ManifestRow],
    *,
    collect_stats: bool,
) -> dict[str, object]:
    torch.set_num_threads(1)
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
        logger.info(
            f"Skipping {NYU_DATASET_NAME}: completion marker exists at {NYU_COMPLETE_MARKER}"
        )
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
    logger.info(f"Preprocessed {total_pairs} pairs into {NYU_PREPROCESSED_ROOT}")
    logger.info(f"Wrote RGB channel stats: {stats_path}")
    logger.info(f"Wrote completion marker: {marker}")
    logger.info(f"Preprocessing of {NYU_DATASET_NAME} is complete")


# --------------------------------------------------------------------------- #
# ImageNet-100 (SSL pretraining source)
# --------------------------------------------------------------------------- #
# Aspect-preserving resize so the shorter side equals the long edge of the crop
# target (256 = max(INPUT_RESOLUTION)). This keeps the whole image (no on-disk
# crop), so the train-time RandomResizedCrop(192x256) can sample diverse
# sub-regions with minimal upsampling, while cutting disk/decode cost vs storing
# full-res JPEGs. Aspect ratio is normalized at train time by the crop, not here.
IMAGENET100_DATASET_NAME = "imagenet100"
IMAGENET100_HANDLE = "ambityga/imagenet100"
IMAGENET100_PREPROCESSED_ROOT = Path("preprocessed_datasets/imagenet100")
IMAGENET100_COMPLETE_MARKER = Path("preprocessed_datasets/.complete") / IMAGENET100_DATASET_NAME
IMAGENET100_STATS_PATH = IMAGENET100_PREPROCESSED_ROOT / "stats.json"
IMAGENET100_LABELS_PATH = IMAGENET100_PREPROCESSED_ROOT / "labels.json"
IMAGENET100_SHORT_SIDE = max(INPUT_RESOLUTION)  # 256
IMAGENET100_JPEG_QUALITY = 90
IMAGENET100_WORKER_COUNT = os.cpu_count() or 1
IMAGENET100_CHUNK_SIZE = 256
IMAGENET100_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
# wnid subfolders live under these top-level dirs in the kaggle archive
IMAGENET100_TRAIN_DIR_GLOB = "train*"
IMAGENET100_VAL_DIR_GLOB = "val*"

# (absolute source path, wnid, label index)
ImageNetSample = tuple[str, str, int]
# (relative output path, label index, wnid)
ImageNetRow = tuple[str, int, str]


def _resolve_imagenet100_raw_root() -> Path:
    """Locate the extracted kaggle archive (cached download, no re-fetch)."""
    import kagglehub

    return Path(kagglehub.dataset_download(IMAGENET100_HANDLE))


def _list_class_dirs(raw_root: Path, dir_glob: str) -> list[Path]:
    """All wnid subfolders under top-level dirs matching `dir_glob` (the archive
    spreads the train classes across train.X1..X4)."""
    class_dirs: list[Path] = []
    for top in sorted(raw_root.glob(dir_glob)):
        if top.is_dir():
            class_dirs.extend(child for child in sorted(top.iterdir()) if child.is_dir())
    return class_dirs


def _build_wnid_index(raw_root: Path) -> dict[str, int]:
    wnids = sorted({d.name for d in _list_class_dirs(raw_root, IMAGENET100_TRAIN_DIR_GLOB)})
    if not wnids:
        raise FileNotFoundError(
            f"No train class folders found under {raw_root} (looked for "
            f"'{IMAGENET100_TRAIN_DIR_GLOB}/<wnid>/'); is the archive extracted?"
        )
    return {wnid: index for index, wnid in enumerate(wnids)}


def _collect_imagenet_samples(raw_root: Path, dir_glob: str, wnid_to_idx: dict[str, int]) -> list[ImageNetSample]:
    samples: list[ImageNetSample] = []
    for class_dir in _list_class_dirs(raw_root, dir_glob):
        wnid = class_dir.name
        if wnid not in wnid_to_idx:
            continue  # a val-only class with no train index; skip
        label = wnid_to_idx[wnid]
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in IMAGENET100_IMAGE_EXTENSIONS:
                samples.append((str(file), wnid, label))
    return samples


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    scale = short_side / min(width, height)
    if scale == 1.0:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, resample=Image.Resampling.BICUBIC)


def preprocess_imagenet100_chunk(
    samples: list[ImageNetSample],
    split: str,
    *,
    collect_stats: bool,
) -> dict[str, object]:
    torch.set_num_threads(1)
    stats = ChannelStats()
    rows: list[ImageNetRow] = []
    for source_path, wnid, label in samples:
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            resized = _resize_short_side(rgb, IMAGENET100_SHORT_SIDE)

            relative_path = Path(split) / wnid / (Path(source_path).stem + ".jpg")
            destination = IMAGENET100_PREPROCESSED_ROOT / relative_path
            _ensure_dir(destination.parent)
            resized.save(destination, format="JPEG", quality=IMAGENET100_JPEG_QUALITY)

            if collect_stats:
                stats.update(to_tensor(resized) * 255)

        rows.append((relative_path.as_posix(), label, wnid))

    return {
        "processed_count": len(samples),
        "rows": rows,
        "stats": stats.to_payload() if collect_stats else None,
    }


def _preprocess_imagenet100_split(
    samples: list[ImageNetSample],
    *,
    split: str,
    collect_stats: bool,
) -> tuple[list[ImageNetRow], ChannelStats | None]:
    rows: list[ImageNetRow] = []
    stats = ChannelStats() if collect_stats else None
    if not samples:
        return rows, stats

    logger.info(f"Preprocessing {split} split of {IMAGENET100_DATASET_NAME} ({len(samples)} images)...")
    chunks = [
        samples[index : index + IMAGENET100_CHUNK_SIZE]
        for index in range(0, len(samples), IMAGENET100_CHUNK_SIZE)
    ]

    with ProcessPoolExecutor(max_workers=IMAGENET100_WORKER_COUNT) as executor:
        futures = [
            executor.submit(preprocess_imagenet100_chunk, chunk, split, collect_stats=collect_stats)
            for chunk in chunks
        ]
        with tqdm(total=len(samples), desc=split) as progress:
            for future in as_completed(futures):
                result = future.result()
                progress.update(int(result["processed_count"]))
                rows.extend(result["rows"])
                if collect_stats and stats is not None and result["stats"] is not None:
                    stats.merge(ChannelStats.from_payload(result["stats"]))

    rows.sort()  # deterministic manifest order regardless of future completion
    return rows, stats


def _write_imagenet100_manifest(rows: list[ImageNetRow], split: str) -> Path:
    csv_path = IMAGENET100_PREPROCESSED_ROOT / f"{split}.csv"
    _ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "wnid"])
        writer.writerows(rows)
    return csv_path


def _write_imagenet100_labels(raw_root: Path, wnid_to_idx: dict[str, int]) -> Path:
    # Labels.json (wnid -> human-readable name) ships with the archive; tolerate
    # its absence by falling back to the wnid as the name.
    human_names: dict[str, str] = {}
    labels_file = next(iter(raw_root.glob("Labels.json")), None)
    if labels_file is not None:
        human_names = json.loads(labels_file.read_text(encoding="utf-8"))

    payload = {
        str(index): {"wnid": wnid, "name": human_names.get(wnid, wnid)}
        for wnid, index in wnid_to_idx.items()
    }
    _ensure_dir(IMAGENET100_LABELS_PATH.parent)
    IMAGENET100_LABELS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return IMAGENET100_LABELS_PATH


def _write_imagenet100_stats(stats: ChannelStats) -> Path:
    _ensure_dir(IMAGENET100_STATS_PATH.parent)
    payload = {
        "dataset": IMAGENET100_DATASET_NAME,
        "split": "train",
        "short_side": IMAGENET100_SHORT_SIDE,
        **stats.as_dict(),
    }
    IMAGENET100_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return IMAGENET100_STATS_PATH


def preprocess_imagenet100() -> None:
    if IMAGENET100_COMPLETE_MARKER.exists():
        logger.info(
            f"Skipping {IMAGENET100_DATASET_NAME}: completion marker exists at {IMAGENET100_COMPLETE_MARKER}"
        )
        return

    raw_root = _resolve_imagenet100_raw_root()
    logger.info(f"Found {IMAGENET100_DATASET_NAME} archive at {raw_root}")

    wnid_to_idx = _build_wnid_index(raw_root)
    logger.info(f"Discovered {len(wnid_to_idx)} classes")

    train_samples = _collect_imagenet_samples(raw_root, IMAGENET100_TRAIN_DIR_GLOB, wnid_to_idx)
    val_samples = _collect_imagenet_samples(raw_root, IMAGENET100_VAL_DIR_GLOB, wnid_to_idx)

    train_rows, train_stats = _preprocess_imagenet100_split(train_samples, split="train", collect_stats=True)
    val_rows, _ = _preprocess_imagenet100_split(val_samples, split="val", collect_stats=False)

    _write_imagenet100_manifest(train_rows, "train")
    _write_imagenet100_manifest(val_rows, "val")
    labels_path = _write_imagenet100_labels(raw_root, wnid_to_idx)

    if train_stats is None:
        raise ValueError("Train stats were not collected")
    stats_path = _write_imagenet100_stats(train_stats)

    IMAGENET100_COMPLETE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    IMAGENET100_COMPLETE_MARKER.write_text("complete\n", encoding="utf-8")

    logger.info(
        f"Preprocessed {len(train_rows)} train + {len(val_rows)} val images into "
        f"{IMAGENET100_PREPROCESSED_ROOT}"
    )
    logger.info(f"Wrote manifests, {labels_path.name} and {stats_path.name}")
    logger.info(f"Preprocessing of {IMAGENET100_DATASET_NAME} is complete")


def preprocess_datasets() -> None:
    preprocess_nyu_depth_v2()
    preprocess_imagenet100()


if __name__ == "__main__":
    preprocess_datasets()
