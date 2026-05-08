import csv
from pathlib import Path

from globals import INPUT_RESOLUTION, NYU_IMAGE_RESOLUTION

from PIL import Image

RESAMPLING = Image.Resampling

NYU_DATASET_NAME = "nyu-depth-v2"
NYU_DATASET_ROOT = Path("datasets/nyu_data")
NYU_TRAIN_CSV = NYU_DATASET_ROOT / "data/nyu2_train.csv"
NYU_TEST_CSV = NYU_DATASET_ROOT / "data/nyu2_test.csv"
NYU_PREPROCESSED_ROOT = Path("preprocessed_datasets/nyu-depth-v2")
NYU_COMPLETE_MARKER = Path("preprocessed_datasets/.complete") / NYU_DATASET_NAME


def expected_pil_size(resolution: tuple[int, int]) -> tuple[int, int]:
    height, width = resolution
    return width, height


def resize_and_save_image(
    source_path: Path,
    destination_path: Path,
    *,
    from_res: tuple[int, int],
    to_res: tuple[int, int],
    resample: int,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        if image.size != expected_pil_size(from_res):
            actual_resolution = (image.height, image.width)
            raise ValueError(
                f"Unexpected input size for {source_path}: "
                f"expected {from_res}, found {actual_resolution}"
            )

        resized = image.resize(expected_pil_size(to_res), resample=resample)
        resized.save(destination_path)


def preprocess_nyu_pair(image_rel_path: str, depth_rel_path: str) -> None:
    source_image_path = NYU_DATASET_ROOT / image_rel_path
    source_depth_path = NYU_DATASET_ROOT / depth_rel_path

    if not source_image_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {source_image_path}")
    if not source_depth_path.exists():
        raise FileNotFoundError(f"Missing depth map: {source_depth_path}")

    resize_and_save_image(
        source_image_path,
        NYU_PREPROCESSED_ROOT / image_rel_path,
        from_res=NYU_IMAGE_RESOLUTION,
        to_res=INPUT_RESOLUTION,
        resample=RESAMPLING.LANCZOS,
    )
    resize_and_save_image(
        source_depth_path,
        NYU_PREPROCESSED_ROOT / depth_rel_path,
        from_res=NYU_IMAGE_RESOLUTION,
        to_res=INPUT_RESOLUTION,
        resample=RESAMPLING.NEAREST,
    )


def iterate_pairs(csv_path: Path) -> list[tuple[str, str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        return [(row[0], row[1]) for row in reader if row]


def write_nyu_completion_marker() -> Path:
    NYU_COMPLETE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    NYU_COMPLETE_MARKER.write_text("complete\n", encoding="utf-8")
    return NYU_COMPLETE_MARKER


def copy_nyu_manifest(csv_path: Path) -> None:
    relative_csv_path = csv_path.relative_to(NYU_DATASET_ROOT)
    destination_path = NYU_PREPROCESSED_ROOT / relative_csv_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")


def preprocess_nyu_depth_v2() -> None:
    if NYU_COMPLETE_MARKER.exists():
        print(
            f"Skipping {NYU_DATASET_NAME}: completion marker exists at {NYU_COMPLETE_MARKER}"
        )
        print(f"Preprocessing of {NYU_DATASET_NAME} is complete")
        return

    csv_paths = [NYU_TRAIN_CSV, NYU_TEST_CSV]

    total_pairs = 0
    for csv_path in csv_paths:
        copy_nyu_manifest(csv_path)
        pairs = iterate_pairs(csv_path)
        total_pairs += len(pairs)
        for index, (image_rel_path, depth_rel_path) in enumerate(pairs, start=1):
            preprocess_nyu_pair(image_rel_path, depth_rel_path)
            if index % 500 == 0 or index == len(pairs):
                print(f"{csv_path.name}: {index}/{len(pairs)}")

    marker = write_nyu_completion_marker()
    print(f"Preprocessed {total_pairs} pairs into {NYU_PREPROCESSED_ROOT}")
    print(f"Wrote completion marker: {marker}")
    print(f"Preprocessing of {NYU_DATASET_NAME} is complete")


def preprocess_datasets() -> None:
    preprocess_nyu_depth_v2()


if __name__ == "__main__":
    preprocess_datasets()
