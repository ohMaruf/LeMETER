import torch
import csv
import os
from globals import FLOATING_PRECISION
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor

NYU_DEPTH_V2 = "nyu-depth-v2"

DATASET_SPECS = {
    NYU_DEPTH_V2: {
        "root": Path("preprocessed_datasets/nyu-depth-v2"),
        "train_manifest": Path("data/nyu2_train.csv"),
        "test_manifest": Path("data/nyu2_test.csv"),
        "depth_divisor": 10.0,
        "depth_unit_to_meters": 0.01,
    },
}


class PreprocessedDepthDataset(Dataset):
    """Loads paired RGB/depth samples from a preprocessed manifest."""

    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path,
        depth_divisor: float = 1.0,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        self.depth_divisor = depth_divisor
        self.samples = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Path]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")  # noqa: E501
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest does not exist: {self.manifest_path}"
            )  # noqa: E501

        samples: list[dict[str, Path]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row_index, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"Malformed manifest row {row_index} in {self.manifest_path}: {row}"  # noqa: E501
                    )

                image_rel_path = Path(row[0])
                depth_rel_path = Path(row[1])
                image_path = self.root_dir / image_rel_path
                depth_path = self.root_dir / depth_rel_path

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Missing RGB image referenced by manifest: {image_path}"
                    )  # noqa: E501
                if not depth_path.exists():
                    raise FileNotFoundError(
                        f"Missing depth map referenced by manifest: {depth_path}"
                    )  # noqa: E501

                samples.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                    }
                )

        if not samples:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        return samples

    @staticmethod
    def _load_image_tensor(image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float() / 255.0

    def _load_depth_tensor(self, depth_path: Path) -> Tensor:
        with Image.open(depth_path) as depth:
            return TF.pil_to_tensor(depth).float() / self.depth_divisor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        image = self._load_image_tensor(sample["image_path"])
        depth = self._load_depth_tensor(sample["depth_path"])

        return {
            "image": image,
            "depth": depth,
        }


class AugmentedDepthDataset(PreprocessedDepthDataset):
    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path,
        depth_divisor: float = 1.0,
        views=1,
    ) -> None:
        super().__init__(root_dir, manifest_path, depth_divisor)

        self.views = views
        self.augmentation = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),  # noqa: E501
                v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # noqa: E501
            ]
        )

        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # noqa: E501
            ]
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = item["image"]
        transform = self.augmentation if self.views > 1 else self.test
        return torch.stack([transform(image) for _ in range(self.views)]), item["depth"]  # noqa: E501


def build_dataset(split: str, views: int = 1) -> PreprocessedDepthDataset:
    spec = DATASET_SPECS[NYU_DEPTH_V2]
    if split == "train":
        manifest_key = "train_manifest"
    elif split == "test":
        manifest_key = "test_manifest"
    else:
        raise ValueError(f"Unsupported split: {split}")

    root_dir = Path(spec["root"])
    manifest_path = root_dir / Path(spec[manifest_key])
    return PreprocessedDepthDataset(
        root_dir=root_dir,
        manifest_path=manifest_path,
        depth_divisor=float(spec.get("depth_divisor", 1.0)),
    )


def build_augmented_dataset(split: str, views: int = 1) -> AugmentedDepthDataset:
    spec = DATASET_SPECS[NYU_DEPTH_V2]
    if split == "train":
        manifest_key = "train_manifest"
    elif split == "test":
        manifest_key = "test_manifest"
    else:
        raise ValueError(f"Unsupported split: {split}")

    root_dir = Path(spec["root"])
    manifest_path = root_dir / Path(spec[manifest_key])
    return AugmentedDepthDataset(
        root_dir=root_dir,
        manifest_path=manifest_path,
        depth_divisor=float(spec.get("depth_divisor", 1.0)),
        views=views,
    )
