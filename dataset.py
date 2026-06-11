import json
import torch
import pandas as pd
from globals import FLOATING_PRECISION, INPUT_RESOLUTION, TRAIN_DEPTH_TO_CM, OUTPUT_RESOLUTION
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from typing import Literal


class NormalizedNyuDataset(Dataset):
    root = Path("preprocessed_datasets/nyu-depth-v2")
    train_manifest_rel = Path("data/nyu2_train.csv")
    test_manifest_rel = Path("data/nyu2_test.csv")
    stats_rel = Path("stats.json")

    def __init__(
        self,
        split: Literal["train", "test"],
    ) -> None:
        super().__init__()
        self.samples = self._load_split(split)
        self.zscore_normalize = v2.Normalize(*self._load_normalization_stats())

    def _load_split(self, split: Literal["train", "test"]) -> pd.DataFrame:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        train_manifest_path = self.root / self.train_manifest_rel
        test_manifest_path = self.root / self.test_manifest_rel

        if not train_manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest does not exist: {train_manifest_path}"
            )
        if not test_manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest does not exist: {test_manifest_path}"
            )

        if split == "test":
            return pd.read_csv(test_manifest_path, names=["image_path", "depth_path"], header=None)

        return pd.read_csv(train_manifest_path, names=["image_path", "depth_path"], header=None)

    def _load_normalization_stats(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        stats_path = self.root / self.stats_rel
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset stats file does not exist: {stats_path}")

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        mean = tuple(float(value) for value in stats["mean_rgb"])
        std = tuple(float(value) for value in stats["std_rgb"])
        return mean, std

    def _resolve_sample_path(self, relative_path: str) -> Path:
        return self.root / Path(relative_path)

    @staticmethod
    def _load_image_tensor(image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float()

    @staticmethod
    def _load_image_tensor_uint8(image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            return TF.pil_to_tensor(image.convert("RGB"))

    def _normalize_image(self, image: Tensor) -> Tensor:
        return self.zscore_normalize(image / 255.0)

    @staticmethod
    def _load_depth_tensor(depth_path: Path) -> Tensor:
        with Image.open(depth_path) as depth:
            depth_tensor = TF.pil_to_tensor(depth).float()
            return depth_tensor

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.samples.iloc[index]
        image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))
        depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"]))
        image = self._normalize_image(image)

        return {
            "image": image,
            "depth": depth,
        }


class AugmentedNyuDataset(NormalizedNyuDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        views=1,
    ) -> None:
        super().__init__(split)

        self.views = views

        # operates on uint8 [0, 255]; ToDtype(scale=True) only rescales on an
        # actual dtype conversion, so the input must stay uint8 until here
        self.spatial = v2.Compose([
            v2.ToImage(),
            # `size` is only the output resolution; `scale` is the fraction of
            # the image area kept by the crop (default would be 0.08-1.0)
            v2.RandomResizedCrop(
                size=INPUT_RESOLUTION,
                scale=(0.4, 1.0),
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(FLOATING_PRECISION, scale=True),
        ])

        self.appearance = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2),
            self.zscore_normalize,
        ])

        self.test = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(INPUT_RESOLUTION),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                self.zscore_normalize,
            ]
        )

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor_uint8(self._resolve_sample_path(sample["image_path"]))
        depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"]))

        if self.views > 1:
            # each view gets its own crop: spatial invariance is the main
            # signal LeJEPA learns from, photometric jitter alone is too weak
            views = torch.stack([
                self.appearance(self.spatial(image)) for _ in range(self.views)
            ])
        else:
            views = self.test(image)

        return views, depth



class DepthTrainDataset(NormalizedNyuDataset):
    """Correctly-scaled image + depth target at decoder resolution, in cm."""

    def __init__(self, split, augment: bool):
        super().__init__(split)
        self.augment = augment

    def __getitem__(self, index):
        item = super().__getitem__(index)
        image = item["image"]
        depth = item["depth"] * TRAIN_DEPTH_TO_CM

        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[-1])
            depth = torch.flip(depth, dims=[-1])

        depth = TF.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)
        return {"image": image, "depth": depth}
