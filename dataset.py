import json
import torch
import pandas as pd
from globals import FLOATING_PRECISION, INPUT_RESOLUTION
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from typing import Literal
from tqdm import tqdm


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

    def _normalize_image(self, image: Tensor) -> Tensor:
        return self.zscore_normalize(image / 255.0)

    @staticmethod
    def _load_depth_tensor(depth_path: Path) -> Tensor:
        with Image.open(depth_path) as depth:
            depth_tensor = TF.pil_to_tensor(depth).float()
            return depth_tensor

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples.iloc[idx]
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
        caching: bool = True,
    ) -> None:
        super().__init__(split)

        self.views = views
        # preload everything into RAM
        self.caching = caching
        if caching:
            self.cache = [
                (
                    self._load_image_tensor(self._resolve_sample_path(row["image_path"])),
                    self._load_depth_tensor(self._resolve_sample_path(row["depth_path"])),
                )
                for _, row in tqdm(self.samples.iterrows(), total=len(self.samples), desc="caching dataset")
            ]

        self.spatial = v2.Compose([
            v2.RandomResizedCrop(
                size=(INPUT_RESOLUTION[0] // 2, INPUT_RESOLUTION[1] // 2),
            ),
            v2.Resize(INPUT_RESOLUTION),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
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
                v2.Resize(INPUT_RESOLUTION),
                v2.ToImage(),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                self.zscore_normalize,
            ]
        )

    def __getitem__(self, idx):
        if self.caching:
            image, depth = self.cache[idx]
        else:
            sample = self.samples.iloc[idx]
            image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))
            depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"]))

        if self.views > 1:
            base = self.spatial(image)  # resize/crop once
            views = torch.stack([self.appearance(base) for _ in range(self.views)])
        else:
            views = self.test(image)

        return views, depth