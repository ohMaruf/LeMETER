import json
import torch
import torch.nn.functional as F
import pandas as pd
from globals import (
    FLOATING_PRECISION,
    INPUT_RESOLUTION,
    TRAIN_DEPTH_TO_CM,
    OUTPUT_RESOLUTION,
)
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Literal

import logger
from augmentation import AugmentationPolicy, ViewAugmentation, PairedDepthAugmentation

DepthDataset = Literal["nyu", "kitti"]


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
        augmentation: AugmentationPolicy = "lejepa",
    ) -> None:
        super().__init__(split)

        self.views = views
        self.augmentation = augmentation
        # scalar target for the online depth probe: flip-invariant and cached,
        # so depth PNG decoding stays out of the training workers
        self.mean_depth_cm = self._load_mean_depth_cache(split)

        # all augmentation lives in augmentation.py; normalize is passed in so
        # the METER channel swap / shifting strategy run on un-normalized pixels
        self.view_aug = ViewAugmentation(augmentation, self.zscore_normalize)

        self.test = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(INPUT_RESOLUTION),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                self.zscore_normalize,
            ]
        )

    def _load_mean_depth_cache(self, split: Literal["train", "test"]) -> Tensor:
        """Per-image mean depth in cm, aligned with the manifest order.
        Built once, then loaded from disk."""
        cache_path = self.root / f"mean_depth_cm_{split}.pt"
        if cache_path.exists():
            return torch.load(cache_path)

        logger.info(f"building mean-depth cache for split '{split}' (one-time)")
        # train depth PNGs are uint8 (255 == 10 m), test PNGs are uint16 mm
        scale = TRAIN_DEPTH_TO_CM if split == "train" else 0.1
        means = torch.empty(len(self.samples))
        for index in tqdm(range(len(self.samples)), desc="mean depth"):
            depth_path = self._resolve_sample_path(self.samples.iloc[index]["depth_path"])
            means[index] = self._load_depth_tensor(depth_path).mean() * scale

        torch.save(means, cache_path)
        logger.info(f"wrote {cache_path}")
        return means

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor_uint8(self._resolve_sample_path(sample["image_path"]))

        # the full depth map stays out of the workers (PNG decode + queue RAM);
        # the online probe only needs the cached per-image mean depth
        mean_depth_cm = self.mean_depth_cm[index]

        if self.views > 1:
            # each view gets its own crop: spatial invariance is the main
            # signal LeJEPA learns from, photometric jitter alone is too weak
            return torch.stack([self.view_aug(image) for _ in range(self.views)]), mean_depth_cm

        return self.test(image), mean_depth_cm



class DepthTrainDataset(NormalizedNyuDataset):
    """Image + depth target at decoder resolution, in cm. The paired
    augmentation (see augmentation.PairedDepthAugmentation) applies the chosen
    policy jointly to image and depth; the default "meter" policy is horizontal
    mirror, RGB channel swap, joint random crop, and the shifting strategy
    (gamma / brightness / color jitter + depth shift), each at p=0.5.
    """

    def __init__(self, split, augment: bool, augmentation: AugmentationPolicy = "meter"):
        super().__init__(split)
        self.augment = augment
        self.aug = PairedDepthAugmentation(augmentation)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))
        depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"])) * TRAIN_DEPTH_TO_CM

        if self.augment:
            image, depth = self.aug(image, depth)

        image = self._normalize_image(image)
        # adaptive pooling handles the arbitrary post-crop depth resolution
        depth = F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)
        return {"image": image, "depth": depth}
