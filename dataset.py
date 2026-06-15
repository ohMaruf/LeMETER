import json
import torch
import torch.nn.functional as F
import pandas as pd
from globals import (
    FLOATING_PRECISION,
    INPUT_RESOLUTION,
    TRAIN_DEPTH_TO_CM,
    OUTPUT_RESOLUTION,
    METER_AUGMENTATION,
    LEJEPA_AUGMENTATION,
)
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Literal

import logger

DepthDataset = Literal["nyu", "kitti"]


def meter_photometric_jitter(image: Tensor) -> Tensor:
    """Image part of METER's 'shifting strategy': gamma, brightness and
    per-channel color jitter (±10%) applied on the 0-255 scale, as in the
    original recipe (augmentation.py / METER paper Sec. III-C)."""
    gamma = float(torch.empty(()).uniform_(0.9, 1.1))
    brightness = float(torch.empty(()).uniform_(0.9, 1.1))
    colors = torch.empty(3, 1, 1).uniform_(0.9, 1.1)
    return (image.pow(gamma) * brightness * colors).clamp(0.0, 255.0)


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
        augmentation: Literal["lejepa", "meter"] = "lejepa",
    ) -> None:
        super().__init__(split)

        self.views = views
        self.augmentation = augmentation
        # scalar target for the online depth probe: flip-invariant and cached,
        # so depth PNG decoding stays out of the training workers
        self.mean_depth_cm = self._load_mean_depth_cache(split)

        if augmentation == "lejepa":
            # operates on uint8 [0, 255]; ToDtype(scale=True) only rescales on
            # an actual dtype conversion, so the input must stay uint8 until here
            self.spatial = v2.Compose([
                v2.ToImage(),
                # `size` is only the output resolution; `scale` is the fraction
                # of the image area kept by the crop (default would be 0.08-1.0)
                v2.RandomResizedCrop(
                    size=INPUT_RESOLUTION,
                    scale=LEJEPA_AUGMENTATION["random_crop_scale"],
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=LEJEPA_AUGMENTATION["mirror"]),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
            ])

            self.appearance = v2.Compose([
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=LEJEPA_AUGMENTATION["color_jitter"]),
                v2.RandomGrayscale(p=LEJEPA_AUGMENTATION["grayscale"]),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=LEJEPA_AUGMENTATION["gaussian_blur"]),
                v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=LEJEPA_AUGMENTATION["solarize"]),
                self.zscore_normalize,
            ])
        else:
            # METER's conservative supervised policy as SSL views (depth shift
            # excluded: pretraining uses no labels); stays uint8 so the
            # photometric jitter can run on the 0-255 scale like the original
            self.spatial = v2.Compose([
                v2.ToImage(),
                v2.RandomApply([
                    v2.RandomResizedCrop(
                        size=INPUT_RESOLUTION,
                        scale=METER_AUGMENTATION["random_crop_scale"],
                        ratio=METER_AUGMENTATION["random_crop_ratio"],
                        antialias=True,
                    ),
                ], p=METER_AUGMENTATION["random_crop"]),
                v2.RandomHorizontalFlip(p=METER_AUGMENTATION["mirror"]),
            ])

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

    def _make_view(self, image: Tensor) -> Tensor:
        if self.augmentation == "lejepa":
            return self.appearance(self.spatial(image))

        view = self.spatial(image).to(FLOATING_PRECISION)
        if torch.rand(()) < METER_AUGMENTATION["shifting_strategy"]:
            view = meter_photometric_jitter(view)
        return self.zscore_normalize(view / 255.0)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor_uint8(self._resolve_sample_path(sample["image_path"]))

        # the full depth map stays out of the workers (PNG decode + queue RAM);
        # the online probe only needs the cached per-image mean depth
        mean_depth_cm = self.mean_depth_cm[index]

        if self.views > 1:
            # each view gets its own crop: spatial invariance is the main
            # signal LeJEPA learns from, photometric jitter alone is too weak
            return torch.stack([self._make_view(image) for _ in range(self.views)]), mean_depth_cm

        return self.test(image), mean_depth_cm



class DepthTrainDataset(NormalizedNyuDataset):
    """Image + depth target at decoder resolution, in cm, with the original
    METER training policy (METER_AUGMENTATION): mirror, joint random crop,
    gamma / brightness / color jitter on the 0-255 scale, depth shift.
    """

    def __init__(self, split, augment: bool):
        super().__init__(split)
        self.augment = augment

    def _augment(self, image: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        """image in [0, 255] float CHW at INPUT_RESOLUTION, depth in cm at
        its stored (full) resolution."""
        aug = METER_AUGMENTATION

        if torch.rand(()) < aug["mirror"]:
            image = torch.flip(image, dims=[-1])
            depth_cm = torch.flip(depth_cm, dims=[-1])

        if torch.rand(()) < aug["random_crop"]:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image,
                scale=list(aug["random_crop_scale"]),
                ratio=list(aug["random_crop_ratio"]),
            )
            # the same relative window on the full-resolution depth map
            sh = depth_cm.shape[-2] / image.shape[-2]
            sw = depth_cm.shape[-1] / image.shape[-1]
            depth_cm = TF.crop(
                depth_cm, round(top * sh), round(left * sw), round(height * sh), round(width * sw)
            )
            image = TF.resized_crop(
                image, top, left, height, width, list(INPUT_RESOLUTION), antialias=True
            )

        if torch.rand(()) < aug["shifting_strategy"]:
            image = meter_photometric_jitter(image)
            shift_cm = float(torch.randint(-10, 11, ()))
            depth_cm = (depth_cm + shift_cm).clamp_min(0.0)

        return image, depth_cm

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))
        depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"])) * TRAIN_DEPTH_TO_CM

        if self.augment:
            image, depth = self._augment(image, depth)

        image = self._normalize_image(image)
        # adaptive pooling handles the arbitrary post-crop depth resolution
        depth = F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)
        return {"image": image, "depth": depth}
