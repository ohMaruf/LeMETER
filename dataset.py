import json
import random
import torch
import torch.nn.functional as F
import pandas as pd
from globals import (
    FLOATING_PRECISION,
    INPUT_RESOLUTION,
    OUTPUT_RESOLUTION,
    SEED,
)
import globals
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from typing import Literal

from augmentation import AugmentationPolicy, ViewAugmentation, PairedDepthAugmentation

DepthDataset = Literal["nyu", "kitti"]
Split = Literal["train", "val", "test"]

VAL_SCENE_FRACTION = 0.05


def _partition_train_scenes(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenes = samples["image_path"].map(lambda path: str(Path(path).parent))
    unique_scenes = sorted(scenes.unique())
    n_val = max(1, round(len(unique_scenes) * VAL_SCENE_FRACTION))
    val_scenes = set(random.Random(SEED).sample(unique_scenes, n_val))

    is_val = scenes.isin(val_scenes)
    train_df = samples[~is_val].reset_index(drop=True)
    val_df = samples[is_val].reset_index(drop=True)
    return train_df, val_df


class NormalizedNyuDataset(Dataset):
    root = Path("preprocessed_datasets/nyu-depth-v2")
    train_manifest_rel = Path("data/nyu2_train.csv")
    test_manifest_rel = Path("data/nyu2_test.csv")
    stats_rel = Path("stats.json")

    def __init__(
        self,
        split: Split,
        holdout_val: bool = True,
        normalization: Literal["imagenet", "dataset"] = "imagenet",
    ) -> None:
        super().__init__()
        self.split = split
        self.samples = self._load_split(split, holdout_val)
        if normalization == "imagenet":
            self.zscore_normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.zscore_normalize = v2.Normalize(*self._load_normalization_stats())

    def _load_split(self, split: Split, holdout_val: bool = True) -> pd.DataFrame:
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

        train_manifest = pd.read_csv(train_manifest_path, names=["image_path", "depth_path"], header=None)
        train_scenes, val_scenes = _partition_train_scenes(train_manifest)
        if split == "val":
            return val_scenes
        return train_scenes if holdout_val else train_manifest

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
        split: Split,
        views=1,
        augmentation: AugmentationPolicy = "lejepa",
        normalization: Literal["imagenet", "dataset"] = "imagenet",
        with_depth: bool = False,
    ) -> None:
        super().__init__(split, holdout_val=False, normalization=normalization)

        self.views = views
        self.augmentation = augmentation
        # when True, each view returns its own depth map with the *same* spatial
        # transform applied, so a dense per-pixel probe trains on aligned pixels.
        # this re-introduces per-worker depth PNG decode (the cost the scalar
        # mean-depth cache avoids), so it is opt-in.
        self.with_depth = with_depth

        if with_depth:
            # PairedDepthAugmentation applies spatial ops to both image and depth
            # and photometric ops to the image only; the resize gives every view
            # a common base size before the (optional) crop
            self.resize = v2.Resize(INPUT_RESOLUTION, antialias=True)
            if augmentation == 'lejepa_multi_view':
                self.paired_global_aug = PairedDepthAugmentation(
                    augmentation,
                    view_type='global'
                )
                self.paired_local_aug = PairedDepthAugmentation(
                    augmentation,
                    view_type='local'
                )
            else:
                self.paired_aug = PairedDepthAugmentation(augmentation)
        else:
            if augmentation == 'lejepa_multi_view':
                self.global_view_aug = ViewAugmentation(
                    'lejepa_multi_view',
                    self.zscore_normalize,
                    view_type='global'
                )
                self.local_view_aug = ViewAugmentation(
                    'lejepa_multi_view',
                    self.zscore_normalize,
                    view_type='local'
                )
            else:
                self.view_aug = ViewAugmentation(augmentation, self.zscore_normalize)

        self.test = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(INPUT_RESOLUTION),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
                self.zscore_normalize,
            ]
        )

    def _augmented_pair(self, image_255: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        view, depth = self.paired_aug(image_255.clone(), depth_cm.clone())
        return self._normalize_image(view), F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)

    def _augmented_pair_with_aug(
        self,
        image_255: Tensor,
        depth_cm: Tensor,
        paired_aug: PairedDepthAugmentation
    ):
        view, depth = paired_aug(image_255.clone(), depth_cm.clone())
        return self._normalize_image(view), F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:
        sample = self.samples.iloc[index]

        if self.with_depth:
            image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))  # float [0, 255]
            depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"]))  # cm, full res
            image = self.resize(image)  # common base size so all views stack

            if self.views > 1:
                if self.augmentation == 'lejepa_multi_view':
                    Vg = globals.GLOBAL_VIEWS
                    Vl = globals.LOCAL_VIEWS

                    global_pairs = [
                        self._augmented_pair_with_aug(
                            image,
                            depth,
                            self.paired_global_aug
                        )
                        for _ in range(Vg)
                    ]
                    local_pairs = [
                        self._augmented_pair_with_aug(
                            image,
                            depth,
                            self.paired_local_aug
                        )
                        for _ in range(Vl)
                    ]

                    views, depths = zip(*(global_pairs + local_pairs))
                    return torch.stack(views), torch.stack(depths)
                else:
                    pairs = [self._augmented_pair(image, depth) for _ in range(self.views)]
                    views, depths = zip(*pairs)
                    return torch.stack(views), torch.stack(depths)

            return self._normalize_image(image), F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)

        image = self._load_image_tensor_uint8(self._resolve_sample_path(sample["image_path"]))
        if self.views > 1:
            if self.augmentation == 'lejepa_multi_view':
                Vg = globals.GLOBAL_VIEWS
                Vl = globals.LOCAL_VIEWS
                views = [self.global_view_aug(image) for _ in range(Vg)] + \
                    [self.local_view_aug(image) for _ in range(Vl)]
            else:
                views = [self.view_aug(image) for _ in range(self.views)]

            return torch.stack(views)

        return self.test(image)



class DepthTrainDataset(NormalizedNyuDataset):
    def __init__(self, split, augment: bool, augmentation: AugmentationPolicy = "meter", normalization: Literal["imagenet", "dataset"] = "imagenet"):
        super().__init__(split, normalization=normalization, holdout_val=True)
        self.augment = augment
        self.aug = PairedDepthAugmentation(augmentation)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image_tensor(self._resolve_sample_path(sample["image_path"]))
        depth = self._load_depth_tensor(self._resolve_sample_path(sample["depth_path"]))

        if self.augment:
            image, depth = self.aug(image, depth)

        image = self._normalize_image(image)
        # adaptive pooling handles the arbitrary post-crop depth resolution
        depth = F.adaptive_avg_pool2d(depth, OUTPUT_RESOLUTION)
        return {"image": image, "depth": depth}
