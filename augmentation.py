"""Augmentation policies for LeMETER.

All augmentation lives here (kept out of dataset.py) and every tunable is a
named constant in the POLICIES block below — no magic numbers buried in the
pipeline, and normalization is passed in (dataset-specific) rather than the
hard-coded ImageNet stats of the original draft.

Three view-generation policies, selected by name:
  - "lejepa":  SSL/DINO-style — aggressive crops + strong photometric jitter.
               Invariance to these is the only training signal.
  - "meter":   the conservative supervised METER policy used as SSL views.
  - "lemeter": stochastic mix — each view independently and randomly picks
               between "meter" augmentation OR "lejepa" augmentation,
               but not both together.

Design principles (inherited from the draft):
  - APPEARANCE is aggressive: depth is invariant to color / brightness / blur.
  - SPATIAL is moderate: depth IS sensitive to location, so no wild crops.
  - The METER policy mirrors AND vertical-flips (as in the paper); the lejepa
    policy keeps the original DINO crops + horizontal flip only.
  - depth-scale shift requires labels, so it only runs in the paired path.
"""

from typing import Literal

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

from globals import INPUT_RESOLUTION, FLOATING_PRECISION

AugmentationPolicy = Literal["lejepa", "meter", "lemeter"]

# ---------------------------------------------------------------------------
# Augmentation parameters
# ---------------------------------------------------------------------------
# Every random transform fires with the listed probability. Crop area is the
# fraction of the image kept (torchvision's RandomResizedCrop `scale`), ratio is
# the aspect-ratio range. The METER "shifting strategy" applies gamma,
# brightness and per-channel color multipliers; in the paired path it also
# shifts the depth label by depth_shift_cm centimeters.
LEJEPA = {
    "mirror": 0.5,
    "vertical_flip": 0.5,
    "random_crop": 1.0,
    "random_crop_scale": (0.4, 0.6),
    "random_crop_ratio": (0.75, 4 / 3),
    "color_jitter": 0.8,
    "color_jitter_strength": (0.8, 0.8, 0.8, 0.2),  # brightness, contrast, saturation, hue
    "grayscale": 0.2,
    "gaussian_blur": 0.5,
    "gaussian_blur_kernel": 7,
    "gaussian_blur_sigma": (0.1, 2.0),
    "solarize": 0.2,
    "solarize_threshold": 0.5,
}

METER = {
    "mirror": 0.5,
    "vertical_flip": 0.5,
    "c_swap": 0.5,
    "random_crop": 0.5,
    "random_crop_scale": (0.6, 1.0),
    "random_crop_ratio": (0.75, 4 / 3),
    "shifting_strategy": 0.5,
    "gamma_range": (0.9, 1.1),
    "brightness_range": (0.9, 1.1),
    "color_range": (0.9, 1.1),
    "depth_shift_cm": (-10, 10),
}

# Stochastic mix: each view independently picks the full "meter" or the full
# "lejepa" policy (never both). `meter_prob` is the probability of the meter pick.
LEMETER = {
    "meter_prob": 0.5,
}

POLICIES: dict[AugmentationPolicy, dict] = {
    "lejepa": LEJEPA,
    "meter": METER,
    "lemeter": LEMETER,
}


# ---------------------------------------------------------------------------
# Low-level METER ops
# ---------------------------------------------------------------------------
def meter_channel_swap(image: Tensor) -> Tensor:
    """METER 'channel swap': replace the RGB channels with a random length-3
    selection (with replacement) from {R, G, B}. augmentation.py's original
    picks one triple uniformly from product([0, 1, 2], repeat=3), which is
    exactly a uniform draw over {0, 1, 2}^3."""
    indices = torch.randint(0, 3, (3,))
    return image[indices]


def meter_photometric_jitter(
    image: Tensor,
    gamma_range: tuple[float, float],
    brightness_range: tuple[float, float],
    color_range: tuple[float, float],
    max_value: float,
) -> Tensor:
    """Image part of METER's 'shifting strategy': gamma, brightness and
    per-channel color multipliers. The original ran on the 0-255 scale
    (max_value=255); the lemeter view path runs it on [0, 1] (max_value=1)."""
    gamma = float(torch.empty(()).uniform_(*gamma_range))
    brightness = float(torch.empty(()).uniform_(*brightness_range))
    colors = torch.empty(3, 1, 1).uniform_(*color_range)
    base = image.clamp(0.0, max_value)
    return (base.pow(gamma) * brightness * colors).clamp(0.0, max_value)


# ---------------------------------------------------------------------------
# View augmentation (SSL, image only)
# ---------------------------------------------------------------------------
class ViewAugmentation:
    """Produce one augmented, normalized view from a uint8 CHW image tensor.

    `normalize` is the dataset's z-score transform (v2.Normalize), applied last
    so the channel swap / shifting strategy operate on un-normalized pixels.
    """

    def __init__(self, policy: AugmentationPolicy, normalize) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown augmentation policy {policy!r}")
        self.policy = policy
        self.config = POLICIES[policy]
        self.normalize = normalize
        cfg = self.config

        if policy == "lemeter":
            # each view randomly gets the full meter OR full lejepa policy; build
            # both and dispatch on a coin flip in __call__ (never both at once).
            self._meter = ViewAugmentation("meter", normalize)
            self._lejepa = ViewAugmentation("lejepa", normalize)
            self.spatial = None
            self.appearance = None
        elif policy == "meter":
            # conservative supervised policy as SSL views; stays uint8 so the
            # photometric jitter runs on the 0-255 scale like the original
            self.spatial = v2.Compose([
                v2.ToImage(),
                v2.RandomApply([
                    v2.RandomResizedCrop(
                        size=INPUT_RESOLUTION,
                        scale=cfg["random_crop_scale"],
                        ratio=cfg["random_crop_ratio"],
                        antialias=True,
                    ),
                ], p=cfg["random_crop"]),
                v2.RandomHorizontalFlip(p=cfg["mirror"]),
                v2.RandomVerticalFlip(p=cfg["vertical_flip"]),
            ])
            self.appearance = None
        else:
            # lejepa: crop to float [0, 1], then DINO photometric jitter.
            # `scale` is the kept-area fraction (default would be 0.08-1.0).
            self.spatial = v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(
                    size=INPUT_RESOLUTION,
                    scale=cfg["random_crop_scale"],
                    ratio=cfg["random_crop_ratio"],
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=cfg["mirror"]),
                v2.RandomVerticalFlip(p=cfg["vertical_flip"]),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
            ])
            self.appearance = v2.Compose([
                v2.RandomApply([v2.ColorJitter(*cfg["color_jitter_strength"])], p=cfg["color_jitter"]),
                v2.RandomGrayscale(p=cfg["grayscale"]),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=cfg["gaussian_blur_kernel"], sigma=cfg["gaussian_blur_sigma"])],
                    p=cfg["gaussian_blur"],
                ),
                v2.RandomApply([v2.RandomSolarize(threshold=cfg["solarize_threshold"])], p=cfg["solarize"]),
            ])

    def __call__(self, image: Tensor) -> Tensor:
        cfg = self.config

        if self.policy == "lemeter":
            # coin flip per view: full meter policy OR full lejepa policy, never both
            chosen = self._meter if torch.rand(()) < cfg["meter_prob"] else self._lejepa
            return chosen(image)

        if self.policy == "meter":
            view = self.spatial(image).to(FLOATING_PRECISION)  # 0-255
            if torch.rand(()) < cfg["c_swap"]:
                view = meter_channel_swap(view)
            if torch.rand(()) < cfg["shifting_strategy"]:
                view = meter_photometric_jitter(
                    view, cfg["gamma_range"], cfg["brightness_range"], cfg["color_range"], max_value=255.0
                )
            return self.normalize(view / 255.0)

        assert self.appearance is not None  # always built for lejepa
        view = self.appearance(self.spatial(image))  # float [0, 1]
        return self.normalize(view)


# ---------------------------------------------------------------------------
# Paired image + depth augmentation (supervised decoder training)
# ---------------------------------------------------------------------------
class PairedDepthAugmentation:
    """Jointly augment an image (float CHW in [0, 255] at INPUT_RESOLUTION) and
    its depth label (cm, full resolution). Spatial ops apply to both; the
    photometric ops and channel swap touch the image only; the depth-scale
    shift touches the label only. The caller normalizes the image afterwards.

    Three policies, mirroring the SSL view path:
      - "meter":   METER reference order — mirror, channel swap, joint crop,
                   shifting strategy (with depth-scale shift). Its leading
                   'random flipping' op was a no-op slice, so it is omitted
                   (no vertical flip).
      - "lejepa":  joint mirror + always-on crop, then DINO appearance
                   corruption (color jitter, grayscale, blur, solarize) on the
                   image only. Depth is never photometrically touched.
      - "lemeter": randomly decide if to apply meter or lejepa augmentation.
    """

    def __init__(
        self,
        policy: AugmentationPolicy = "meter",
        target_size: tuple[int, int] = INPUT_RESOLUTION,
        crop_scale: tuple[float, float] | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown augmentation policy {policy!r}")
        self.policy = policy
        self.config = POLICIES[policy]
        # multi-crop knobs: the crop resizes to `target_size` (locals are
        # smaller), and `crop_scale` overrides the policy's kept-area fraction
        # (global vs local zoom). Both default to the single-resolution behavior.
        self.target_size = target_size
        self.crop_scale = crop_scale

        if policy == "lemeter":
            self._meter = PairedDepthAugmentation("meter", target_size, crop_scale)
            self._lejepa = PairedDepthAugmentation("lejepa", target_size, crop_scale)
            self.appearance = None
        elif policy == "lejepa":
            cfg = self.config
            # DINO appearance corruption on the image only; built to run on
            # [0, 1] (matching the SSL view path) so thresholds line up.
            self.appearance = v2.Compose([
                v2.RandomApply([v2.ColorJitter(*cfg["color_jitter_strength"])], p=cfg["color_jitter"]),
                v2.RandomGrayscale(p=cfg["grayscale"]),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=cfg["gaussian_blur_kernel"], sigma=cfg["gaussian_blur_sigma"])],
                    p=cfg["gaussian_blur"],
                ),
                v2.RandomApply([v2.RandomSolarize(threshold=cfg["solarize_threshold"])], p=cfg["solarize"]),
            ])
        else:
            self.appearance = None

    def _joint_crop(self, image: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the same relative crop window to image and full-res depth."""
        cfg = self.config
        scale = self.crop_scale if self.crop_scale is not None else cfg["random_crop_scale"]
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=list(scale),
            ratio=list(cfg["random_crop_ratio"]),
        )
        sh = depth_cm.shape[-2] / image.shape[-2]
        sw = depth_cm.shape[-1] / image.shape[-1]
        depth_cm = TF.crop(
            depth_cm, round(top * sh), round(left * sw), round(height * sh), round(width * sw)
        )
        image = TF.resized_crop(
            image, top, left, height, width, list(self.target_size), antialias=True
        )
        return image, depth_cm

    def __call__(self, image: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config

        if self.policy == "lemeter":
            # coin flip per sample: full meter OR full lejepa policy, never both
            chosen = self._meter if torch.rand(()) < cfg["meter_prob"] else self._lejepa
            return chosen(image, depth_cm)

        if self.policy == "lejepa":
            if torch.rand(()) < cfg["mirror"]:
                image = torch.flip(image, dims=[-1])
                depth_cm = torch.flip(depth_cm, dims=[-1])
            if torch.rand(()) < cfg["random_crop"]:
                image, depth_cm = self._joint_crop(image, depth_cm)
            # appearance runs on [0, 1] then back to the [0, 255] caller scale
            assert self.appearance is not None
            image = self.appearance(image / 255.0) * 255.0
            return image, depth_cm

        # meter
        if torch.rand(()) < cfg["mirror"]:
            image = torch.flip(image, dims=[-1])
            depth_cm = torch.flip(depth_cm, dims=[-1])

        # vertical flip on the height axis, applied jointly to image and depth
        if torch.rand(()) < cfg["vertical_flip"]:
            image = torch.flip(image, dims=[-2])
            depth_cm = torch.flip(depth_cm, dims=[-2])

        # channel swap affects colour only; the depth target is untouched
        if torch.rand(()) < cfg["c_swap"]:
            image = meter_channel_swap(image)

        if torch.rand(()) < cfg["random_crop"]:
            image, depth_cm = self._joint_crop(image, depth_cm)

        if torch.rand(()) < cfg["shifting_strategy"]:
            image = meter_photometric_jitter(
                image, cfg["gamma_range"], cfg["brightness_range"], cfg["color_range"], max_value=255.0
            )
            low, high = cfg["depth_shift_cm"]
            shift_cm = float(torch.randint(low, high + 1, ()))
            depth_cm = (depth_cm + shift_cm).clamp_min(0.0)

        return image, depth_cm
