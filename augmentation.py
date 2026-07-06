from typing import Literal

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

from globals import INPUT_RESOLUTION, FLOATING_PRECISION

AugmentationPolicy = Literal["lejepa", "lejepa_multi_view", "meter", "lemeter"]

# Augmentation Hyperparameters
LEJEPA = {
    "mirror": 0.5,
    "vertical_flip": 0.5,
    "random_crop": 1.0,
    "random_crop_scale": (0.4, 0.6),
    "random_crop_ratio": (0.75, 4 / 3),
    "color_jitter": 0.8,
    # brightness, contrast, saturation, hue
    "color_jitter_strength": (0.8, 0.8, 0.8, 0.2),
    "grayscale": 0.2,
    "gaussian_blur": 0.5,
    "gaussian_blur_kernel": 7,
    "gaussian_blur_sigma": (0.1, 2.0),
    "solarize": 0.2,
    "solarize_threshold": 0.5,
}

LEJEPA_MULTI_VIEW = {
    "global": {
        "random_crop": 1.0,
        "random_crop_scale": (0.3, 1.0),
        "random_crop_ratio": (0.75, 4 / 3),
        "mirror": 0.5,
        "color_jitter": 0.8,
        # brightness, contrast, saturation, hue
        "color_jitter_strength": (0.4, 0.4, 0.2, 0.1),
        "grayscale": 0.2,
        "gaussian_blur": 0.5,
        "gaussian_blur_kernel": 7,
        "gaussian_blur_sigma": (0.1, 2.0),
        "solarize": 0.2,
        "solarize_threshold": 0.5,
    },
    "local": {
        "random_crop": 1.0,
        "random_crop_scale": (0.05, 0.3),
        "random_crop_ratio": (0.75, 4 / 3),
        "mirror": 0.5,
        "color_jitter": 0.8,
        # brightness, contrast, saturation, hue
        "color_jitter_strength": (0.4, 0.4, 0.2, 0.1),
        "grayscale": 0.2,
        "gaussian_blur": 0.5,
        "gaussian_blur_kernel": 7,
        "gaussian_blur_sigma": (0.1, 2.0),
        "solarize": 0.2,
        "solarize_threshold": 0.5,
    }
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
    "lejepa_multi_view": LEJEPA_MULTI_VIEW,
    "meter": METER,
    "lemeter": LEMETER,
}


def meter_channel_swap(image: Tensor) -> Tensor:
    indices = torch.randint(0, 3, (3,))
    return image[indices]


def meter_photometric_jitter(
    image: Tensor,
    gamma_range: tuple[float, float],
    brightness_range: tuple[float, float],
    color_range: tuple[float, float],
    max_value: float,
) -> Tensor:
    gamma = float(torch.empty(()).uniform_(*gamma_range))
    brightness = float(torch.empty(()).uniform_(*brightness_range))
    colors = torch.empty(3, 1, 1).uniform_(*color_range)
    base = image.clamp(0.0, max_value)
    return (base.pow(gamma) * brightness * colors).clamp(0.0, max_value)



class ViewAugmentation:
    def __init__(
        self,
        policy: AugmentationPolicy,
        normalize,
        view_type: Literal['global', 'local'] | None = None
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown augmentation policy {policy!r}")
        self.policy = policy
        self.config = POLICIES[policy]
        self.normalize = normalize

        if policy == "lejepa_multi_view":
            if view_type not in ('global', 'local'):
                raise ValueError(
                    "view_type must be 'global' or 'local'"
                    "for 'lejepa_multi_view' augmentation"
                )
            cfg = self.config[view_type]
            self.view_type = view_type
        else:
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
        elif policy == "lejepa_multi_view":
            self.spatial = v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(
                    size=INPUT_RESOLUTION,
                    scale=cfg['random_crop_scale'],
                    ratio=cfg['random_crop_ratio'],
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=cfg['mirror']),
                v2.ToDtype(FLOATING_PRECISION, scale=True),
            ])

            self.appearance = v2.Compose([
                v2.RandomApply(
                    [v2.ColorJitter(*cfg['color_jitter_strength'])],
                    p=cfg['color_jitter'],
                ),
                v2.RandomGrayscale(p=cfg['grayscale']),
                v2.RandomApply(
                    [
                        v2.GaussianBlur(
                            kernel_size=cfg["gaussian_blur_kernel"],
                            sigma=cfg["gaussian_blur_sigma"]
                        )
                    ],
                    p=cfg["gaussian_blur"],
                ),
                v2.RandomApply(
                    [
                        v2.RandomSolarize(threshold=cfg['solarize_threshold']),
                    ],
                    p=cfg["solarize"],
                ),
            ])
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


# Paired image + depth augmentation for supervised decoder training
class PairedDepthAugmentation:
    def __init__(
        self,
        policy: AugmentationPolicy = "meter",
        view_type: Literal['global', 'local'] | None = None
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown augmentation policy {policy!r}")
        self.policy = policy
        self.config = POLICIES[policy]

        if policy == 'lejepa_multi_view':
            if view_type not in ('global', 'local'):
                raise ValueError(
                    "view_type must be 'global' or 'local'"
                    "for 'lejepa_multi_view' augmentation"
                )
            self.cfg = self.config[view_type]
            cfg = self.cfg
            self.appearance = v2.Compose([
                v2.RandomApply([v2.ColorJitter(*cfg["color_jitter_strength"])], p=cfg["color_jitter"]),
                v2.RandomGrayscale(p=cfg["grayscale"]),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=cfg["gaussian_blur_kernel"], sigma=cfg["gaussian_blur_sigma"])],
                    p=cfg["gaussian_blur"],
                ),
                v2.RandomApply([v2.RandomSolarize(threshold=cfg["solarize_threshold"])], p=cfg["solarize"]),
            ])
        elif policy == "lemeter":
            self._meter = PairedDepthAugmentation("meter")
            self._lejepa = PairedDepthAugmentation("lejepa")
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
            self.cfg = self.config
            self.appearance = None

    def _joint_crop(self, image: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the same relative crop window to image and full-res depth."""
        cfg = self.cfg if hasattr(self, 'cfg') else self.config
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=list(cfg["random_crop_scale"]),
            ratio=list(cfg["random_crop_ratio"]),
        )
        sh = depth_cm.shape[-2] / image.shape[-2]
        sw = depth_cm.shape[-1] / image.shape[-1]
        depth_cm = TF.crop(
            depth_cm, round(top * sh), round(left * sw), round(height * sh), round(width * sw)
        )
        image = TF.resized_crop(
            image, top, left, height, width, list(INPUT_RESOLUTION), antialias=True
        )
        return image, depth_cm

    def __call__(self, image: Tensor, depth_cm: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg if hasattr(self, 'cfg') else self.config

        if self.policy == "lemeter":
            # coin flip per sample: full meter OR full lejepa policy, never both
            chosen = self._meter if torch.rand(()) < cfg["meter_prob"] else self._lejepa
            return chosen(image, depth_cm)

        if self.policy == 'lejepa_multi_view':
            if torch.rand(()) < cfg["mirror"]:
                image = torch.flip(image, dims=[-1])
                depth_cm = torch.flip(depth_cm, dims=[-1])

            if torch.rand(()) < cfg["random_crop"]:
                image, depth_cm = self._joint_crop(image, depth_cm)

            assert self.appearance is not None
            image = self.appearance(image / 255.0) * 255.0
            return image, depth_cm

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
