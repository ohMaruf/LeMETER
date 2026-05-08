NYU_IMAGE_RESOLUTION = (480, 640)  # height, width (rows x columns notation)
INPUT_RESOLUTION = (192, 256)  # height, width (rows x columns notation)

augmentation_parameters = {
    # We avoid flip-based geometry for NYU indoor scenes and instead rely on
    # stronger crop and photometric perturbations for LeJEPA view generation.
    "flip": 0.0,
    "mirror": 0.9,
    "c_swap": 0.0,
    "random_crop": 0.9,
    "random_crop_scale": (0.6, 1.0),
    "random_crop_ratio": (0.75, 1.3333333333),
    "shifting_strategy": 0.9,
}

dts_type = "nyu-depth-v2"
