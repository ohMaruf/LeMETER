import torch

NYU_IMAGE_RESOLUTION = (480, 640)  # height, width (rows x columns notation)
INPUT_RESOLUTION = (192, 256)  # height, width (rows x columns notation)

# Two augmentation presets (probabilities and ranges):
# - METER: the conservative supervised policy from the METER paper (Sec. III-C,
#   Table VI) — mirror, gentle crop, ±10% gamma/brightness/color on the 0-255
#   scale, ±10 cm depth shift (labels only). Used by DepthTrainDataset.
# - LEJEPA: the SSL view-generation recipe (LeJEPA/DINO-style) — aggressive
#   crops and strong photometric jitter, since invariance to them is the
#   training signal. Default for AugmentedNyuDataset.
METER_AUGMENTATION = {
    "mirror": 0.9,
    "random_crop": 0.9,
    "random_crop_scale": (0.6, 1.0),
    "random_crop_ratio": (0.75, 4 / 3),
    "shifting_strategy": 0.9,
}

LEJEPA_AUGMENTATION = {
    "mirror": 0.5,
    "random_crop_scale": (0.4, 1.0),
    "color_jitter": 0.8,
    "grayscale": 0.2,
    "gaussian_blur": 0.5,
    "solarize": 0.2,
}

SEED = 3407

# the training box has 32 cores; augmentation (8 views per sample) is the
# pipeline bottleneck, so most of them should feed the DataLoader. RAM cost is
# workers x prefetch_factor x batch size (~290 MiB for a pretraining batch).
DATALOADER_WORKERS = 24

FLOATING_PRECISION = torch.float32
NUM_SLICES = 256
PROJ_DIM = 128
EMBEDDING_DIM = 160
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# pretraining starts from a random encoder, so it needs a full schedule
# (original METER was trained for 60 epochs); the decoder alone converges faster
PRETRAIN_EPOCHS = 20
DECODER_EPOCHS = 20
LAMBDA = 5e-2
VIEWS = 8

# unit convention (validated against the published METER numbers): the model
# works in centimeters — eval.run_inference multiplies its output by 10 to get
# the millimeters of the test labels, and loss.py's ssim(val_range=1000)
# assumes targets in [0, 1000]. Train depth maps are uint8 with 255 == 10 m.
TRAIN_DEPTH_TO_CM = 1000.0 / 255.0
OUTPUT_RESOLUTION = (INPUT_RESOLUTION[0] // 4, INPUT_RESOLUTION[1] // 4)