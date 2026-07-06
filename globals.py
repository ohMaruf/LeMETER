import os
import torch

NYU_IMAGE_RESOLUTION = (480, 640)  # height, width (rows x columns notation)
INPUT_RESOLUTION = (192, 256)  # height, width (rows x columns notation)
OUTPUT_RESOLUTION = (INPUT_RESOLUTION[0] // 4, INPUT_RESOLUTION[1] // 4)

SEED = 3407
DATALOADER_WORKERS = min(24, os.cpu_count() or 1)
FLOATING_PRECISION = torch.float32

# pretraining hyperparameters
NUM_SLICES = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PRETRAIN_EPOCHS = 20
LAMBDA = 5e-2
VIEWS = 8
GLOBAL_VIEWS = 2
LOCAL_VIEWS = VIEWS - GLOBAL_VIEWS
