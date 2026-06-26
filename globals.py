import torch

NYU_IMAGE_RESOLUTION = (480, 640)  # height, width (rows x columns notation)
INPUT_RESOLUTION = (192, 256)  # height, width (rows x columns notation)
OUTPUT_RESOLUTION = (INPUT_RESOLUTION[0] // 4, INPUT_RESOLUTION[1] // 4)

SEED = 3407

# the training box has 32 cores; augmentation (8 views per sample) is the
# pipeline bottleneck, so most of them should feed the DataLoader. RAM cost is
# workers x prefetch_factor x batch size (~290 MiB for a pretraining batch).
DATALOADER_WORKERS = 24

FLOATING_PRECISION = torch.float32
NUM_SLICES = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# pretraining starts from a random encoder, so it needs a full schedule
# (original METER was trained for 60 epochs); the decoder alone converges faster
PRETRAIN_EPOCHS = 20
LAMBDA = 5e-2
VIEWS = 8

GLOBAL_VIEWS = 2
LOCAL_VIEWS = 6
LOCAL_RESOLUTION = (128, 128)
GLOBAL_CROP_SCALE = (0.3, 1.0)
LOCAL_CROP_SCALE = (0.05, 0.3)
GLOBAL_POLICY = "lejepa"
LOCAL_POLICY = "lejepa"
