# LeMETER

Self-supervised pretraining of a MobileViT-based monocular depth encoder using
LeJEPA (SIGReg + invariance loss), followed by supervised decoder fine-tuning
on NYU Depth V2.

## Pipeline overview

`main.py` runs the full pipeline in order:

1. `download_datasets()` - downloads NYU Depth V2 via `kagglehub` into
   `datasets/`.
2. `preprocess_datasets()` - resizes RGB frames to the model's input
   resolution and converts depth maps to a canonical uint16-centimeter
   format under `preprocessed_datasets/`. Writes a completion marker so this
   step is skipped on subsequent runs.
3. `pretrain_lejepa_encoder(...)` - self-supervised pretraining of the
   MobileViT encoder with the LeJEPA objective (SIGReg regularization +
   view-invariance loss), tracked with a dense per-pixel depth probe that is
   detached from the encoder gradient.
4. `train_decoder(...)` (optional, via `--train-decoder`) - loads an encoder
   checkpoint from step 3 and trains a decoder head on top of it, using one
   of three schedules (`warm_start`, `freeze_encoder`, `finetune`).

Each stage checkpoints to disk and resumes automatically unless
`--no-resume` is passed.

## Requirements

- Python 3.14 (the committed `.venv` and cached bytecode target `cpython-314`)
- PyTorch 2.12, torchvision 0.27
- See `requirements.txt` for the full pinned dependency list

Install dependencies depending on your hardware:

```bash
# CPU / Apple Silicon (MPS)
pip install -r requirements.txt

# NVIDIA CUDA (12.x)
pip install -r requirements.cuda.txt

# AMD ROCm (tested on RX 9060 XT / gfx1200)
pip install -r requirements.amd.txt
```

Only NYU Depth V2 is currently wired up end to end. `kitti` is accepted as a
`--dataset` value in the CLI and referenced in the architecture code, but
`download_datasets.py` and `preprocessing.py` only implement the NYU path -
selecting `kitti` will fail before training starts.

## Running training

Main command, pretrains the `xxs`-size encoder on NYU and then fine-tunes the
decoder jointly with it:

```bash
python main.py --name lemeter --arch xxs --train-decoder --decoder-schedule finetune
```

The run name `lemeter` is significant: `LeMeterEncoder.load` in `meter.py`
hardcodes this run name to load the pretrained encoder checkpoint from
`runs/nyu_s_lemeter_encoder/last_checkpoint.pt`. Use this exact `--name` (with
the default `--dataset nyu`) if you want `LeMeterEncoder.load(...)` to find
the checkpoint without modification.

This downloads and preprocesses NYU Depth V2 on first run (skipped on later
runs once the completion markers exist), then pretrains the encoder for
`globals.PRETRAIN_EPOCHS` (20) epochs, then trains the decoder for 60 epochs
with the `finetune` schedule (encoder and decoder both trainable from
epoch 0).

Basic invocation, encoder pretraining only (no decoder):

```bash
python main.py --name pretrain_encoder
```

### CLI arguments (`main.py`)

| Flag | Default | Choices | Description |
|---|---|---|---|
| `--name` | (required) | any string | Run name. Checkpoints and logs are written to `runs/<dataset>_<arch>_<name>_encoder/` and, if decoder training is enabled, `runs/<dataset>_<arch>_<name>_decoder_<schedule>/`. |
| `--config`, `-c` | `default` | `default`, `rx-9060xt` | Hardware acceleration profile. `default` auto-selects MPS, then CUDA, then CPU. `rx-9060xt` sets the ROCm environment variables needed for that specific AMD GPU. |
| `--no-resume` | resume enabled | flag | Disable resuming from the last checkpoint in the run directory; starts training from scratch. |
| `--arch` | `xxs` | `xxs`, `xs`, `s` | MobileViT backbone size for the encoder/decoder. |
| `--dataset` | `nyu` | `nyu`, `kitti` | Dataset to train on. Only `nyu` is currently functional (see above). |
| `--train-decoder` | disabled | flag | After encoder pretraining completes, also train the decoder head. |
| `--decoder-schedule` | `warm_start` | `warm_start`, `freeze_encoder`, `finetune` | Decoder training schedule, see below. |
| `--checkpoint-epoch` | `20` (`globals.PRETRAIN_EPOCHS`) | int | Which encoder snapshot epoch to load when training the decoder. Encoder snapshots are saved every 5 epochs and at the final epoch. |

### Decoder schedules

Defined in `train_decoder.py`:

| Schedule | Epochs | Encoder LR | Decoder LR | Behavior |
|---|---|---|---|---|
| `freeze_encoder` | 60 | 0.0 | 1e-3 | Encoder weights are frozen for the entire run; only the decoder is trained. |
| `finetune` | 60 | 1e-3 | 1e-3 | Encoder and decoder are both trained from epoch 0. |
| `warm_start` | 65 | 1e-3 (from epoch 5) | 1e-3 | Encoder is frozen for the first 5 epochs, then unfrozen and trained jointly with the decoder for the remainder. |

Both optimizer groups use AdamW with weight decay `1e-2`, and the learning
rate decays by a factor of 10 every 20 epochs (`StepLR`).

## Examples

Pretrain an `xxs` encoder on NYU with default settings:

```bash
python main.py --name 01baseline
```

Pretrain an `s`-size encoder and immediately fine-tune the decoder with the
`finetune` schedule:

```bash
python main.py --name 02full --arch s --train-decoder --decoder-schedule finetune
```

Train only the decoder on top of an existing encoder checkpoint at epoch 15,
without resuming any partially completed decoder run:

```bash
python main.py --name 01baseline --train-decoder --checkpoint-epoch 15 --no-resume
```

Run on the AMD RX 9060 XT with the dedicated ROCm configuration:

```bash
python main.py --name amd_run --config rx-9060xt --train-decoder
```

Resume an interrupted run (default behavior, no flag needed):

```bash
python main.py --name 01baseline --train-decoder
```

## Run outputs

Each run directory (`runs/<dataset>_<arch>_<name>_encoder/` or
`..._decoder_<schedule>/`) contains:

- `last_checkpoint.pt` - full training state (model, optimizer, scheduler,
  GradScaler, RNG state, history) used to resume training.
- `best_checkpoint.pt` (decoder only) - checkpoint with the best validation
  RMSE seen so far.
- `encoder_epoch_XXX.pt` (encoder only) - lightweight encoder-only snapshots
  saved every 5 epochs and at the final epoch, used as inputs to decoder
  training.
- `losses.json` - per-epoch metric history, rewritten every epoch.
- `test_metrics.json` (decoder only) - final test-set metrics computed with
  the best checkpoint, written once training completes.

## Individual pipeline stages

Each stage can also be run standalone:

```bash
python download_datasets.py     # download NYU Depth V2 into datasets/
python preprocessing.py         # resize + convert depth maps into preprocessed_datasets/
python lejepa.py --name <name>  # encoder pretraining only (same CLI args as main.py)
python train_decoder.py --name <name> --train-decoder ...  # decoder training only
```

`lejepa.py` and `train_decoder.py` share the same argument parser as
`main.py` (`cli.parse_cli_args`).

## Other scripts

These are auxiliary analysis/evaluation entry points, not part of the main
training pipeline:

- `train_supervised.py` - trains a `GaussianMeter` model end to end with
  direct supervision (no LeJEPA pretraining), used as a baseline.
- `eval_gaussian_meter.py` - evaluates a fixed set of supervised ablation
  checkpoints (DeRF/LayerNorm x ReLU/GELU) on the NYU test set.
- `pca_analysis.py` - PCA probing of encoder feature maps at multiple depths
  (`--arch`, `--dataset`, `--output`, see `--help` for full options).
- `meter_gaussianity.py` - checks Gaussianity of pooled encoder embeddings
  along random directions and plots the result.

## Hardware configuration notes

- `--config default` auto-detects MPS (Apple Silicon), then CUDA, then falls
  back to CPU.
- `--config rx-9060xt` sets ROCm-specific environment variables
  (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`, `HSA_OVERRIDE_GFX_VERSION`,
  `HSA_XNACK`, `TORCH_BLAS_PREFER_HIPBLASLT`) required for that GPU and
  asserts that a CUDA-mapped (ROCm) device is available.
- `torch.compile` is enabled automatically when running on CUDA/ROCm and
  skipped on MPS/CPU, where it is not stable.
- Mixed precision (`bfloat16` autocast, `GradScaler`) is used during the
  forward/backward pass in both pretraining and decoder training.

## Key hyperparameters (`globals.py`)

- `INPUT_RESOLUTION`: 192 x 256 (height x width), downsampled from the
  native NYU 480 x 640 frame.
- `SEED`: 3407, applied via `torch.manual_seed` at the start of every stage.
- `BATCH_SIZE`: 64 (pretraining).
- `LEARNING_RATE`: 1e-3.
- `PRETRAIN_EPOCHS`: 20.
- `LAMBDA`: 5e-2, weight of the SIGReg term relative to the invariance term
  in the LeJEPA loss (`sigreg * LAMBDA + invariance * (1 - LAMBDA)`).
- `VIEWS` / `GLOBAL_VIEWS` / `LOCAL_VIEWS`: 8 total augmented views per
  sample, 2 global and 6 local.

Decoder-specific hyperparameters (batch size, weight decay, LR schedule,
epoch counts per schedule) live in `train_decoder.py` as module-level
constants.
