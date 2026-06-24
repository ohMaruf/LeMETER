from train_decoder import DecoderSchedule
import globals
from meter import MeterArchitecture
import argparse

from hardware_acceleration import Config
from typing import NamedTuple
from dataset import DepthDataset


class CliArgs(NamedTuple):
    config: Config
    resume: bool
    name: str
    arch: MeterArchitecture
    dataset: DepthDataset
    checkpoint_epoch: int
    train_decoder: bool
    decoder_schedule: DecoderSchedule


def parse_cli_args() -> CliArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=Config,
        choices=list(Config),
        default=Config.DEFAULT,
        help="the gpu configuration to use",
    )

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="disable resuming training from previous runs",
    )

    parser.add_argument(
        "--name",
        default="pretrain_encoder",
        help="run name; checkpoints and logs go to runs/<name>/",
        required=True,
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["xxs", "xs", "s"],
        default="xxs",
        help="the model architecture to use",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nyu", "kitti"],
        default="nyu",
        help="the dataset to use",
    )

    parser.add_argument(
        "--train-decoder",
        dest="train_decoder",
        action="store_true",
        default=False,
        help="enable training decoder",
    )

    parser.add_argument(
        "--decoder-schedule",
        dest="decoder_schedule",
        default="warm_start",
        type=str,
        choices=["warm_start", "freeze_encoder", "finetune"],
        help="decoder training schedule",
    )

    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=globals.PRETRAIN_EPOCHS,
        help="epoch of encoder checkpoint to load for decoder training",
    )

    args = parser.parse_args()
    return CliArgs(**vars(args))
