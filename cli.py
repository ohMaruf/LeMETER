from meter import MeterArchitecture
import argparse

from hardware_acceleration import Config
from typing import NamedTuple
from dataset import DepthDataset


class CliArgs(NamedTuple):
    config: Config
    resume: bool
    name: str
    arch: MeterArchitecture = "xxs"
    dataset: DepthDataset = "nyu"

def parse_cli_args() -> CliArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        '-c',
        type=Config,
        choices=list(Config),
        default=Config.DEFAULT,
        help='the gpu configuration to use',
    )

    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='disable resuming training from previous runs',
    )

    parser.add_argument(
        '--name',
        default='pretrain_encoder',
        help='run name; checkpoints and logs go to runs/<name>/',
    )
    
    parser.add_argument(
        '--arch',
        type=str,
        choices=["xxs", "xs", "s"],
        default="xxs",
        help='the model architecture to use',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=["nyu", "kitti"],
        default="nyu",
        help='the dataset to use',
    )

    args = parser.parse_args()
    return CliArgs(**vars(args))
