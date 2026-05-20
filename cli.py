import argparse

from hardware_acceleration import Config
from typing import NamedTuple


class CliArgs(NamedTuple):
    config: Config
    resume: bool


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

    args = parser.parse_args()
    return CliArgs(**vars(args))
