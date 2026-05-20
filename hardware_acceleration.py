import os
import logger
import torch
import psutil

from enum import Enum


class Config(Enum):
    DEFAULT = 'default'
    RX9060XT = 'rx-9060xt'

    def __str__(self):
        return self.value


def _set_environ(key: str, value: str, verbose: bool = True) -> None:
    os.environ[key] = value
    if verbose:
        logger.info(f'{key}={os.environ[key]}')


def print_device_memory_snapshot(device: torch.device, step_name: str):
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"[{step_name}] Allocated VRAM: {allocated:.2f}GB | Reserved VRAM: {reserved:.2f}GB")  # noqa: E501
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver_mem = torch.mps.driver_allocated_memory() / (1024 ** 3)
        logger.info(f"[{step_name}] Tensors VRAM: {allocated:.2f}GB | Metal Driver Total VRAM: {driver_mem:.2f}GB")  # noqa: E501
    elif device.type == "cpu":
        process = psutil.Process()
        ram_gb = process.memory_info().rss / (1024 ** 3)
        logger.info(f"[{step_name}] Total Process RAM: {ram_gb:.2f}GB")


def _default_config(config: Config) -> torch.device:
    logger.warn(f'hardware acceleration is not available for {config} config')  # noqa: E501
    logger.info('run with --config to select a different configuration')
    if torch.backends.mps.is_available():
        logger.warn('using torch mps backend')
        return torch.device('mps')
    else:
        logger.warn('using torch cpu backend')
        return torch.device('cpu')


def _rx9060xt_config(config: Config) -> torch.device:
    logger.info(f'enabling hardware acceleration for {config}')
    _set_environ('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
    _set_environ('HSA_OVERRIDE_GFX_VERSION', '12.0.0')
    _set_environ('HSA_XNACK', '0')

    cuda_available = torch.cuda.is_available()
    assert cuda_available, f'ROCM (CUDA over AMD) not available for {config}'  # noqa: E501
    logger.info(f'ROCm (CUDA over AMD) Available: {cuda_available}')

    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    compute_cap = torch.cuda.get_device_capability(0)

    logger.info(f'ROCm Device Name: {device_name}')
    logger.info(f'Compute Capability (HIP Proxy): {compute_cap}')

    total_vram_gb = props.total_memory / (1024 ** 3)
    logger.info(f'Total Device VRAM: {total_vram_gb:.2f} GB')

    logger.info(f'PyTorch Version: {torch.__version__}')
    logger.info(f'Compiled ROCm/HIP Version: {torch.version.cuda}')
    return torch.device('cuda')


def enable_hardware_acceleration(config: Config) -> torch.device:
    if config == Config.DEFAULT:
        return _default_config(config)
    elif config == Config.RX9060XT:
        return _rx9060xt_config(config)
    else:
        assert False, f"unreachable, got unknown config {config}"
