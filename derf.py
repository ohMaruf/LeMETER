from typing import Literal
import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicErf(nn.Module):
    def __init__(self, normalized_shape, channels_last: bool, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        x = self.alpha * x + self.shift
        x = torch.erf(x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, shift_init_value={self.shift_init_value}, channels_last={self.channels_last}"


def convert_ln_to_derf(module: nn.Module, alpha_init_value=0.5, shift_init_value=0.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicErf(module.normalized_shape, not isinstance(module, LayerNorm2d), alpha_init_value, shift_init_value)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_derf(child, alpha_init_value, shift_init_value))
    del module
    return module_output


def convert_relu_to_gelu(module: nn.Module, approximate: Literal["tanh", "none"] = "tanh"):
    module_output = module
    if isinstance(module, nn.ReLU):
        module_output = nn.GELU(approximate=approximate)
    for name, child in module.named_children():
        module_output.add_module(
            name,
            convert_relu_to_gelu(child),
        )
    del module
    return module_output