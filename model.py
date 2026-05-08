import torch.nn as nn

from derf import Dynamic_erf

class Conv2dBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.act(x2)
        return x3

class DerfTransformer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = Dynamic_erf(d_model, channels_last=True)
        self.norm2 = Dynamic_erf(d_model, channels_last=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        a = self.norm1(x)
        attn_out, _ = self.attn(a, a, a, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
