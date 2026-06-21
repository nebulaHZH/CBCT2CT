from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) * (c**-0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    """用于 Peng 风格条件 DDPM 噪声预测的 2D 条件 U-Net。"""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        downs = []
        channels = [base_channels]
        current = base_channels
        for mult in channel_mults:
            out = base_channels * mult
            downs.append(nn.ModuleList([ResBlock(current, out, time_dim), ResBlock(out, out, time_dim), nn.Conv2d(out, out, 4, 2, 1)]))
            channels.append(out)
            current = out
        self.downs = nn.ModuleList(downs)

        self.mid1 = ResBlock(current, current, time_dim)
        self.attn = AttentionBlock(current) if use_attention else nn.Identity()
        self.mid2 = ResBlock(current, current, time_dim)

        ups = []
        for mult, skip_channels in zip(reversed(channel_mults), reversed(channels[:-1])):
            out = base_channels * mult
            ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(current, out, 4, 2, 1),
                        ResBlock(out + skip_channels, out, time_dim),
                        ResBlock(out, out, time_dim),
                    ]
                )
            )
            current = out
        self.ups = nn.ModuleList(ups)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(current), current),
            nn.SiLU(),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timesteps)
        x = torch.cat([noisy_target, condition], dim=1)
        x = self.input(x)
        skips = [x]
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skips.append(x)
            x = downsample(x)
        x = self.mid1(x, t)
        x = self.attn(x)
        x = self.mid2(x, t)
        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t)
            x = block2(x, t)
        return self.output(x)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
