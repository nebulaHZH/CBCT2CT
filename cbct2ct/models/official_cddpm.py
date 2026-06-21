from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class OfficialTimeEmbedding(nn.Module):
    """对齐官方 conditional_DDPM 的固定正弦时间嵌入。"""

    def __init__(self, timesteps: int, base_channels: int, embedding_dim: int) -> None:
        super().__init__()
        if base_channels % 2 != 0:
            raise ValueError("官方时间嵌入要求 base_channels 为偶数")
        positions = torch.arange(timesteps).float()
        frequencies = torch.arange(0, base_channels, step=2).float() / base_channels
        frequencies = torch.exp(-frequencies * math.log(10000))
        table = positions[:, None] * frequencies[None, :]
        table = torch.stack([torch.sin(table), torch.cos(table)], dim=-1).view(timesteps, base_channels)
        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(table, freeze=True),
            nn.Linear(base_channels, embedding_dim),
            Swish(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.embedding(timesteps)


class OfficialDownSample(nn.Module):
    """官方实现使用 stride=2 的 3x3 卷积下采样。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OfficialUpSample(nn.Module):
    """官方实现先最近邻上采样，再接 3x3 卷积。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class OfficialAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in [self.q, self.k, self.v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        k = self.k(h).reshape(batch, channels, height * width)
        weights = torch.softmax(torch.bmm(q, k) * (channels**-0.5), dim=-1)
        v = self.v(h).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        h = torch.bmm(weights, v).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return x + self.proj(h)


class OfficialResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float, attention: bool) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_proj = nn.Sequential(Swish(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = OfficialAttentionBlock(out_channels) if attention else nn.Identity()
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.block2(h)
        return self.attention(h + self.shortcut(x))


class OfficialConditionalUNet(nn.Module):
    """官方 conditional_DDPM 的 UNet 结构本地实现。

    结构来源：junbopeng/conditional_DDPM 的 Model_condition.py。
    这里没有直接复制外部文件，而是按官方公开结构在本工程中实现，
    便于接入 SynthRAD manifest、中文配置和本地训练脚本。
    """

    def __init__(
        self,
        timesteps: int = 1000,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 3, 4),
        attention_levels: tuple[int, ...] = (2,),
        num_res_blocks: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if any(level >= len(channel_mults) or level < 0 for level in attention_levels):
            raise ValueError("attention_levels 中存在超出 channel_mults 范围的索引")

        time_dim = base_channels * 4
        self.time_embedding = OfficialTimeEmbedding(timesteps, base_channels, time_dim)
        self.head = nn.Conv2d(2, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        skip_channels = [base_channels]
        current_channels = base_channels
        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    OfficialResBlock(
                        current_channels,
                        out_channels,
                        time_dim,
                        dropout,
                        attention=level in attention_levels,
                    )
                )
                current_channels = out_channels
                skip_channels.append(current_channels)
            if level != len(channel_mults) - 1:
                self.downblocks.append(OfficialDownSample(current_channels))
                skip_channels.append(current_channels)

        self.middleblocks = nn.ModuleList(
            [
                OfficialResBlock(current_channels, current_channels, time_dim, dropout, attention=True),
                OfficialResBlock(current_channels, current_channels, time_dim, dropout, attention=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    OfficialResBlock(
                        skip_channels.pop() + current_channels,
                        out_channels,
                        time_dim,
                        dropout,
                        attention=level in attention_levels,
                    )
                )
                current_channels = out_channels
            if level != 0:
                self.upblocks.append(OfficialUpSample(current_channels))

        if skip_channels:
            raise RuntimeError("官方 U-Net 的 skip 连接数量不匹配，请检查网络配置")

        self.tail = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            Swish(),
            nn.Conv2d(current_channels, 1, 3, padding=1),
        )
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(timesteps)
        h = self.head(x)
        skips = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            skips.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, OfficialResBlock):
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
            h = layer(h, temb)
        if skips:
            raise RuntimeError("官方 U-Net 前向传播结束时仍残留 skip 特征")
        return self.tail(h)


class OfficialConditionalDDPM(nn.Module):
    """官方 conditional_DDPM 训练器和采样器的本地封装。"""

    official_repo = "https://github.com/junbopeng/conditional_DDPM"
    paper_doi = "10.1002/mp.16704"

    def __init__(
        self,
        denoiser: OfficialConditionalUNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss: str = "mse_sum",
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.loss = loss
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars, [1, 0], value=1.0)[:timesteps]

        self.register_buffer("betas", betas.float())
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars).float())
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars).float())
        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas).float())
        self.register_buffer("coeff2", (torch.sqrt(1.0 / alphas) * (1.0 - alphas) / torch.sqrt(1.0 - alpha_bars)).float())
        self.register_buffer("posterior_var", (betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)).float())

    def p_losses(self, target: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = target.shape[0]
        timesteps = torch.randint(self.timesteps, size=(batch,), device=target.device)
        noise = torch.randn_like(target)
        noisy_target = (
            _extract(self.sqrt_alpha_bars, timesteps, target.shape) * target
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, target.shape) * noise
        )
        official_input = torch.cat([noisy_target, condition], dim=1)
        predicted_noise = self.denoiser(official_input, timesteps)
        if self.loss == "masked_mse_mean":
            if mask is None:
                raise ValueError("masked_mse_mean 损失需要提供 mask")
            return masked_noise_mse(predicted_noise, noise, mask)
        if self.loss == "mse_mean":
            return F.mse_loss(predicted_noise, noise)
        return F.mse_loss(predicted_noise, noise, reduction="sum")

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        target_shape = shape or condition.shape
        ct = torch.randn(target_shape, device=condition.device)
        x_t = torch.cat([ct, condition], dim=1)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((target_shape[0],), step, device=condition.device, dtype=torch.long)
            ct, x_t = self._sample_step(x_t, timesteps)
        return ct.clamp(-1.0, 1.0)

    def _sample_step(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ct = x_t[:, 0:1]
        cbct = x_t[:, 1:2]
        eps = self.denoiser(x_t, timesteps)
        mean = _extract(self.coeff1, timesteps, ct.shape) * ct - _extract(self.coeff2, timesteps, ct.shape) * eps
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = _extract(var, timesteps, ct.shape)
        noise = torch.randn_like(ct)
        nonzero = (timesteps != 0).float().reshape(ct.shape[0], *((1,) * (ct.ndim - 1)))
        ct = mean + nonzero * torch.sqrt(var) * noise
        x_t = torch.cat([ct, cbct], dim=1)
        if torch.isnan(x_t).any():
            raise RuntimeError("采样过程中出现 NaN，请检查模型权重或输入归一化")
        return ct, x_t


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, timesteps).float().to(timesteps.device)
    return out.view([timesteps.shape[0]] + [1] * (len(shape) - 1))


def masked_noise_mse(predicted_noise: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (predicted_noise - noise).square()
    weights = mask.to(device=squared_error.device, dtype=squared_error.dtype)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
