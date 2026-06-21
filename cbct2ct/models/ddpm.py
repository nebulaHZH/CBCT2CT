from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalDDPM(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, target: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(target)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, target.shape) * target
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, target.shape) * noise
        )

    def p_losses(self, target: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = target.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=target.device, dtype=torch.long)
        noise = torch.randn_like(target)
        noisy_target = self.q_sample(target, timesteps, noise)
        predicted_noise = self.denoiser(noisy_target, condition, timesteps)
        loss = F.l1_loss(predicted_noise, noise, reduction="none")
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp_min(1.0)
        return loss.mean()

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        betas_t = _extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alpha_bars, timesteps, x.shape)
        sqrt_recip = _extract(self.sqrt_recip_alphas, timesteps, x.shape)
        model_mean = sqrt_recip * (x - betas_t * self.denoiser(x, condition, timesteps) / sqrt_one_minus)
        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        image = torch.randn(shape or condition.shape, device=condition.device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((image.shape[0],), step, device=condition.device, dtype=torch.long)
            image = self.p_sample(image, condition, timesteps)
        return image.clamp(-1.0, 1.0)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))
