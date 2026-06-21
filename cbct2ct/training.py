from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from cbct2ct.torch_utils import require_torch


def should_use_mixed_precision(device: str, requested: bool) -> bool:
    return requested and device.startswith("cuda")


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]):
    torch = require_torch()
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "official_cddpm")
    if name in {"official_cddpm", "cddpm"}:
        from cbct2ct.models.official_cddpm import OfficialConditionalDDPM, OfficialConditionalUNet

        denoiser = OfficialConditionalUNet(
            timesteps=int(model_cfg.get("timesteps", 1000)),
            base_channels=int(model_cfg.get("base_channels", 128)),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 3, 4])),
            attention_levels=tuple(model_cfg.get("attention_levels", [2])),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
        return OfficialConditionalDDPM(
            denoiser=denoiser,
            timesteps=int(model_cfg.get("timesteps", 1000)),
            beta_start=float(model_cfg.get("beta_start", 1e-4)),
            beta_end=float(model_cfg.get("beta_end", 2e-2)),
            loss=str(model_cfg.get("loss", "mse_sum")),
        )
    if name == "resunet":
        from cbct2ct.models.resunet import ResUNet2D

        return ResUNet2D(
            base_channels=int(model_cfg.get("base_channels", 32)),
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
        )
    raise ValueError(f"不支持的 model.name：{name}")


def parameter_statistics(model, bytes_per_param: int = 4) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "parameter_bytes": total * bytes_per_param,
        "adamw_training_bytes": estimate_adamw_memory_bytes(total, trainable, bytes_per_param),
    }


def estimate_adamw_memory_bytes(
    total_parameters: int,
    trainable_parameters: int,
    bytes_per_param: int = 4,
) -> int:
    parameter_bytes = total_parameters * bytes_per_param
    gradient_bytes = trainable_parameters * bytes_per_param
    optimizer_state_bytes = trainable_parameters * bytes_per_param * 2
    return parameter_bytes + gradient_bytes + optimizer_state_bytes


def format_int(value: int) -> str:
    return f"{value:,}"


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        amount /= 1024.0
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
    raise RuntimeError("unreachable")


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, config: dict[str, Any]) -> None:
    torch = require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config,
        },
        output,
    )


def load_checkpoint(path: str | Path, model, device: str):
    torch = require_torch()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
