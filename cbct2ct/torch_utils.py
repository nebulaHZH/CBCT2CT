from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "训练和推理需要安装 PyTorch。"
            "请根据你的 CUDA 版本安装对应的 PyTorch，例如："
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    return torch
