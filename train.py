from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from cbct2ct.cli import ChineseArgumentParser
from cbct2ct.config import load_config
from cbct2ct.data.samplers import CaseBlockSampler
from cbct2ct.data.torch_dataset import SynthRADSliceDataset, pad_slice_batch
from cbct2ct.torch_utils import require_torch
from cbct2ct.training import (
    build_model,
    format_bytes,
    format_int,
    parameter_statistics,
    save_checkpoint,
    set_seed,
    should_use_mixed_precision,
)


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="在 SynthRAD manifest 上训练 CBCT-to-CT 基线模型。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--override", action="append", default=[], help="点号形式的配置覆盖，例如 training.epochs=2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(item.split("=", 1) for item in args.override)
    config = load_config(args.config, overrides=overrides)
    torch = require_torch()
    seed = int(config.get("seed", 42))
    set_seed(seed)

    training_cfg = config.get("training", {})
    device = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = should_use_mixed_precision(device, bool(training_cfg.get("mixed_precision", False)))
    data_cfg = config.get("data", {})
    dataset = SynthRADSliceDataset(
        manifest_path=data_cfg["train_manifest"],
        hu_min=float(data_cfg.get("hu_min", -1000.0)),
        hu_max=float(data_cfg.get("hu_max", 2000.0)),
        axis=int(data_cfg.get("axis", 2)),
        min_mask_pixels=int(data_cfg.get("min_mask_pixels", 32)),
        max_slices_per_case=data_cfg.get("max_slices_per_case"),
    )
    case_block_size = int(data_cfg.get("case_block_size", 0) or 0)
    sampler = CaseBlockSampler(dataset.index, block_size=case_block_size, seed=seed) if case_block_size > 0 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=device.startswith("cuda"),
        collate_fn=pad_slice_batch,
    )

    model = build_model(config).to(device)
    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 4))
    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "runs/checkpoints"))
    model_name = config.get("model", {}).get("name", "official_cddpm")
    _print_training_summary(
        torch=torch,
        config=config,
        model=model,
        dataset_size=len(dataset),
        loader_size=len(loader),
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        model_name=model_name,
        use_amp=use_amp,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    timing_every = max(1, int(training_cfg.get("timing_every", 50)))
    synchronize_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = _progress_bar(loader, desc=f"epoch {epoch}/{epochs}", unit="batch")
        iterator = iter(pbar)
        for step in range(1, len(loader) + 1):
            data_started = perf_counter()
            batch = next(iterator)
            data_seconds = perf_counter() - data_started
            should_time_step = step == 1 or step % timing_every == 0
            if should_time_step and synchronize_cuda:
                torch.cuda.synchronize()
            step_started = perf_counter()
            condition = batch["condition"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            if epoch == 1 and step == 1:
                _print_first_batch_summary(condition, target, mask, batch)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                if hasattr(model, "p_losses"):
                    loss = model.p_losses(target, condition, mask=mask)
                else:
                    pred = model(condition)
                    loss = ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if should_time_step and synchronize_cuda:
                torch.cuda.synchronize()
            step_seconds = perf_counter() - step_started
            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            average_loss = total_loss / step
            progress = {
                "loss": f"{loss_value:.6f}",
                "avg": f"{average_loss:.6f}",
                "shape": "x".join(str(value) for value in condition.shape),
                "cuda": _cuda_progress_text(torch, device),
            }
            if should_time_step:
                progress.update(_format_training_timing(data_seconds, step_seconds, batch_size))
            pbar.set_postfix(**progress)
        print(f"轮次={epoch} 损失={total_loss / max(len(loader), 1):.6f}")
        if epoch % int(config.get("training", {}).get("save_every", 1)) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, config)
        _run_validation_preview(torch, model, dataset, device, config, epoch)


def _print_training_summary(
    torch: Any,
    config: dict[str, Any],
    model: Any,
    dataset_size: int,
    loader_size: int,
    batch_size: int,
    epochs: int,
    device: str,
    model_name: str,
    use_amp: bool,
) -> None:
    stats = parameter_statistics(model)
    experiment_name = config.get("experiment", {}).get("name", "unnamed")
    print("=== Training summary ===")
    print(f"experiment={experiment_name} model={model_name} device={device}")
    print(f"混合精度={use_amp}")
    print(
        "dataset_slices="
        f"{format_int(dataset_size)} batch_size={batch_size} "
        f"steps_per_epoch={format_int(loader_size)} epochs={epochs} "
        f"total_steps={format_int(loader_size * max(epochs, 0))}"
    )
    print(
        "parameters="
        f"{format_int(stats['total'])} trainable={format_int(stats['trainable'])} "
        f"frozen={format_int(stats['frozen'])}"
    )
    print(
        "estimated_model_memory="
        f"parameters={format_bytes(stats['parameter_bytes'])} "
        f"adamw_train_state={format_bytes(stats['adamw_training_bytes'])} "
        "(excludes activations, temporary tensors, and dataloader memory)"
    )
    print(_cuda_memory_text(torch, device))


def _print_first_batch_summary(condition: Any, target: Any, mask: Any, batch: dict[str, Any]) -> None:
    batch_bytes = _tensor_bytes(condition) + _tensor_bytes(target) + _tensor_bytes(mask)
    original_shapes = batch.get("original_shape", [])
    preview = ", ".join(f"{height}x{width}" for height, width in original_shapes[:4])
    if len(original_shapes) > 4:
        preview += ", ..."
    print(
        "first_batch="
        f"shape={tuple(condition.shape)} tensor_memory={format_bytes(batch_bytes)} "
        f"original_shapes=[{preview}]"
    )


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _format_training_timing(data_seconds: float, step_seconds: float, batch_size: int) -> dict[str, str]:
    samples_per_second = batch_size / max(step_seconds, 1e-12)
    return {
        "读盘秒": f"{data_seconds:.3f}",
        "训练秒": f"{step_seconds:.3f}",
        "样本每秒": f"{samples_per_second:.2f}",
    }


def _run_validation_preview(torch: Any, model: Any, dataset: Any, device: str, config: dict[str, Any], epoch: int) -> None:
    validation_cfg = config.get("validation", {})
    every = int(validation_cfg.get("every", 0) or 0)
    if every <= 0 or epoch % every != 0:
        return

    indices = _validation_indices(len(dataset), int(validation_cfg.get("num_slices", 4)))
    if not indices:
        return

    output_dir = Path(validation_cfg.get("output_dir", "runs/validation"))
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for preview_number, dataset_index in enumerate(indices, start=1):
            sample = dataset[dataset_index]
            batch = pad_slice_batch([sample])
            condition = batch["condition"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                prediction = model.sample(condition) if hasattr(model, "sample") else model(condition)

            case_id = str(batch["case_id"][0])
            slice_index = int(batch["slice_index"][0])
            output_path = output_dir / f"epoch_{epoch:04d}_{preview_number:02d}_{case_id}_slice_{slice_index:04d}.png"
            _save_validation_preview_png(
                _as_preview_array(condition),
                _as_preview_array(prediction),
                _as_preview_array(target),
                _as_preview_array(mask),
                output_path,
            )
            print(f"validation_preview={output_path}")
    finally:
        if was_training:
            model.train()


def _validation_indices(dataset_size: int, num_slices: int) -> list[int]:
    if dataset_size <= 0 or num_slices <= 0:
        return []
    count = min(dataset_size, num_slices)
    return [int(value) for value in np.linspace(0, dataset_size - 1, count, dtype=int)]


def _as_preview_array(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return np.asarray(array).squeeze().astype(np.float32)


def _save_validation_preview_png(
    condition: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("CBCT", condition, -1.0, 1.0),
        ("Pred CT", prediction, -1.0, 1.0),
        ("Target CT", target, -1.0, 1.0),
        ("Mask", mask, 0.0, 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(8.0, 2.4), dpi=150)
    for axis, (title, image, vmin, vmax) in zip(axes, panels):
        axis.imshow(np.asarray(image), cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(output)
    plt.close(fig)


def _progress_bar(iterable: Iterable[Any], **kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _PlainProgress(iterable)
    return tqdm(iterable, **kwargs)


class _PlainProgress:
    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs: Any) -> None:
        return None


def _cuda_progress_text(torch: Any, device: str) -> str:
    info = _cuda_memory_info(torch, device)
    if info is None:
        return "off"
    return f"{format_bytes(info['allocated'])}/{format_bytes(info['reserved'])}"


def _cuda_memory_text(torch: Any, device: str) -> str:
    info = _cuda_memory_info(torch, device)
    if info is None:
        return "cuda=unavailable_or_disabled"
    free_total = ""
    if info["free"] is not None and info["total"] is not None:
        free_total = f" free={format_bytes(info['free'])} total={format_bytes(info['total'])}"
    return (
        "cuda_memory="
        f"allocated={format_bytes(info['allocated'])} reserved={format_bytes(info['reserved'])} "
        f"peak_allocated={format_bytes(info['peak_allocated'])}{free_total}"
    )


def _cuda_memory_info(torch: Any, device: str) -> dict[str, int | None] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    cuda_device = torch.device(device)
    free = None
    total = None
    try:
        free, total = torch.cuda.mem_get_info(cuda_device)
    except RuntimeError:
        pass
    return {
        "allocated": int(torch.cuda.memory_allocated(cuda_device)),
        "reserved": int(torch.cuda.memory_reserved(cuda_device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(cuda_device)),
        "free": int(free) if free is not None else None,
        "total": int(total) if total is not None else None,
    }


if __name__ == "__main__":
    main()
