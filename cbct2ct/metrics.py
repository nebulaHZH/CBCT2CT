from __future__ import annotations

import math

import numpy as np


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = 3000.0,
) -> dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred_arr.shape}, {target_arr.shape}")

    if mask is None:
        valid = np.ones(pred_arr.shape, dtype=bool)
    else:
        valid = np.asarray(mask) > 0
        if valid.shape != pred_arr.shape:
            raise ValueError(f"Mask shape must match images, got {valid.shape}, {pred_arr.shape}")

    if not valid.any():
        raise ValueError("Metric mask does not contain any valid pixels")

    pred_valid = pred_arr[valid]
    target_valid = target_arr[valid]
    diff = pred_valid - target_valid
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    psnr = math.inf if mse == 0.0 else float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))
    ncc = _normalized_cross_correlation(pred_valid, target_valid)
    ssim = _ssim_2d_or_flat(pred_arr, target_arr, valid, data_range)
    return {"mae": mae, "psnr": psnr, "ncc": ncc, "ssim": ssim}


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def _ssim_2d_or_flat(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, data_range: float) -> float:
    if np.array_equal(pred[mask], target[mask]):
        return 1.0
    x = pred[mask]
    y = target[mask]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    denom = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    if denom == 0.0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / denom)
