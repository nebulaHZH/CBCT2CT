from __future__ import annotations

from typing import Mapping

import numpy as np


def clip_and_normalize_hu(
    image: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 2000.0,
) -> np.ndarray:
    clipped = np.clip(image.astype(np.float32), hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def denormalize_hu(
    image: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 2000.0,
) -> np.ndarray:
    return (((image.astype(np.float32) + 1.0) * 0.5) * (hu_max - hu_min) + hu_min).astype(
        np.float32
    )


def crop_to_mask(
    volumes: Mapping[str, np.ndarray],
    margin: int = 8,
    mask_key: str = "mask",
) -> dict[str, np.ndarray | tuple[tuple[int, int], ...]]:
    mask = np.asarray(volumes[mask_key]) > 0
    if not mask.any():
        bbox = tuple((0, size) for size in mask.shape)
        return {**{key: np.asarray(value).copy() for key, value in volumes.items()}, "bbox": bbox}

    coords = np.argwhere(mask)
    starts = np.maximum(coords.min(axis=0) - margin, 0)
    stops = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    bbox = tuple((int(start), int(stop)) for start, stop in zip(starts, stops))

    cropped: dict[str, np.ndarray | tuple[tuple[int, int], ...]] = {
        key: np.asarray(value)[slices].copy() for key, value in volumes.items()
    }
    cropped["bbox"] = bbox
    return cropped


def extract_masked_slices(
    cbct: np.ndarray,
    ct: np.ndarray,
    mask: np.ndarray,
    axis: int = 2,
    min_mask_pixels: int = 32,
) -> list[dict[str, np.ndarray | int]]:
    if cbct.shape != ct.shape or cbct.shape != mask.shape:
        raise ValueError(f"CBCT、CT 和 mask 的形状必须一致，当前为 {cbct.shape}, {ct.shape}, {mask.shape}")

    records: list[dict[str, np.ndarray | int]] = []
    for index in range(cbct.shape[axis]):
        slicer = [slice(None)] * cbct.ndim
        slicer[axis] = index
        key = tuple(slicer)
        mask_slice = (mask[key] > 0).astype(np.uint8)
        if int(mask_slice.sum()) < min_mask_pixels:
            continue
        records.append(
            {
                "index": index,
                "cbct": cbct[key].astype(np.float32),
                "ct": ct[key].astype(np.float32),
                "mask": mask_slice,
            }
        )
    return records
