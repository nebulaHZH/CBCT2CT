from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cbct2ct.data.manifest import CaseRecord, read_manifest
from cbct2ct.data.preprocessing import clip_and_normalize_hu
from cbct2ct.io import read_volume
from cbct2ct.torch_utils import require_torch


@dataclass(frozen=True)
class SliceRecord:
    case_index: int
    slice_index: int


class SynthRADSliceDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        hu_min: float = -1000.0,
        hu_max: float = 2000.0,
        axis: int = 2,
        min_mask_pixels: int = 32,
        max_slices_per_case: int | None = None,
    ) -> None:
        torch = require_torch()
        self._dataset_base = torch.utils.data.Dataset
        self.records = read_manifest(manifest_path)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.axis = axis
        self.min_mask_pixels = min_mask_pixels
        self.index = self._build_index(max_slices_per_case=max_slices_per_case)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        torch = require_torch()
        slice_record = self.index[item]
        case = self.records[slice_record.case_index]
        cbct, ct, mask = _load_case_arrays(case.cbct_path, case.ct_path, case.mask_path)
        cbct_slice = _take_slice(cbct, self.axis, slice_record.slice_index)
        ct_slice = _take_slice(ct, self.axis, slice_record.slice_index)
        mask_slice = (_take_slice(mask, self.axis, slice_record.slice_index) > 0).astype(np.float32)

        cbct_norm = clip_and_normalize_hu(cbct_slice, self.hu_min, self.hu_max)
        ct_norm = clip_and_normalize_hu(ct_slice, self.hu_min, self.hu_max)
        return {
            "condition": torch.from_numpy(cbct_norm[None, ...]),
            "target": torch.from_numpy(ct_norm[None, ...]),
            "mask": torch.from_numpy(mask_slice[None, ...]),
            "case_id": case.case_id,
            "slice_index": slice_record.slice_index,
            "anatomy": case.anatomy,
        }

    def _build_index(self, max_slices_per_case: int | None) -> list[SliceRecord]:
        index: list[SliceRecord] = []
        for case_index, case in enumerate(self.records):
            mask = read_volume(case.mask_path).array > 0
            selected = []
            for slice_index in range(mask.shape[self.axis]):
                if int(_take_slice(mask, self.axis, slice_index).sum()) >= self.min_mask_pixels:
                    selected.append(SliceRecord(case_index=case_index, slice_index=slice_index))
            if max_slices_per_case is not None:
                selected = selected[:max_slices_per_case]
            index.extend(selected)
        if not index:
            raise ValueError(f"manifest 中没有找到有效切片：共 {len(self.records)} 个病例")
        return index


def pad_slice_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-sized 2D slices so PyTorch can stack them into one batch."""
    if not batch:
        raise ValueError("batch must contain at least one sample")

    torch = require_torch()
    max_height = max(int(sample["condition"].shape[-2]) for sample in batch)
    max_width = max(int(sample["condition"].shape[-1]) for sample in batch)

    collated: dict[str, Any] = {
        "condition": _pad_and_stack_tensors(torch, batch, "condition", max_height, max_width, fill_value=-1.0),
        "target": _pad_and_stack_tensors(torch, batch, "target", max_height, max_width, fill_value=-1.0),
        "mask": _pad_and_stack_tensors(torch, batch, "mask", max_height, max_width, fill_value=0.0),
        "case_id": [sample["case_id"] for sample in batch],
        "slice_index": [sample["slice_index"] for sample in batch],
        "anatomy": [sample["anatomy"] for sample in batch],
        "original_shape": [
            (int(sample["condition"].shape[-2]), int(sample["condition"].shape[-1])) for sample in batch
        ],
    }
    return collated


def _pad_and_stack_tensors(
    torch: Any,
    batch: list[dict[str, Any]],
    key: str,
    max_height: int,
    max_width: int,
    fill_value: float,
) -> Any:
    padded = []
    for sample in batch:
        tensor = sample[key]
        if tensor.ndim != 3:
            raise ValueError(f"{key} must have shape [C, H, W], got {tuple(tensor.shape)}")
        channels, height, width = tensor.shape
        output = torch.full(
            (channels, max_height, max_width),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        output[:, :height, :width] = tensor
        padded.append(output)
    return torch.stack(padded, dim=0)


@lru_cache(maxsize=8)
def _load_case_arrays(cbct_path: str, ct_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cbct = read_volume(cbct_path).array
    ct = read_volume(ct_path).array
    mask = read_volume(mask_path).array
    if cbct.shape != ct.shape or cbct.shape != mask.shape:
        raise ValueError(f"病例中的 CBCT、CT 和 mask 体数据形状必须一致，当前为 {cbct.shape}, {ct.shape}, {mask.shape}")
    return cbct, ct, mask


def _take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    slicer = [slice(None)] * array.ndim
    slicer[axis] = index
    return np.asarray(array[tuple(slicer)], dtype=np.float32)
