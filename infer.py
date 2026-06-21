from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cbct2ct.cli import ChineseArgumentParser
from cbct2ct.config import load_config
from cbct2ct.data.manifest import read_manifest
from cbct2ct.data.preprocessing import clip_and_normalize_hu, denormalize_hu
from cbct2ct.io import read_volume, write_volume
from cbct2ct.torch_utils import require_torch
from cbct2ct.training import build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="对 manifest 中的病例执行 CBCT-to-CT 推理。")
    parser.add_argument("--config", required=True, help="用于构建模型的 YAML 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint .pt 文件路径")
    parser.add_argument("--manifest", required=True, help="输入 manifest JSONL 文件")
    parser.add_argument("--output-dir", required=True, help="生成 NIfTI 文件的输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch = require_torch()
    device = config.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    axis = int(data_cfg.get("axis", 2))
    hu_min = float(data_cfg.get("hu_min", -1000.0))
    hu_max = float(data_cfg.get("hu_max", 2000.0))

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in read_manifest(args.manifest):
        cbct_volume = read_volume(record.cbct_path)
        mask = read_volume(record.mask_path).array > 0
        prediction = cbct_volume.array.copy().astype(np.float32)
        for index in range(cbct_volume.array.shape[axis]):
            if _take_slice(mask, axis, index).sum() == 0:
                continue
            cbct_slice = clip_and_normalize_hu(_take_slice(cbct_volume.array, axis, index), hu_min, hu_max)
            condition = torch.from_numpy(cbct_slice[None, None]).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                if hasattr(model, "sample"):
                    pred_norm = model.sample(condition).cpu().numpy()[0, 0]
                else:
                    pred_norm = model(condition).cpu().numpy()[0, 0]
            _assign_slice(prediction, denormalize_hu(pred_norm, hu_min, hu_max), axis, index)
        write_volume(output_dir / f"{record.case_id}_sct.nii.gz", prediction, reference=cbct_volume)
        print(f"已写出 {record.case_id}_sct.nii.gz")


def _take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    slicer = [slice(None)] * array.ndim
    slicer[axis] = index
    return array[tuple(slicer)]


def _assign_slice(volume: np.ndarray, value: np.ndarray, axis: int, index: int) -> None:
    slicer = [slice(None)] * volume.ndim
    slicer[axis] = index
    volume[tuple(slicer)] = value


if __name__ == "__main__":
    main()
