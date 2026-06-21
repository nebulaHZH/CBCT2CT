from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cbct2ct.cli import ChineseArgumentParser
from cbct2ct.data.manifest import read_manifest
from cbct2ct.io import read_volume
from cbct2ct.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="评估生成的 sCT 体数据与真实 CT 的差异。")
    parser.add_argument("--manifest", required=True, help="参考 manifest JSONL 文件")
    parser.add_argument("--prediction-dir", required=True, help="包含 {case_id}_sct.nii.gz 的预测结果目录")
    parser.add_argument("--output-csv", required=True, help="逐病例指标 CSV 输出路径")
    parser.add_argument("--data-range", type=float, default=3000.0, help="计算 PSNR/SSIM 时使用的强度范围")
    parser.add_argument("--visual-dir", default=None, help="可选：保存 CBCT/sCT/CT/误差 PNG 面板的目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in read_manifest(args.manifest):
        pred_path = prediction_dir / f"{record.case_id}_sct.nii.gz"
        pred = read_volume(pred_path).array
        target = read_volume(record.ct_path).array
        cbct = read_volume(record.cbct_path).array
        mask = read_volume(record.mask_path).array
        metrics = compute_metrics(pred, target, mask=mask, data_range=args.data_range)
        rows.append({"case_id": record.case_id, "anatomy": record.anatomy, **metrics})
        if args.visual_dir:
            _write_visual_panel(Path(args.visual_dir), record.case_id, cbct, pred, target, mask)

    summary = _summarize(rows)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "anatomy", "mae", "psnr", "ncc", "ssim"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for key, value in summary.items():
        print(f"{key}={value:.6f}")


def _summarize(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("没有生成任何评估结果，请检查 manifest 和预测目录")
    metric_names = ["mae", "psnr", "ncc", "ssim"]
    return {
        f"{name}_mean": sum(float(row[name]) for row in rows) / len(rows)
        for name in metric_names
    }


def _write_visual_panel(output_dir: Path, case_id: str, cbct, pred, target, mask) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    slice_scores = mask.reshape(-1, mask.shape[-1]).sum(axis=0)
    index = int(slice_scores.argmax())
    images = [cbct[:, :, index], pred[:, :, index], target[:, :, index], pred[:, :, index] - target[:, :, index]]
    titles = ["CBCT", "sCT", "CT", "sCT - CT 误差"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        cmap = "coolwarm" if " - " in title else "gray"
        vmin, vmax = (-200, 200) if cmap == "coolwarm" else (-1000, 1000)
        axis.imshow(image.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(output_dir / f"{case_id}_panel.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
