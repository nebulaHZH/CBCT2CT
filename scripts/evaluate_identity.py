from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbct2ct.cli import ChineseArgumentParser
from cbct2ct.data.manifest import read_manifest
from cbct2ct.io import read_volume
from cbct2ct.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="把原始 CBCT 作为 identity baseline 进行评估。")
    parser.add_argument("--manifest", required=True, help="参考 manifest JSONL 文件")
    parser.add_argument("--output-csv", required=True, help="逐病例指标 CSV 输出路径")
    parser.add_argument("--data-range", type=float, default=3000.0, help="计算 PSNR/SSIM 时使用的强度范围")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in read_manifest(args.manifest):
        cbct = read_volume(record.cbct_path).array
        ct = read_volume(record.ct_path).array
        mask = read_volume(record.mask_path).array
        rows.append({"case_id": record.case_id, "anatomy": record.anatomy, **compute_metrics(cbct, ct, mask, args.data_range)})

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "anatomy", "mae", "psnr", "ncc", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    for metric in ["mae", "psnr", "ncc", "ssim"]:
        print(f"{metric}_mean={sum(float(row[metric]) for row in rows) / max(len(rows), 1):.6f}")


if __name__ == "__main__":
    main()
