from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbct2ct.cli import ChineseArgumentParser
from cbct2ct.data.manifest import discover_synthrad_cases, write_manifest


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="为 SynthRAD2023 Task2 CBCT-to-CT 创建 JSONL manifest。")
    parser.add_argument("--root", required=True, help="已解压的 SynthRAD2023 Task2 根目录，或包含 Task2/ 的父目录")
    parser.add_argument("--output-dir", default="manifests/synthrad2023", help="写出 JSONL manifest 的目录")
    parser.add_argument(
        "--anatomies",
        nargs="+",
        default=["brain", "pelvis"],
        choices=["brain", "pelvis"],
        help="需要纳入的 Task2 解剖部位",
    )
    parser.add_argument("--split", default="train", help="写入 manifest 的数据划分标签")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for anatomy in args.anatomies:
        records = discover_synthrad_cases(args.root, anatomies=[anatomy], split=args.split)
        write_manifest(records, output_dir / f"task2_{anatomy}_{args.split}.jsonl")
        print(f"{anatomy}: {len(records)} 个病例")
        all_records.extend(records)
    if len(args.anatomies) > 1:
        write_manifest(all_records, output_dir / f"task2_{'_'.join(args.anatomies)}_{args.split}.jsonl")
        print(f"合并: {len(all_records)} 个病例")


if __name__ == "__main__":
    main()
