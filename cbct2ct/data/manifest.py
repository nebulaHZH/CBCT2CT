from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    anatomy: str
    cbct_path: str
    ct_path: str
    mask_path: str
    split: str = "train"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "CaseRecord":
        return cls(**json.loads(line))


def discover_synthrad_cases(
    root: str | Path,
    anatomies: Sequence[str] = ("brain", "pelvis"),
    split: str = "train",
) -> list[CaseRecord]:
    root_path = Path(root)
    task_root = root_path / "Task2"
    if not task_root.exists():
        task_root = root_path

    records: list[CaseRecord] = []
    for anatomy in anatomies:
        anatomy_root = task_root / anatomy
        if not anatomy_root.exists():
            continue
        for case_dir in sorted(path for path in anatomy_root.iterdir() if path.is_dir()):
            cbct = case_dir / "cbct.nii.gz"
            ct = case_dir / "ct.nii.gz"
            mask = case_dir / "mask.nii.gz"
            if cbct.exists() and ct.exists() and mask.exists():
                records.append(
                    CaseRecord(
                        case_id=case_dir.name,
                        anatomy=anatomy,
                        cbct_path=str(cbct),
                        ct_path=str(ct),
                        mask_path=str(mask),
                        split=split,
                    )
                )
    return records


def write_manifest(records: Iterable[CaseRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def read_manifest(path: str | Path) -> list[CaseRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [CaseRecord.from_json(line) for line in handle if line.strip()]
