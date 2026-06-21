import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from cbct2ct.config import load_config
from cbct2ct.data.manifest import (
    CaseRecord,
    discover_synthrad_cases,
    read_manifest,
    write_manifest,
)


class ConfigAndManifestTests(unittest.TestCase):
    def test_load_config_merges_dot_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment": {"name": "brain"},
                        "training": {"epochs": 10, "batch_size": 4},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(
                config_path,
                overrides={"training.epochs": "2", "experiment.name": "smoke"},
            )

        self.assertEqual(config["training"]["epochs"], 2)
        self.assertEqual(config["training"]["batch_size"], 4)
        self.assertEqual(config["experiment"]["name"], "smoke")

    def test_discover_synthrad_cases_finds_task2_layout_and_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "Task2" / "brain" / "2BA001"
            case_dir.mkdir(parents=True)
            image = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
            for name in ("cbct.nii.gz", "ct.nii.gz", "mask.nii.gz"):
                nib.save(image, case_dir / name)

            records = discover_synthrad_cases(root, anatomies=["brain"])
            manifest_path = root / "manifest.jsonl"
            write_manifest(records, manifest_path)
            restored = read_manifest(manifest_path)

        self.assertEqual(len(records), 1)
        self.assertIsInstance(restored[0], CaseRecord)
        self.assertEqual(restored[0].case_id, "2BA001")
        self.assertEqual(restored[0].anatomy, "brain")
        self.assertTrue(restored[0].cbct_path.endswith("cbct.nii.gz"))
        self.assertEqual(json.loads(restored[0].to_json())["case_id"], "2BA001")


if __name__ == "__main__":
    unittest.main()
