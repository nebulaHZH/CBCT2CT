import tempfile
import unittest
from pathlib import Path

import numpy as np

from train import _save_validation_preview_png, _validation_indices


class TrainingPreviewTests(unittest.TestCase):
    def test_validation_indices_are_evenly_spaced_and_bounded(self):
        self.assertEqual(_validation_indices(dataset_size=10, num_slices=4), [0, 3, 6, 9])
        self.assertEqual(_validation_indices(dataset_size=3, num_slices=10), [0, 1, 2])
        self.assertEqual(_validation_indices(dataset_size=0, num_slices=4), [])
        self.assertEqual(_validation_indices(dataset_size=10, num_slices=0), [])

    def test_save_validation_preview_png_writes_comparison_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "preview.png"
            condition = np.zeros((8, 8), dtype=np.float32)
            prediction = np.ones((8, 8), dtype=np.float32) * 0.25
            target = np.ones((8, 8), dtype=np.float32) * 0.5
            mask = np.ones((8, 8), dtype=np.float32)

            _save_validation_preview_png(condition, prediction, target, mask, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
