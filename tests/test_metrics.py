import math
import unittest

import numpy as np

from cbct2ct.metrics import compute_metrics


class MetricTests(unittest.TestCase):
    def test_identical_pair_has_perfect_metrics_inside_mask(self):
        target = np.arange(64, dtype=np.float32).reshape(8, 8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        metrics = compute_metrics(target, target.copy(), mask=mask, data_range=100.0)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertTrue(math.isinf(metrics["psnr"]) or metrics["psnr"] > 90.0)
        self.assertAlmostEqual(metrics["ncc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=6)

    def test_metrics_use_only_masked_voxels_for_mae_ncc_and_psnr(self):
        target = np.zeros((4, 4), dtype=np.float32)
        pred = np.zeros((4, 4), dtype=np.float32)
        pred[0, 0] = 1000.0
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1

        metrics = compute_metrics(pred, target, mask=mask, data_range=2000.0)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertTrue(math.isinf(metrics["psnr"]))


if __name__ == "__main__":
    unittest.main()
