import unittest

import numpy as np

from cbct2ct.data.preprocessing import (
    clip_and_normalize_hu,
    crop_to_mask,
    denormalize_hu,
    extract_masked_slices,
)


class PreprocessingTests(unittest.TestCase):
    def test_clip_normalize_and_denormalize_hu_round_trip(self):
        hu = np.array([-2000.0, -1000.0, 0.0, 1000.0, 3000.0], dtype=np.float32)

        normalized = clip_and_normalize_hu(hu, hu_min=-1000.0, hu_max=1000.0)
        restored = denormalize_hu(normalized, hu_min=-1000.0, hu_max=1000.0)

        np.testing.assert_allclose(
            normalized,
            np.array([-1.0, -1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            restored,
            np.array([-1000.0, -1000.0, 0.0, 1000.0, 1000.0], dtype=np.float32),
        )

    def test_crop_to_mask_uses_margin_and_preserves_alignment(self):
        cbct = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
        ct = cbct + 1000.0
        mask = np.zeros_like(cbct, dtype=np.uint8)
        mask[2:4, 3:5, 4:6] = 1

        cropped = crop_to_mask({"cbct": cbct, "ct": ct, "mask": mask}, margin=1)

        self.assertEqual(cropped["cbct"].shape, (4, 4, 4))
        np.testing.assert_array_equal(cropped["ct"], cropped["cbct"] + 1000.0)
        self.assertEqual(cropped["bbox"], ((1, 5), (2, 6), (3, 7)))

    def test_extract_masked_slices_filters_empty_and_tiny_slices(self):
        cbct = np.zeros((4, 5, 3), dtype=np.float32)
        ct = np.ones((4, 5, 3), dtype=np.float32)
        mask = np.zeros((4, 5, 3), dtype=np.uint8)
        mask[:, :, 0] = 1
        mask[0, 0, 1] = 1

        slices = extract_masked_slices(cbct, ct, mask, axis=2, min_mask_pixels=4)

        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0]["index"], 0)
        self.assertEqual(slices[0]["cbct"].shape, (4, 5))
        self.assertEqual(slices[0]["ct"].shape, (4, 5))
        self.assertEqual(slices[0]["mask"].sum(), 20)


if __name__ == "__main__":
    unittest.main()
