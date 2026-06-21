import unittest

from cbct2ct.data.torch_dataset import pad_slice_batch
from cbct2ct.torch_utils import require_torch


class TorchDatasetBatchingTests(unittest.TestCase):
    def test_pad_slice_batch_stacks_variable_sized_slices(self):
        torch = require_torch()
        batch = [
            {
                "condition": torch.ones((1, 2, 3), dtype=torch.float32),
                "target": torch.full((1, 2, 3), 2.0, dtype=torch.float32),
                "mask": torch.ones((1, 2, 3), dtype=torch.float32),
                "case_id": "case-a",
                "slice_index": 1,
                "anatomy": "brain",
            },
            {
                "condition": torch.full((1, 3, 2), 3.0, dtype=torch.float32),
                "target": torch.full((1, 3, 2), 4.0, dtype=torch.float32),
                "mask": torch.ones((1, 3, 2), dtype=torch.float32),
                "case_id": "case-b",
                "slice_index": 2,
                "anatomy": "brain",
            },
        ]

        collated = pad_slice_batch(batch)

        self.assertEqual(collated["condition"].shape, (2, 1, 3, 3))
        self.assertEqual(collated["target"].shape, (2, 1, 3, 3))
        self.assertEqual(collated["mask"].shape, (2, 1, 3, 3))
        self.assertEqual(collated["case_id"], ["case-a", "case-b"])
        self.assertEqual(collated["slice_index"], [1, 2])
        self.assertEqual(collated["original_shape"], [(2, 3), (3, 2)])
        self.assertEqual(float(collated["condition"][0, 0, 1, 2]), 1.0)
        self.assertEqual(float(collated["condition"][0, 0, 2, 2]), -1.0)
        self.assertEqual(float(collated["target"][1, 0, 2, 1]), 4.0)
        self.assertEqual(float(collated["target"][1, 0, 2, 2]), -1.0)
        self.assertEqual(float(collated["mask"][0, 0, 2, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
