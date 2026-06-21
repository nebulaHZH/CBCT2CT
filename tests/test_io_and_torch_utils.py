import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cbct2ct.io import read_volume, write_volume
from cbct2ct.torch_utils import require_torch


class IoAndTorchUtilityTests(unittest.TestCase):
    def test_nifti_round_trip_preserves_array_and_affine(self):
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 11.0],
                [0.0, 0.0, 3.0, 12.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.nii.gz"
            write_volume(path, array, affine=affine)

            volume = read_volume(path)

        np.testing.assert_allclose(volume.array, array)
        np.testing.assert_allclose(volume.affine, affine)
        self.assertEqual(volume.path.name, "image.nii.gz")

    def test_require_torch_raises_actionable_error_when_torch_is_missing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named torch")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "pip install.*torch"):
                require_torch()


if __name__ == "__main__":
    unittest.main()
