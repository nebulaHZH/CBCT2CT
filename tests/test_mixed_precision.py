import unittest

from cbct2ct.training import should_use_mixed_precision


class MixedPrecisionTests(unittest.TestCase):
    def test_mixed_precision_requires_cuda_and_explicit_enablement(self):
        self.assertTrue(should_use_mixed_precision(device="cuda", requested=True))
        self.assertFalse(should_use_mixed_precision(device="cpu", requested=True))
        self.assertFalse(should_use_mixed_precision(device="cuda", requested=False))


if __name__ == "__main__":
    unittest.main()
