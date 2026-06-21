import unittest

from cbct2ct.data.samplers import CaseBlockSampler
from cbct2ct.data.torch_dataset import SliceRecord


class CaseBlockSamplerTests(unittest.TestCase):
    def test_blocks_keep_indices_from_one_case_and_cover_every_item_once(self):
        records = [SliceRecord(0, index) for index in range(5)] + [SliceRecord(1, index) for index in range(5)]
        sampler = CaseBlockSampler(records, block_size=3, seed=42)

        blocks = sampler.blocks_for_epoch(0)

        self.assertEqual(sorted(index for block in blocks for index in block), list(range(len(records))))
        self.assertTrue(all(len(block) <= 3 for block in blocks))
        self.assertTrue(all(len({records[index].case_index for index in block}) == 1 for block in blocks))

    def test_same_seed_and_epoch_produce_the_same_blocks(self):
        records = [SliceRecord(case, index) for case in range(3) for index in range(4)]

        first = CaseBlockSampler(records, block_size=2, seed=7).blocks_for_epoch(2)
        second = CaseBlockSampler(records, block_size=2, seed=7).blocks_for_epoch(2)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
