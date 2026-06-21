from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from cbct2ct.data.torch_dataset import SliceRecord


class CaseBlockSampler(Sampler[int]):
    """按病例短块排列切片索引，提升整卷数据缓存命中率。"""

    def __init__(self, records: Sequence[SliceRecord], block_size: int, seed: int) -> None:
        if block_size <= 0:
            raise ValueError("病例切片块大小必须大于 0")
        self.records = tuple(records)
        self.block_size = block_size
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        blocks = self.blocks_for_epoch(self._epoch)
        self._epoch += 1
        for block in blocks:
            yield from block

    def __len__(self) -> int:
        return len(self.records)

    def blocks_for_epoch(self, epoch: int) -> list[list[int]]:
        by_case: dict[int, list[int]] = defaultdict(list)
        for dataset_index, record in enumerate(self.records):
            by_case[record.case_index].append(dataset_index)

        rng = random.Random(self.seed + epoch)
        blocks: list[list[int]] = []
        for indices in by_case.values():
            rng.shuffle(indices)
            for start in range(0, len(indices), self.block_size):
                blocks.append(indices[start : start + self.block_size])
        rng.shuffle(blocks)
        return blocks
