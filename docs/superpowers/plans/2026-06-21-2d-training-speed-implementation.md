# 2D Training Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up 2D CBCT-to-CT DDPM training on an RTX 4060 8GB through AMP and case-blocked data ordering without changing the single-slice task.

**Architecture:** Add a small sampler that keeps a short random burst of slices from the same case together, allowing the existing LRU whole-volume cache to be reused. Extend the training loop with CUDA-only AMP, periodic synchronized timing, and Chinese progress messages. Use a separate no-attention experiment configuration so existing checkpoints remain untouched.

**Tech Stack:** Python 3, PyTorch, unittest, YAML.

---

### Task 1: Add Case-Block Sampling

**Files:**
- Create: `cbct2ct/data/samplers.py`
- Modify: `train.py:1-50`
- Create: `tests/test_case_block_sampler.py`

- [x] **Step 1: Write failing sampler tests**

```python
from cbct2ct.data.samplers import CaseBlockSampler
from cbct2ct.data.torch_dataset import SliceRecord


def test_blocks_keep_indices_from_one_case_and_cover_every_item_once():
    records = [SliceRecord(0, i) for i in range(5)] + [SliceRecord(1, i) for i in range(5)]
    sampler = CaseBlockSampler(records, block_size=3, seed=42)

    blocks = sampler.blocks_for_epoch(0)

    assert sorted(index for block in blocks for index in block) == list(range(len(records)))
    assert all(len(block) <= 3 for block in blocks)
    assert all({records[index].case_index for index in block}.__len__() == 1 for block in blocks)


def test_same_seed_and_epoch_produce_the_same_blocks():
    records = [SliceRecord(case, index) for case in range(3) for index in range(4)]

    assert CaseBlockSampler(records, block_size=2, seed=7).blocks_for_epoch(2) == (
        CaseBlockSampler(records, block_size=2, seed=7).blocks_for_epoch(2)
    )
```

- [x] **Step 2: Run the sampler tests and confirm they fail because the module is absent**

Run: `python -m unittest tests.test_case_block_sampler`

Expected: import failure for `cbct2ct.data.samplers`.

- [x] **Step 3: Implement the sampler**

```python
class CaseBlockSampler(Sampler[int]):
    def __init__(self, records: Sequence[SliceRecord], block_size: int, seed: int) -> None:
        if block_size <= 0:
            raise ValueError("病例切片块大小必须大于 0")
        self.records = tuple(records)
        self.block_size = block_size
        self.seed = seed
        self._epoch = 0

    def blocks_for_epoch(self, epoch: int) -> list[list[int]]:
        by_case: dict[int, list[int]] = defaultdict(list)
        for dataset_index, record in enumerate(self.records):
            by_case[record.case_index].append(dataset_index)
        rng = random.Random(self.seed + epoch)
        blocks: list[list[int]] = []
        for indices in by_case.values():
            rng.shuffle(indices)
            blocks.extend([indices[start : start + self.block_size] for start in range(0, len(indices), self.block_size)])
        rng.shuffle(blocks)
        return blocks

    def __iter__(self):
        blocks = self.blocks_for_epoch(self._epoch)
        self._epoch += 1
        for block in blocks:
            yield from block

    def __len__(self) -> int:
        return len(self.records)
```

In `train.py`, construct this sampler only when `data.case_block_size` is positive, pass it to `DataLoader`, and use `shuffle=False` in that case. Otherwise retain `shuffle=True`.

- [x] **Step 4: Run the sampler tests and confirm they pass**

Run: `python -m unittest tests.test_case_block_sampler`

Expected: `Ran 2 tests ... OK`.

### Task 2: Add CUDA AMP Controls

**Files:**
- Modify: `cbct2ct/training.py:1-45`
- Modify: `train.py:1-115`
- Create: `tests/test_mixed_precision.py`

- [x] **Step 1: Write failing AMP-selection tests**

```python
from cbct2ct.training import should_use_mixed_precision


def test_mixed_precision_requires_cuda_and_explicit_enablement():
    assert should_use_mixed_precision(device="cuda", requested=True) is True
    assert should_use_mixed_precision(device="cpu", requested=True) is False
    assert should_use_mixed_precision(device="cuda", requested=False) is False
```

- [x] **Step 2: Run the AMP-selection test and confirm it fails because the function is absent**

Run: `python -m unittest tests.test_mixed_precision`

Expected: import failure for `should_use_mixed_precision`.

- [x] **Step 3: Implement CUDA-only AMP in the training loop**

```python
def should_use_mixed_precision(device: str, requested: bool) -> bool:
    return requested and device.startswith("cuda")
```

```python
use_amp = should_use_mixed_precision(device, bool(training_cfg.get("mixed_precision", False)))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    loss = model.p_losses(target, condition, mask=mask)
if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

Keep the existing non-DDPM loss branch inside the autocast scope. All new comments and exceptions must be Chinese.

- [x] **Step 4: Run the AMP-selection test and confirm it passes**

Run: `python -m unittest tests.test_mixed_precision`

Expected: `Ran 1 test ... OK`.

### Task 3: Add Chinese Performance Metrics

**Files:**
- Modify: `train.py:60-110`
- Modify: `tests/test_training_stats.py`

- [x] **Step 1: Write a failing test for Chinese timing output formatting**

```python
from train import _format_training_timing


def test_timing_format_reports_data_step_and_throughput_in_chinese():
    text = _format_training_timing(data_seconds=0.125, step_seconds=0.25, batch_size=1)

    assert text == {"读盘秒": "0.125", "训练秒": "0.250", "样本每秒": "4.00"}
```

- [x] **Step 2: Run the timing test and confirm it fails because the formatter is absent**

Run: `python -m unittest tests.test_training_stats`

Expected: import failure for `_format_training_timing`.

- [x] **Step 3: Implement periodic synchronized timing**

```python
def _format_training_timing(data_seconds: float, step_seconds: float, batch_size: int) -> dict[str, str]:
    samples_per_second = batch_size / max(step_seconds, 1e-12)
    return {"读盘秒": f"{data_seconds:.3f}", "训练秒": f"{step_seconds:.3f}", "样本每秒": f"{samples_per_second:.2f}"}
```

Iterate over the loader explicitly with `next(iterator)` so elapsed time before the batch is the reading time. On the first step and every `training.timing_every` steps, synchronize CUDA immediately before and after the optimization step, then add the formatter result to `pbar.set_postfix`. Use `timing_every=50` in the new configuration; on other steps retain the most recent timing values without synchronizing.

- [x] **Step 4: Run timing tests and existing training-stat tests**

Run: `python -m unittest tests.test_training_stats`

Expected: all tests pass.

### Task 4: Create the RTX 4060 AMP/Cached Configuration

**Files:**
- Create: `configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml`
- Modify: `tests/test_official_cddpm_baseline.py`

- [x] **Step 1: Write a failing configuration test**

```python
from cbct2ct.config import load_config


def test_amp_cached_rtx4060_config_has_the_speed_controls():
    config = load_config("configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml")

    assert config["model"]["attention_levels"] == []
    assert config["training"]["mixed_precision"] is True
    assert config["data"]["case_block_size"] == 8
```

- [x] **Step 2: Run the configuration test and confirm it fails because the configuration is absent**

Run: `python -m unittest tests.test_official_cddpm_baseline`

Expected: file-not-found failure for the new YAML file.

- [x] **Step 3: Create the new configuration**

```yaml
experiment:
  name: cddpm_synthrad2023_brain_fast_rtx4060_amp_cached
data:
  max_slices_per_case: 64
  case_block_size: 8
model:
  name: official_cddpm
  base_channels: 64
  channel_mults: [1, 2, 3, 4]
  attention_levels: []
training:
  device: cuda
  epochs: 50
  batch_size: 1
  num_workers: 0
  mixed_precision: true
  timing_every: 50
```

Copy the existing fast configuration's unchanged normalization, optimizer, DDPM schedule, checkpoint cadence, validation cadence, and inference device into this file. Give checkpoints and validation previews the new experiment directory.

- [x] **Step 4: Run the configuration test and confirm it passes**

Run: `python -m unittest tests.test_official_cddpm_baseline`

Expected: all tests pass.

### Task 5: Verify the End-to-End Training Entry Point

**Files:**
- Modify: no production files

- [x] **Step 1: Run the complete test suite**

Run: `python -m unittest discover -s tests`

Expected: all tests pass.

- [x] **Step 2: Run a CPU startup smoke test**

Run: `python train.py --config configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml --override training.device=cpu --override training.epochs=0`

Expected: Chinese summary reports `混合精度=False`, and startup succeeds without CUDA.

- [x] **Step 3: Run one CUDA epoch with one slice and inspect the timing fields**

Run: `python train.py --config configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml --override training.epochs=1 --override data.max_slices_per_case=1 --override training.timing_every=1`

Expected: progress output includes `读盘秒`, `训练秒`, and `样本每秒`; CUDA peak allocation remains below 8 GiB.

- [x] **Step 4: Check the working tree without committing user files**

Run: `git status --short`

Expected: report the modified and new files. Do not create a commit because this repository has no initial commit and the user did not request history initialization.
