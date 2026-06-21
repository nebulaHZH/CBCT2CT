# Brain Masked DDPM Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a brain-mask-weighted DDPM noise loss while preserving complete CBCT input at training and inference.

**Architecture:** Keep `OfficialConditionalDDPM` and its two-channel denoiser input unchanged: noisy CT plus CBCT. Add a masked mean-squared-noise helper and select it through a new `masked_mse_mean` loss mode. Build a separate AMP/cached brain configuration that uses this mode and writes to independent run directories.

**Tech Stack:** Python 3, PyTorch, unittest, YAML.

---

### Task 1: Add Masked Noise-Loss Tests

**Files:**
- Create: `tests/test_masked_ddpm_loss.py`
- Modify: `cbct2ct/models/official_cddpm.py:1-280`

- [ ] **Step 1: Write failing tests for the desired loss behavior**

```python
import unittest

import torch
from torch.nn import functional as F

from cbct2ct.models.official_cddpm import (
    OfficialConditionalDDPM,
    OfficialConditionalUNet,
    masked_noise_mse,
)


class MaskedDDPMLossTests(unittest.TestCase):
    def test_all_one_mask_matches_mean_squared_error(self):
        predicted = torch.tensor([[[[1.0, 3.0]]]])
        noise = torch.tensor([[[[0.0, 1.0]]]])
        mask = torch.ones_like(predicted)

        self.assertTrue(torch.allclose(masked_noise_mse(predicted, noise, mask), F.mse_loss(predicted, noise)))

    def test_masked_pixels_do_not_contribute_to_loss(self):
        predicted = torch.tensor([[[[1.0, 100.0]]]])
        noise = torch.zeros_like(predicted)
        mask = torch.tensor([[[[1.0, 0.0]]]])

        self.assertEqual(masked_noise_mse(predicted, noise, mask).item(), 1.0)

    def test_empty_mask_has_finite_zero_loss(self):
        predicted = torch.ones((1, 1, 2, 2))
        noise = torch.zeros_like(predicted)
        mask = torch.zeros_like(predicted)

        self.assertEqual(masked_noise_mse(predicted, noise, mask).item(), 0.0)

    def test_masked_loss_mode_uses_the_supplied_mask(self):
        denoiser = OfficialConditionalUNet(
            timesteps=4, base_channels=32, channel_mults=(1,), attention_levels=(), num_res_blocks=1, dropout=0.0
        )
        model = OfficialConditionalDDPM(denoiser=denoiser, timesteps=4, loss="masked_mse_mean")
        target = torch.zeros((1, 1, 8, 8))
        condition = torch.zeros_like(target)
        mask = torch.zeros_like(target)

        self.assertEqual(model.p_losses(target, condition, mask=mask).item(), 0.0)
```

- [ ] **Step 2: Run the tests and confirm they fail because `masked_noise_mse` is absent**

Run: `python -m unittest tests.test_masked_ddpm_loss`

Expected: import failure for `masked_noise_mse`.

- [ ] **Step 3: Implement masked mean-squared-noise loss and loss-mode selection**

```python
def masked_noise_mse(predicted_noise: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (predicted_noise - noise).square()
    weights = mask.to(device=squared_error.device, dtype=squared_error.dtype)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
```

In `OfficialConditionalDDPM.p_losses`, select this helper only for `self.loss == "masked_mse_mean"`; raise `ValueError("masked_mse_mean 损失需要提供 mask")` if the caller omits a mask. Retain the existing `mse_mean` and `mse_sum` behavior unchanged.

- [ ] **Step 4: Run the masked-loss tests and confirm they pass**

Run: `python -m unittest tests.test_masked_ddpm_loss`

Expected: `Ran 4 tests ... OK`.

### Task 2: Add an Isolated Brain Masked-Loss Configuration

**Files:**
- Create: `configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss.yaml`
- Modify: `tests/test_official_cddpm_baseline.py`

- [ ] **Step 1: Write a failing configuration test**

```python
def test_masked_loss_rtx4060_config_uses_masked_noise_loss(self):
    config_path = ROOT / "configs" / "cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    self.assertEqual(config["model"]["loss"], "masked_mse_mean")
    self.assertTrue(config["training"]["mixed_precision"])
    self.assertEqual(config["model"]["attention_levels"], [])
```

- [ ] **Step 2: Run the configuration tests and confirm they fail because the YAML file is absent**

Run: `python -m unittest tests.test_official_cddpm_baseline`

Expected: file-not-found error for the masked-loss YAML file.

- [ ] **Step 3: Create the masked-loss configuration**

```yaml
experiment:
  name: cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss
model:
  name: official_cddpm
  base_channels: 64
  attention_levels: []
  loss: masked_mse_mean
training:
  device: cuda
  batch_size: 1
  mixed_precision: true
  timing_every: 50
```

Copy all other data normalization, mask-slice selection, DDPM schedule, optimizer, cache block, validation, and inference settings from `cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml`. Use `runs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss/` for checkpoints and previews.

- [ ] **Step 4: Run the configuration tests and confirm they pass**

Run: `python -m unittest tests.test_official_cddpm_baseline`

Expected: all tests pass.

### Task 3: Verify Training and Inference Compatibility

**Files:**
- Modify: no production files

- [ ] **Step 1: Run the complete suite**

Run: `python -m unittest discover -s tests`

Expected: all tests pass.

- [ ] **Step 2: Run a CPU zero-epoch startup**

Run: `python train.py --config configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss.yaml --override training.device=cpu --override training.epochs=0`

Expected: startup succeeds and reports the masked-loss experiment name with `混合精度=False`.

- [ ] **Step 3: Run a CUDA masked-loss smoke epoch**

Run: `python train.py --config configs/cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss.yaml --override training.epochs=1 --override data.max_slices_per_case=1 --override training.timing_every=50`

Expected: training completes without requiring a mask at inference; progress includes Chinese timing fields and the checkpoint is isolated to the new run directory.

- [ ] **Step 4: Check the working tree without creating a commit**

Run: `git status --short`

Expected: report modified and newly created files. Do not commit because the repository has no initial commit and the user did not request Git history initialization.
