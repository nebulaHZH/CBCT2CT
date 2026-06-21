# 2D Training Speed Design

## Goal

Make the existing 2D CBCT-to-CT DDPM practical on an RTX 4060 8GB laptop GPU without changing the single-slice training target.

All new source-code comments and user-facing training messages use Chinese.

## Scope

- Create a separate speed-oriented experiment configuration.
- Remove global attention from that experiment to avoid the observed 18 GiB peak allocation of the original model.
- Enable CUDA AMP with gradient scaling and retain an FP32 CPU fallback.
- Replace fully random slice ordering with seeded case blocks of eight slices. Each epoch must still include every slice exactly once.
- Report data loading time, GPU training-step time, and samples per second.

## Data Flow

`CaseBlockSampler` groups dataset indices by case, shuffles the slices inside each case, splits each case into blocks of eight, then shuffles the blocks. The DataLoader receives this sampler with `shuffle=False`. Consecutive samples in one block reuse the dataset's existing whole-volume LRU cache while cases remain mixed across blocks.

## AMP Behavior

When `training.mixed_precision` is true and the selected device is CUDA, the forward/loss pass runs under FP16 autocast and optimizer updates use a GradScaler. On CPU or when disabled, training remains FP32 with the current optimizer behavior.

## Configuration

The new configuration uses `base_channels: 64`, `batch_size: 1`, `attention_levels: []`, `mixed_precision: true`, and `case_block_size: 8`. It has dedicated experiment, checkpoint, and validation paths. Removing attention changes the state-dict structure, so it starts a new experiment rather than resuming any old checkpoint.

## Verification

- Unit tests prove that the sampler is seeded, contains every index once, and keeps each block within one case.
- Unit tests cover AMP enablement on CUDA and safe disablement on CPU.
- A CPU smoke run validates configuration loading and zero-epoch startup.
- A GPU one-step timing confirms memory stays within the 8GB dedicated-GPU budget and reports the new timing fields.
