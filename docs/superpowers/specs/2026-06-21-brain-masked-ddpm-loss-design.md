# Brain Masked DDPM Loss Design

## Goal

Train a brain CBCT-to-CT DDPM with supervision restricted to the provided brain mask while keeping the CBCT input unchanged at both training and inference time.

## Data Flow

The model continues to receive a complete normalized CBCT slice and a complete noisy CT slice. The mask is not concatenated to the model input and is not needed during inference. During training, the squared noise-prediction error is multiplied by the binary mask and normalized by the number of valid mask pixels.

## Loss

For predicted noise `epsilon_hat`, sampled noise `epsilon`, and binary mask `m`, use:

`sum((epsilon_hat - epsilon)^2 * m) / max(sum(m), 1)`

This gives each brain pixel equal weight and makes the loss scale stable across variable slice dimensions and brain areas. The mask has no effect when it contains all ones.

## Experiment Isolation

Create a `masked_loss` configuration derived from the RTX 4060 AMP/cached configuration. The model architecture remains unchanged, but the changed loss requires training from scratch and a separate checkpoint directory.

## Verification

- Unit tests prove all-one masks match ordinary mean MSE, masked pixels do not affect the loss, and empty masks remain finite.
- Unit tests prove `OfficialConditionalDDPM.p_losses` uses the supplied mask.
- Run the full suite, CPU zero-epoch startup, and a one-epoch CUDA smoke run with the new configuration.
