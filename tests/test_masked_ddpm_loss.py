import unittest

import torch
from torch.nn import functional as F

from cbct2ct.models.official_cddpm import OfficialConditionalDDPM, OfficialConditionalUNet, masked_noise_mse


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
            timesteps=4,
            base_channels=32,
            channel_mults=(1,),
            attention_levels=(),
            num_res_blocks=1,
            dropout=0.0,
        )
        model = OfficialConditionalDDPM(denoiser=denoiser, timesteps=4, loss="masked_mse_mean")
        target = torch.zeros((1, 1, 8, 8))
        condition = torch.zeros_like(target)
        mask = torch.zeros_like(target)

        self.assertEqual(model.p_losses(target, condition, mask=mask).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
