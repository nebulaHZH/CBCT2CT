import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class OfficialCDDPMBaselineTests(unittest.TestCase):
    def test_cddpm_configs_use_official_paper_baseline_defaults(self):
        for config_name in [
            "cddpm_synthrad2023_brain.yaml",
            "cddpm_synthrad2023_pelvis.yaml",
            "cddpm_synthrad2023_brain_pelvis.yaml",
        ]:
            with self.subTest(config=config_name):
                config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
                model = config["model"]
                self.assertEqual(model["name"], "official_cddpm")
                self.assertEqual(model["official_repo"], "https://github.com/junbopeng/conditional_DDPM")
                self.assertEqual(model["paper_doi"], "10.1002/mp.16704")
                self.assertEqual(model["base_channels"], 128)
                self.assertEqual(model["channel_mults"], [1, 2, 3, 4])
                self.assertEqual(model["attention_levels"], [2])
                self.assertEqual(model["num_res_blocks"], 2)
                self.assertAlmostEqual(model["dropout"], 0.3)
                self.assertEqual(model["loss"], "mse_sum")

    def test_training_builder_mentions_official_cddpm(self):
        source = (ROOT / "cbct2ct" / "training.py").read_text(encoding="utf-8")
        self.assertIn("official_cddpm", source)
        self.assertIn("OfficialConditionalDDPM", source)

    def test_amp_cached_rtx4060_config_has_speed_controls(self):
        config_path = ROOT / "configs" / "cddpm_synthrad2023_brain_fast_rtx4060_amp_cached.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["model"]["attention_levels"], [])
        self.assertTrue(config["training"]["mixed_precision"])
        self.assertEqual(config["data"]["case_block_size"], 8)

    def test_masked_loss_rtx4060_config_uses_masked_noise_loss(self):
        config_path = ROOT / "configs" / "cddpm_synthrad2023_brain_fast_rtx4060_amp_cached_masked_loss.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["model"]["loss"], "masked_mse_mean")
        self.assertTrue(config["training"]["mixed_precision"])
        self.assertEqual(config["model"]["attention_levels"], [])


if __name__ == "__main__":
    unittest.main()
