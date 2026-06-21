import unittest

from train import _format_training_timing
from cbct2ct.torch_utils import require_torch
from cbct2ct.training import estimate_adamw_memory_bytes, format_bytes, format_int, parameter_statistics


class TrainingStatisticsTests(unittest.TestCase):
    def test_parameter_statistics_counts_total_and_trainable_parameters(self):
        torch = require_torch()
        layer = torch.nn.Linear(3, 2)
        layer.bias.requires_grad_(False)

        stats = parameter_statistics(layer, bytes_per_param=4)

        self.assertEqual(stats["total"], 8)
        self.assertEqual(stats["trainable"], 6)
        self.assertEqual(stats["frozen"], 2)
        self.assertEqual(stats["parameter_bytes"], 32)
        self.assertEqual(stats["adamw_training_bytes"], 104)

    def test_estimate_adamw_memory_includes_parameters_gradients_and_optimizer_states(self):
        self.assertEqual(
            estimate_adamw_memory_bytes(total_parameters=10, trainable_parameters=4, bytes_per_param=4),
            88,
        )

    def test_format_helpers_make_large_numbers_readable(self):
        self.assertEqual(format_int(1234567), "1,234,567")
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(1536), "1.50 KiB")
        self.assertEqual(format_bytes(2 * 1024 * 1024), "2.00 MiB")

    def test_timing_format_reports_data_step_and_throughput_in_chinese(self):
        timing = _format_training_timing(data_seconds=0.125, step_seconds=0.25, batch_size=1)

        self.assertEqual(timing, {"读盘秒": "0.125", "训练秒": "0.250", "样本每秒": "4.00"})


if __name__ == "__main__":
    unittest.main()
