import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_prepare_script_help_runs_from_script_path(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_synthrad2023.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("创建 JSONL manifest", result.stdout)

    def test_identity_script_help_runs_from_script_path(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_identity.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("identity baseline 进行评估", result.stdout)


    def test_train_script_reports_resources_and_uses_tqdm_progress(self):
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("parameter_statistics", train_source)
        self.assertIn("tqdm", train_source)
        self.assertIn("pbar.set_postfix", train_source)
        self.assertIn("tqdm", requirements)


if __name__ == "__main__":
    unittest.main()
