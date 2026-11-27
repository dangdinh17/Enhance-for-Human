"""Check bundle attachment and equivalence against actual dataset frames."""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import torch

from src.data.training_bundle import bootstrap, bundle_file, verify_bundle_inputs
from src.data.window_mixed import SharedWindowDataset, close_window_environments
from src.utils.experiment import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]


class TrainingBundleTests(unittest.TestCase):
    """Read real bundle data; no synthetic dataset or model replacements."""

    def test_flat_path_rejects_traversal(self) -> None:
        for name in ("../data.mdb", "/tmp/data.mdb", "..", ""):
            with self.assertRaises(ValueError):
                bundle_file(ROOT, name)

    def test_configured_policy_and_complete_validation(self) -> None:
        for qp in (22, 27, 32, 37):
            config = load_yaml_mapping(ROOT / f"configs/kaggle/QP{qp}.yml")
            self.assertEqual(config["seed"], 7 + qp)
            self.assertEqual(config["train"]["iterations"], 300000)
            self.assertNotIn("validation_max_samples", config["train"])
            self.assertEqual(config["train"]["checkpoint_selection"],
                             {"metric": "enhanced_psnr", "mode": "max"})
            self.assertEqual(config["dataset"]["bundle_receipt"], "data/bundle.json")
            self.assertTrue(config["comet_logging"]["using"])
            self.assertEqual(config["comet_logging"]["project_name"], "qe-for-human")
            self.assertEqual(config["comet_logging"]["workspace"], "dangdinh17")
            self.assertEqual(config["comet_logging"]["api_key_env"], "COMET_API_KEY")
            self.assertNotIn("api_key", config["comet_logging"])
            self.assertEqual(config["run_dir"],
                             f"runs/research/newqe_kaggle_comet_charb_300k_v1/QP{qp}")

    def test_verified_attachment_is_idempotent_and_samples_match(self) -> None:
        path = ROOT / "data/bundle.json"
        if not path.is_file():
            self.skipTest("Attach the real dataset bundle first")
        cache_before = (ROOT / "data/bundle_verified.json").read_bytes()
        identity = json.loads(cache_before)["databases"]["gt"]["identity"]
        bootstrap(Path(identity["path"]).parent, ROOT)
        self.assertEqual(cache_before, (ROOT / "data/bundle_verified.json").read_bytes())
        receipt = json.loads(path.read_text())
        try:
            for qp in (22, 27, 32, 37):
                config = load_yaml_mapping(ROOT / f"configs/kaggle/QP{qp}.yml")
                verify_bundle_inputs(ROOT, config)
                data = SharedWindowDataset(ROOT / f"data/manifests/QP{qp}.json",
                                           "validation", 128, False)
                sample = data[0]
                self.assertEqual(tuple(sample["lq"].shape), (7, 1, 128, 128))
                self.assertTrue(torch.isfinite(sample["lq"]).all())
                self.assertEqual(receipt["databases"][f"lq_QP{qp}"]["entries"], 38389)
                random_state = random.getstate()
                first = data[0]
                second = data[0]
                self.assertEqual(random_state, random.getstate())
                torch.testing.assert_close(first["lq"], second["lq"], rtol=0, atol=0)
        finally:
            close_window_environments()


if __name__ == "__main__":
    unittest.main()
