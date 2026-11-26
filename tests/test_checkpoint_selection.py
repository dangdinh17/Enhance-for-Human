"""Checkpoint selection regressions using recorded validation measurements."""

from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

from src.train.checkpoint_selection import CheckpointSelection
from src.utils.experiment import load_yaml_mapping


ROOT = Path(__file__).resolve().parents[1]


class CheckpointSelectionTests(unittest.TestCase):
    """Verify PSNR ranking, config inheritance and resume policy safety."""

    def test_all_local_and_kaggle_qp_configs_select_psnr(self) -> None:
        for folder in ("newqe", "kaggle"):
            for qp in (22, 27, 32, 37):
                with self.subTest(folder=folder, qp=qp):
                    config = load_yaml_mapping(ROOT / f"configs/{folder}/QP{qp}.yml")
                    self.assertEqual(CheckpointSelection.from_config(config),
                                     CheckpointSelection("enhanced_psnr", "max"))

    def test_psnr_wins_when_recorded_loss_ranks_differently(self) -> None:
        path = (ROOT / "runs/research/newqe_charb_300k_qp22_37_v1"
                / "QP37/validation.csv")
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = json.loads((ROOT / "tests/fixtures/qp37_recorded_validation.json").read_text())["rows"]
        selection = CheckpointSelection("enhanced_psnr", "max")
        best, best_row = selection.initial_value, None
        for row in rows:
            candidate = selection.score({"per_qp": {row["qp"]: row}})
            if selection.improves(candidate, best):
                best, best_row = candidate, row
        expected = max(rows, key=lambda row: float(row["enhanced_psnr"]))
        loss_winner = min(rows, key=lambda row: float(row["total_loss"]))
        self.assertEqual(best_row, expected)
        self.assertNotEqual(best_row["iteration"], loss_winner["iteration"])
        self.assertGreater(float(best_row["total_loss"]),
                           float(loss_winner["total_loss"]))
        self.assertFalse(selection.improves(best, best))
        self.assertFalse(selection.improves(float(loss_winner["enhanced_psnr"]), best))

    def test_legacy_config_keeps_loss_policy(self) -> None:
        selection = CheckpointSelection.from_config({"train": {}})
        self.assertEqual(selection, CheckpointSelection("total_loss", "min"))
        self.assertEqual(selection.initial_value, math.inf)

    def test_resume_metadata_roundtrip_and_reject_policy_changes(self) -> None:
        selection = CheckpointSelection("enhanced_psnr", "max")
        path = (ROOT / "runs/research/newqe_charb_300k_qp22_37_v1"
                / "QP37/validation.csv")
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        else:
            row = json.loads((ROOT / "tests/fixtures/qp37_recorded_validation.json").read_text())["rows"][0]
        best = selection.score({"per_qp": {"37": row}})
        saved = json.loads(json.dumps(selection.metadata([37], best), allow_nan=False))
        selection.check_resume(saved, [37])
        self.assertEqual(saved["value"], best)
        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            CheckpointSelection().check_resume(saved, [37])
        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            selection.check_resume(saved, [22])
        initial = selection.metadata([37], selection.initial_value)
        self.assertIsNone(initial["value"])
        json.dumps(initial, allow_nan=False)

    def test_invalid_validation_and_policy_are_rejected(self) -> None:
        selection = CheckpointSelection("enhanced_psnr", "max")
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                selection.score({"per_qp": {"37": {"enhanced_psnr": value}}})
            with self.assertRaises(ValueError):
                selection.improves(value, selection.initial_value)
        with self.assertRaises(ValueError):
            selection.score({"per_qp": {}})
        for metric, mode in (("enhanced_psnr", "min"), ("total_loss", "max"),
                             ("unknown", "max")):
            with self.assertRaises(ValueError):
                CheckpointSelection(metric, mode)


if __name__ == "__main__":
    unittest.main()
