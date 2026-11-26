"""Protect Human-only quality aggregation and result-scope boundaries."""
from __future__ import annotations

import unittest
from typing import Any

from scripts.evaluate_newqe_human_quality import METHOD, summarize


class HumanQualitySummaryTests(unittest.TestCase):
    """Check averaging across unequal sequence lengths and reject wrong scope."""

    @staticmethod
    def rows() -> list[dict[str, Any]]:
        return [
            {"method": METHOD, "scope": "full_video", "qp": 22,
             "sequence": "short", "frames": 1, "vvc_psnr_y": 29.0,
             "psnr_y": 30.0, "delta_psnr_y": 1.0,
             "vvc_ssim_y": .7, "ssim_y": .8},
            {"method": METHOD, "scope": "full_video", "qp": 22,
             "sequence": "long", "frames": 3, "vvc_psnr_y": 37.0,
             "psnr_y": 40.0, "delta_psnr_y": 3.0,
             "vvc_ssim_y": .9, "ssim_y": .9},
        ]

    def test_each_sequence_has_equal_weight(self) -> None:
        summary = summarize(self.rows())[0]
        self.assertEqual(summary["sequences"], 2)
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["psnr_y"], 35.0)
        self.assertEqual(summary["vvc_psnr_y"], 33.0)
        self.assertEqual(summary["delta_psnr_y"], 2.0)
        self.assertAlmostEqual(summary["ssim_y"], .85)
        self.assertAlmostEqual(summary["delta_ssim_y"], .05)

    def test_duplicate_sequence_is_rejected(self) -> None:
        rows = self.rows()
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])

    def test_other_models_and_smoke_are_rejected(self) -> None:
        for key, value in [("method", "NewQE_JDE_100k"), ("scope", "smoke")]:
            with self.subTest(key=key):
                rows = self.rows()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    summarize(rows)

    def test_nonfinite_quality_is_rejected(self) -> None:
        rows = self.rows()
        rows[0]["ssim_y"] = float("nan")
        with self.assertRaises(ValueError):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
