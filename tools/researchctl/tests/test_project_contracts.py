"""Read-only checks of the real project's configured control boundaries."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from researchctl.config import campaign_config, load_model_config, load_project
from researchctl.engine import node_context, plan_campaign

ROOT = Path(__file__).resolve().parents[3]


class ProjectContractsTest(unittest.TestCase):
    def test_production_matrices_and_evaluation_are_fully_connected(self) -> None:
        config = load_project(ROOT, "local_1gpu")
        for identifier, count in (
            ("stdr_loss_matrix", 24),
            ("mfqev2_jde_af", 8),
            ("newqe_perqp", 4),
            ("newqe_window_pilots", 3),
        ):
            with self.subTest(campaign=identifier):
                campaign, _, trials, nodes = plan_campaign(config, identifier)
                counts = Counter(node["purpose"] for node in nodes)
                self.assertEqual(len(trials), count)
                self.assertEqual(counts["main"], count)
                self.assertGreater(counts["evaluation"], 0)
                self.assertEqual(counts["aggregate"], 1)
                self.assertEqual(campaign["selection"]["mode"], "all")
                self.assertTrue(
                    all(
                        node["requires_approval"]
                        for node in nodes
                        if node["purpose"] == "main"
                    )
                )

    def test_main_newqe_and_jde_configs_preserve_legacy_checkpoint_identity(
        self,
    ) -> None:
        config = load_project(ROOT, "local_1gpu")
        for identifier in ("newqe_perqp", "newqe_window_pilots", "mfqev2_jde_af"):
            campaign, identity, trials, _ = plan_campaign(config, identifier)
            effective = campaign_config(config, campaign)
            for trial in trials:
                variant = next(
                    item for item in campaign["variants"] if item["id"] == trial.variant
                )
                expected = load_model_config(ROOT / variant["base_config"])
                actual, _ = node_context(effective, campaign, identity, "main", trial)
                self.assertEqual(actual, expected)

    def test_smoke_profile_has_no_tuning_main_or_full_evaluation_nodes(self) -> None:
        config = load_project(ROOT, "smoke")
        for identifier in ("stdr_loss_matrix", "newqe_perqp", "newqe_adapter_smoke"):
            _, _, _, nodes = plan_campaign(config, identifier)
            self.assertFalse(
                {node["purpose"] for node in nodes}
                & {"main", "tuning", "evaluation", "aggregate"}
            )
