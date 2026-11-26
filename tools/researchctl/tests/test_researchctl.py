"""Unit and integration tests for the portable research controller."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from researchctl.cli import main as cli_main
from researchctl.config import (
    ConfigError,
    canonical_hash,
    deep_merge,
    expand_trials,
    interpolate,
    load_project,
    redact,
)
from researchctl.journal import Journal
from researchctl.orchestrator import approve_campaign, open_journal, run_campaign
from researchctl.preflight import require_preflight, run_preflight


FIXTURE = Path(__file__).parent / "fixture_experiment.py"


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


class ConfigurationTest(unittest.TestCase):
    def test_merge_redaction_interpolation_and_trial_hash(self) -> None:
        self.assertEqual(
            deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}),
            {"a": {"b": 3, "c": 2}},
        )
        self.assertEqual(redact({"api_key": "secret"}), {"api_key": "<redacted>"})
        self.assertEqual(interpolate("x/{trial.id}", {"trial.id": "abc"}), "x/abc")
        with self.assertRaises(ConfigError):
            interpolate("{unknown}", {})
        campaign = {
            "campaign": {"seeds": [7, 17]},
            "variants": [{"id": "base", "base_config": "base.yaml"}],
            "matrix": {"train.lr": [0.1, 0.2]},
        }
        first = expand_trials(campaign, "campaign-hash")
        second = expand_trials(campaign, "campaign-hash")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first[0].trial_id, first[1].trial_id)
        self.assertNotEqual(canonical_hash({"seed": 7}), canonical_hash({"seed": 8}))

    def test_journal_state_and_hash_bound_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = Journal(Path(temporary) / "journal.sqlite")
            payload = {
                "node_id": "node",
                "campaign_id": "campaign",
                "campaign_hash": "hash-a",
                "trial_id": "trial",
                "stage": "tuning",
            }
            journal.ensure_node(payload)
            journal.transition("node", "RUNNING", pid=123, pgid=123)
            journal.transition("node", "COMPLETE", metrics={"loss": 1.0})
            journal.approve("campaign", "hash-a", "main", ["trial"], {}, "tester")
            self.assertIsNotNone(journal.approval("campaign", "hash-a", "main"))
            self.assertIsNone(journal.approval("campaign", "hash-b", "main"))
            journal.close()


class CampaignIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(FIXTURE, self.root / "scripts/experiment.py")
        write_yaml(
            self.root / "configs/base.yaml",
            {
                "paths": {"run_root": "runs/original"},
                "experiment": {"name": "base", "random_seed": 0},
                "train": {"lr": 0.1},
                "comet_logging": {"using": False, "api_key_env": "COMET_API_KEY"},
            },
        )
        common = {"schema_version": "1.0"}
        write_yaml(
            self.root / ".research/project.yaml",
            {
                **common,
                "project": {
                    "id": "integration",
                    "title": "Integration",
                    "domain": "ml",
                },
                "runtime": {"python": sys.executable, "environment": {}},
                "paths": {
                    "datasets": "data",
                    "runs": "runs/research",
                    "outputs": "outputs/research",
                    "temp": str(self.root / "tmp"),
                    "required": [],
                },
                "provenance": {"include": ["scripts/**/*.py", "configs/**/*.yaml"]},
                "comet": {"enabled": False, "api_key_env": "COMET_API_KEY"},
            },
        )
        task = {
            "executor": "host",
            "argv": [
                sys.executable,
                "scripts/experiment.py",
                "--config",
                "{trial.config}",
            ],
            "timeout_seconds": 30,
            "resources": {"gpu_count": 0},
            "outputs": {"required": ["{trial.model_run_dir}/DONE.json"]},
            "collectors": [
                {
                    "type": "done_json",
                    "path": "{trial.model_run_dir}/DONE.json",
                    "metrics": {"validation.loss": "checkpoint_selection.best_value"},
                }
            ],
        }
        write_yaml(
            self.root / ".research/tasks.yaml",
            {**common, "tasks": {"experiment": task}},
        )
        write_yaml(
            self.root / ".research/pipeline.yaml",
            {
                **common,
                "pipeline": {
                    "stages": [
                        {"id": "smoke", "kind": "trials", "task": "experiment"},
                        {"id": "tuning", "kind": "trials", "task": "experiment"},
                        {
                            "id": "main",
                            "kind": "trials",
                            "task": "experiment",
                            "requires_approval": True,
                        },
                    ]
                },
            },
        )
        write_yaml(
            self.root / ".research/metrics.yaml",
            {
                **common,
                "metric_contract": {
                    "primary": {"name": "validation.loss", "direction": "minimize"},
                    "secondary": [],
                    "validation": {},
                },
            },
        )
        write_yaml(
            self.root / ".research/resources.yaml",
            {
                **common,
                "resources": {
                    "max_parallel_nodes": 1,
                    "max_node_gb": 1,
                    "stop_before_disk_full_gb": 0,
                    "rlimits": {"max_processes": 64, "max_open_files": 256},
                },
            },
        )
        write_yaml(
            self.root / ".research/safety.yaml",
            {
                **common,
                "execution": {"shell_commands_allowed": False},
                "filesystem": {
                    "writable_roots": [
                        "runs/research",
                        "outputs/research",
                        str(self.root / "tmp"),
                    ],
                    "forbidden_write_paths": [".git"],
                },
            },
        )
        for profile in ("smoke", "local_1gpu", "production"):
            write_yaml(self.root / f".research/profiles/{profile}.yaml", common)
        write_yaml(
            self.root / ".research/campaigns/integration.yaml",
            {
                **common,
                "campaign": {
                    "id": "integration",
                    "title": "Integration",
                    "seeds": [7, 17],
                },
                "variants": [{"id": "base", "base_config": "configs/base.yaml"}],
                "matrix": {"train.lr": [0.1]},
                "allowed_change_scope": [
                    "train.lr",
                    "experiment.name",
                    "experiment.random_seed",
                    "paths.run_root",
                ],
                "bindings": {
                    "experiment.name": "experiment",
                    "paths.run_root": "{trial.output_dir}",
                },
                "dataset_manifests": [],
                "stages": {
                    "smoke": {"enabled": True, "max_trials": 1},
                    "tuning": {"enabled": True},
                    "main": {"enabled": True},
                },
                "selection": {"source_stage": "tuning", "max_trials": 1},
            },
        )
        self.config = load_project(self.root, "smoke")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_run_approval_resume_and_idempotence(self) -> None:
        self.assertEqual(run_campaign(self.config, "integration"), 2)
        selected = approve_campaign(self.config, "integration", "main")
        self.assertEqual(len(selected), 1)
        self.assertEqual(run_campaign(self.config, "integration"), 0)
        self.assertEqual(run_campaign(self.config, "integration"), 0)
        journal = open_journal(self.config)
        try:
            nodes = journal.nodes(campaign_id="integration")
        finally:
            journal.close()
        self.assertEqual(sum(row["state"] == "COMPLETE" for row in nodes), 4)
        self.assertEqual(
            sum(row["stage"] == "main" and row["state"] == "COMPLETE" for row in nodes),
            1,
        )

    def test_gpu_preflight_uses_cuda_and_warns_on_nvml_failure(self) -> None:
        self.config.tasks["tasks"]["experiment"]["resources"] = {
            "gpu_count": 1,
            "gpu": 0,
        }
        calls = []

        def cuda_works(argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[0] == sys.executable:
                return subprocess.CompletedProcess(
                    argv, 0, '{"device": "test GPU"}', ""
                )
            return subprocess.CompletedProcess(argv, 1, "", "NVML mismatch")

        with patch("researchctl.preflight.subprocess.run", side_effect=cuda_works):
            checks = require_preflight(self.config)
        self.assertTrue(next(check for check in checks if check.name == "gpu-cuda").ok)
        nvml = next(check for check in checks if check.name == "gpu-nvml")
        self.assertFalse(nvml.ok)
        self.assertFalse(nvml.required)
        self.assertEqual(calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "0")

        def cuda_fails(argv, **kwargs):
            if argv[0] == sys.executable:
                return subprocess.CompletedProcess(argv, 1, "", "CUDA unavailable")
            return subprocess.CompletedProcess(argv, 0, "0, 1000", "")

        with patch("researchctl.preflight.subprocess.run", side_effect=cuda_fails):
            checks = run_preflight(self.config)
            self.assertFalse(
                next(check for check in checks if check.name == "gpu-cuda").ok
            )
            with self.assertRaises(ConfigError):
                require_preflight(self.config)

    def test_cli_init_copies_portable_template(self) -> None:
        destination = self.root / "new-project"
        destination.mkdir()
        self.assertEqual(cli_main(["init", "--destination", str(destination)]), 0)
        self.assertTrue((destination / ".research/project.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
