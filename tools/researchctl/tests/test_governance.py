"""Real-process regression tests for campaign governance and host isolation."""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

import test_researchctl as support
from researchctl.cli import main as cli_main
from researchctl.config import (
    ConfigError,
    campaign_config,
    load_model_config,
    load_project,
)
from researchctl.engine import (
    approve_campaign,
    node_context,
    open_journal,
    plan_campaign,
    run_campaign,
    launch_campaign,
    reap_background,
)
from researchctl.executor import execute
from researchctl.journal import Journal
from researchctl.processes import start_ticks
from researchctl.processes import signal_node
from researchctl.contracts import check_outputs


class GovernanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = support.CampaignIntegrationTest()
        self.fixture.setUp()
        self.root = self.fixture.root
        shutil.copy2(
            Path(__file__).with_name("fixture_pipeline.py"),
            self.root / "scripts/pipeline.py",
        )
        support.write_yaml(
            self.root / "configs/base.yaml",
            {"seed": 0, "run_dir": "unused", "steps": 20},
        )
        task = {
            "executor": "host",
            "resources": {"gpu_count": 0},
            "timeout_seconds": 20,
            "argv": [
                sys.executable,
                "scripts/pipeline.py",
                "train",
                "--config",
                "{trial.config}",
                "--output",
                "{trial.output_dir}/task.json",
            ],
            "collectors": [
                {"type": "result_json", "path": "{trial.output_dir}/task.json"}
            ],
            "result_contract": {
                "path": "{trial.output_dir}/task.json",
                "scope": "training",
                "primary_metric": True,
            },
        }
        evaluation = json.loads(json.dumps(task))
        evaluation["argv"][2] = "evaluate"
        evaluation["result_contract"] = {
            "path": "{trial.output_dir}/task.json",
            "scope": "evaluation",
        }
        aggregate = json.loads(json.dumps(task))
        aggregate["argv"] = [
            sys.executable,
            "scripts/pipeline.py",
            "aggregate",
            "--inputs",
            "{runtime.inputs}",
            "--output",
            "{trial.output_dir}/task.json",
        ]
        aggregate["result_contract"] = {
            "path": "{trial.output_dir}/task.json",
            "scope": "aggregate",
        }
        support.write_yaml(
            self.root / ".research/tasks.yaml",
            {
                "schema_version": "1.0",
                "tasks": {
                    "train": task,
                    "evaluate": evaluation,
                    "aggregate": aggregate,
                },
            },
        )
        stages = [
            {"id": "smoke", "kind": "trials", "task": "train"},
            {"id": "tuning", "kind": "trials", "task": "train"},
            {
                "id": "main",
                "kind": "trials",
                "task": "train",
                "requires_approval": True,
            },
            {
                "id": "evaluate",
                "purpose": "evaluation",
                "uses": "main",
                "kind": "trials",
                "task": "evaluate",
            },
            {"id": "aggregate", "kind": "campaign", "task": "aggregate"},
        ]
        support.write_yaml(
            self.root / ".research/pipeline.yaml",
            {"schema_version": "1.0", "pipeline": {"stages": stages}},
        )
        support.write_yaml(
            self.root / ".research/metrics.yaml",
            {
                "schema_version": "1.0",
                "metric_contract": {
                    "primary": {"name": "validation.mse", "direction": "minimize"},
                    "validation": {"require_baseline": True},
                },
            },
        )
        campaign = {
            "schema_version": "1.0",
            "campaign": {"id": "integration", "seeds": [7, 17]},
            "variants": [{"id": "linear", "base_config": "configs/base.yaml"}],
            "matrix": {},
            "config_contract": {"seed": "seed", "run_dir": "run_dir"},
            "allowed_change_scope": ["seed", "run_dir", "steps"],
            "bindings": {"run_dir": "{trial.output_dir}/model"},
            "dataset_manifests": [],
            "stages": {"smoke": {"max_trials": 1}},
            "selection": {"source_stage": "tuning", "mode": "all"},
        }
        support.write_yaml(self.root / ".research/campaigns/integration.yaml", campaign)
        self.reload()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def reload(self) -> None:
        self.config = load_project(self.root, "smoke")

    def edit(self, relative: str, modify) -> None:
        path = self.root / relative
        value = yaml.safe_load(path.read_text())
        modify(value)
        support.write_yaml(path, value)
        self.reload()

    def test_full_cpu_train_approval_evaluate_aggregate_and_cache(self) -> None:
        self.assertEqual(run_campaign(self.config, "integration"), 2)
        self.assertEqual(len(approve_campaign(self.config, "integration", "main")), 2)
        self.assertEqual(run_campaign(self.config, "integration"), 0)
        journal = open_journal(self.config)
        rows = journal.nodes("integration")
        journal.close()
        self.assertEqual(len([row for row in rows if row["state"] == "COMPLETE"]), 8)
        files = {
            path: path.stat().st_mtime_ns
            for path in self.root.glob("runs/research/**/result.json")
        }
        self.assertEqual(run_campaign(self.config, "integration"), 0)
        self.assertEqual(files, {path: path.stat().st_mtime_ns for path in files})
        summary = next(row for row in rows if row["stage"] == "aggregate")
        self.assertLess(json.loads(summary["metrics_json"])["aggregate.mse"], 1)

    def test_invalid_mutation_and_approval_removal_are_rejected_before_run(
        self,
    ) -> None:
        self.edit(
            ".research/campaigns/integration.yaml",
            lambda value: value["bindings"].update({"dataset.split": "test"}),
        )
        with self.assertRaises(ConfigError):
            plan_campaign(self.config, "integration")
        self.assertFalse((self.root / "runs").exists())
        self.edit(
            ".research/campaigns/integration.yaml",
            lambda value: value["bindings"].pop("dataset.split"),
        )
        self.edit(
            ".research/pipeline.yaml",
            lambda value: value["pipeline"]["stages"][2].update(
                {"requires_approval": False}
            ),
        )
        with self.assertRaises(ConfigError):
            plan_campaign(self.config, "integration")

    def test_inheritance_and_policy_changes_invalidate_identity(self) -> None:
        support.write_yaml(
            self.root / "configs/parent.yaml",
            {"seed": 0, "run_dir": "unused", "steps": 20},
        )
        support.write_yaml(
            self.root / "configs/base.yaml", {"base_config": "parent.yaml"}
        )
        first = plan_campaign(self.config, "integration")[1]
        support.write_yaml(
            self.root / "configs/parent.yaml",
            {"seed": 0, "run_dir": "unused", "steps": 21},
        )
        second = plan_campaign(self.config, "integration")[1]
        self.assertNotEqual(first, second)
        self.edit(
            ".research/tasks.yaml",
            lambda value: value["tasks"]["train"].update({"timeout_seconds": 21}),
        )
        self.assertNotEqual(second, plan_campaign(self.config, "integration")[1])
        support.write_yaml(
            self.root / "configs/parent.yaml", {"base_config": "base.yaml"}
        )
        with self.assertRaises(ConfigError):
            load_model_config(self.root / "configs/base.yaml")

    def test_changed_policy_requires_new_approval(self) -> None:
        self.assertEqual(run_campaign(self.config, "integration"), 2)
        approve_campaign(self.config, "integration", "main")
        self.edit(
            ".research/metrics.yaml",
            lambda value: value["metric_contract"].update(
                {"secondary": [{"name": "validation.mse", "direction": "maximize"}]}
            ),
        )
        self.assertEqual(run_campaign(self.config, "integration"), 2)

    def test_evaluation_uses_training_config_and_checkpoint(self) -> None:
        campaign, identity, trials, _ = plan_campaign(self.config, "integration")
        config = campaign_config(self.config, campaign)
        main, main_context = node_context(config, campaign, identity, "main", trials[0])
        evaluate, eval_context = node_context(
            config, campaign, identity, "evaluate", trials[0]
        )
        self.assertEqual(main, evaluate)
        self.assertEqual(
            main_context["trial.model_run_dir"], eval_context["trial.model_run_dir"]
        )
        self.assertNotEqual(
            main_context["trial.output_dir"], eval_context["trial.output_dir"]
        )

    def test_missing_cached_output_is_revalidated_and_repaired(self) -> None:
        self.assertEqual(run_campaign(self.config, "integration", until="smoke"), 0)
        done = next(self.root.glob("runs/research/**/DONE.json"))
        done.unlink()
        self.assertEqual(run_campaign(self.config, "integration", until="smoke"), 0)
        self.assertTrue(done.is_file())

    def test_status_does_not_create_journal(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(["--project-root", str(self.root), "status"]), 0)
        self.assertFalse((self.root / "runs").exists())

    def test_baseline_and_dataset_evidence_are_mandatory(self) -> None:
        output = self.root / "evidence.json"
        value = {
            "status": "complete",
            "scope": "evaluation",
            "dataset_fingerprint": "expected",
            "metrics": {},
        }
        output.write_text(json.dumps(value))
        task = {"result_contract": {"path": str(output), "scope": "evaluation"}}
        context = {"runtime.dataset_hash": "expected"}
        contract = {"validation": {"require_baseline": True}}
        with self.assertRaisesRegex(ConfigError, "baseline"):
            check_outputs(task, context, contract)
        value["dataset_fingerprint"] = "wrong"
        output.write_text(json.dumps(value))
        with self.assertRaisesRegex(ConfigError, "dataset"):
            check_outputs(task, context, contract)

    def test_portable_template_runs_another_project_without_copying_engine(
        self,
    ) -> None:
        target = self.root / "another-project"
        target.mkdir()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(["init", "--destination", str(target)]), 0)
        for path in (target / ".research").rglob("*.yaml"):
            self.assertNotIn("/home/lab704", path.read_text())
            self.assertNotIn("/work/lab704", path.read_text())
        (target / "scripts").mkdir()
        shutil.copy2(self.root / "scripts/pipeline.py", target / "scripts/pipeline.py")
        shutil.copytree(self.root / "configs", target / "configs")
        project = yaml.safe_load((self.root / ".research/project.yaml").read_text())
        project["paths"]["temp"] = "runs/research/.tmp"
        tasks = yaml.safe_load((self.root / ".research/tasks.yaml").read_text())
        tasks["tasks"]["unit_tests"] = {
            "argv": [sys.executable, "-c", "assert sum(range(4)) == 6"],
            "resources": {"gpu_count": 0},
        }
        for name, value in (
            ("project.yaml", project),
            ("tasks.yaml", tasks),
            (
                "metrics.yaml",
                yaml.safe_load((self.root / ".research/metrics.yaml").read_text()),
            ),
            (
                "campaigns/default.yaml",
                yaml.safe_load(
                    (self.root / ".research/campaigns/integration.yaml").read_text()
                ),
            ),
        ):
            support.write_yaml(target / ".research" / name, value)
        config = load_project(target, "local_1gpu")
        self.assertEqual(run_campaign(config, "default"), 2)
        approve_campaign(config, "default", "main")
        self.assertEqual(run_campaign(config, "default"), 0)
        self.assertFalse((target / "tools/researchctl").exists())

    def test_cancel_uses_journal_and_preserves_resumable_progress(self) -> None:
        self.edit(
            "configs/base.yaml",
            lambda value: value.update({"steps": 100, "delay": 0.05}),
        )
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "researchctl.cli",
                "--project-root",
                str(self.root),
                "--profile",
                "smoke",
                "run",
                "--campaign",
                "integration",
                "--until",
                "smoke",
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline = time.monotonic() + 10
        checkpoint = None
        while time.monotonic() < deadline:
            checkpoint = next(self.root.glob("runs/research/**/checkpoint.json"), None)
            if checkpoint and json.loads(checkpoint.read_text())["step"] >= 2:
                break
            time.sleep(0.03)
        self.assertIsNotNone(checkpoint)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                cli_main(
                    [
                        "--project-root",
                        str(self.root),
                        "cancel",
                        "--campaign",
                        "integration",
                    ]
                ),
                0,
            )
        process.communicate(timeout=10)
        self.assertLess(json.loads(checkpoint.read_text())["step"], 100)
        journal = open_journal(self.config)
        self.assertEqual(journal.nodes("integration")[0]["state"], "CANCELLED")
        journal.close()

    def test_smoke_profile_cannot_reach_main(self) -> None:
        support.write_yaml(
            self.root / ".research/profiles/smoke.yaml",
            {"schema_version": "1.0", "execution": {"max_stage": "smoke"}},
        )
        self.reload()
        self.assertEqual(run_campaign(self.config, "integration"), 0)
        with self.assertRaises(ConfigError):
            run_campaign(self.config, "integration", until="main")
        with self.assertRaises(ConfigError):
            approve_campaign(self.config, "integration", "main")

    def test_controller_stops_if_policy_files_change_between_nodes(self) -> None:
        self.edit(
            "configs/base.yaml",
            lambda value: value.update({"steps": 15, "delay": 0.04}),
        )
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "researchctl.cli",
                "--project-root",
                str(self.root),
                "--profile",
                "smoke",
                "run",
                "--campaign",
                "integration",
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline = time.monotonic() + 10
        while not next(self.root.glob("runs/research/**/checkpoint.json"), None):
            if time.monotonic() > deadline or process.poll() is not None:
                self.fail("Controller did not begin the bounded CPU trial")
            time.sleep(0.02)
        self.edit(
            ".research/safety.yaml",
            lambda value: value["filesystem"]["forbidden_write_paths"].append(
                "protected"
            ),
        )
        _, error = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 1)
        self.assertIn("changed while running", error.decode())
        journal = open_journal(self.config)
        self.assertEqual(
            {row["stage"] for row in journal.nodes("integration")}, {"smoke"}
        )
        journal.close()

    def test_persistent_launch_has_one_controller_and_runs_same_bounded_graph(
        self,
    ) -> None:
        self.edit(
            "configs/base.yaml",
            lambda value: value.update({"steps": 20, "delay": 0.05}),
        )
        environment = {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
        with patch.dict(os.environ, environment):
            first = launch_campaign(self.config, "integration", "smoke", until="smoke")
            second = launch_campaign(self.config, "integration", "smoke", until="smoke")
        self.assertEqual(first["state"], "launched")
        self.assertEqual(second["state"], "already_running")
        self.assertEqual(first["pid"], second["pid"])
        deadline = time.monotonic() + 10
        while start_ticks(first["pid"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNone(start_ticks(first["pid"]))
        reap_background()
        journal = open_journal(self.config)
        rows = journal.nodes("integration")
        journal.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "COMPLETE")

    def test_controller_interruption_recovers_checkpoint_without_duplicate_nodes(
        self,
    ) -> None:
        self.edit(
            "configs/base.yaml",
            lambda value: value.update({"steps": 30, "delay": 0.05}),
        )
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "researchctl.cli",
                "--project-root",
                str(self.root),
                "--profile",
                "smoke",
                "run",
                "--campaign",
                "integration",
                "--until",
                "smoke",
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline = time.monotonic() + 10
        checkpoint = None
        while time.monotonic() < deadline:
            checkpoint = next(self.root.glob("runs/research/**/checkpoint.json"), None)
            if checkpoint and json.loads(checkpoint.read_text())["step"] >= 2:
                break
            if process.poll() is not None:
                self.fail(process.stderr.read().decode())
            time.sleep(0.03)
        self.assertIsNotNone(checkpoint)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        saved_step = json.loads(checkpoint.read_text())["step"]
        self.assertLess(saved_step, 30)
        self.assertEqual(run_campaign(self.config, "integration", until="smoke"), 0)
        self.assertEqual(json.loads(checkpoint.read_text())["step"], 30)
        journal = open_journal(self.config)
        self.assertEqual(len(journal.nodes("integration")), 1)
        journal.close()


class HostPolicyTest(unittest.TestCase):
    def run_task(
        self, code: str, environment: dict | None = None
    ) -> tuple[str, str, Path]:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        output = root / "output"
        output.mkdir()
        (root / "data").mkdir()
        protected = root / "data/truth.txt"
        protected.write_text("unchanged")
        journal = Journal(root / "journal.sqlite")
        journal.ensure_node(
            {
                "node_id": "test",
                "campaign_id": "test",
                "campaign_hash": "hash",
                "trial_id": "trial",
                "stage": "test",
            }
        )
        try:
            result = execute(
                node_id="test",
                task={
                    "argv": [sys.executable, "-c", code],
                    "timeout_seconds": 5,
                    "shutdown_grace_seconds": 0.2,
                    "resources": {"gpu_count": 0},
                },
                context={
                    "project.root": str(root),
                    "trial.output_dir": str(output),
                    "runtime.writable_roots": [str(output)],
                },
                environment=environment or {},
                journal=journal,
                limits={"rlimits": {}},
                lock_root=root / "locks",
                safety={
                    "network": {"default": "deny"},
                    "filesystem": {"allow_proc_runtime": True},
                },
            )
        finally:
            journal.close()
        self.assertEqual(protected.read_text(), "unchanged")
        return (
            result.state,
            (output / "stdout.log").read_text() + (output / "stderr.log").read_text(),
            output,
        )

    def test_real_landlock_blocks_dataset_and_source_writes_in_descendants(
        self,
    ) -> None:
        state, log, _ = self.run_task(
            "from pathlib import Path; Path('data/truth.txt').write_text('corrupt')"
        )
        self.assertEqual(state, "FAILED")
        self.assertIn("PermissionError", log)

    def test_real_seccomp_blocks_network(self) -> None:
        state, log, _ = self.run_task(
            "import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)"
        )
        self.assertEqual(state, "FAILED")
        self.assertIn("Operation not permitted", log)

    def test_secret_is_redacted_across_output_chunks(self) -> None:
        secret = "integration-test-secret-73"
        state, log, _ = self.run_task(
            "import os,time; value=os.environ['COMET_API_KEY']; "
            "os.write(1,value[:8].encode()); time.sleep(.2); os.write(1,value[8:].encode())",
            {"COMET_API_KEY": secret},
        )
        self.assertEqual(state, "COMPLETE")
        self.assertNotIn(secret, log)
        self.assertIn("<redacted>", log)

    def test_timeout_kills_descendants_that_ignore_sigterm(self) -> None:
        code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)']); "
            "Path('output/child.pid').write_text(str(child.pid)); time.sleep(20)"
        )
        state, _, output = self.run_task(code)
        self.assertEqual(state, "TIMED_OUT")
        self.assertIsNone(start_ticks(int((output / "child.pid").read_text())))

    def test_reused_pid_cannot_be_signalled(self) -> None:
        row = {"pid": os.getpid(), "pgid": os.getpid(), "process_start_ticks": "wrong"}
        with patch("researchctl.processes.os.killpg") as kill:
            self.assertFalse(signal_node(row))
            kill.assert_not_called()

    def test_proc_runtime_permission_does_not_allow_writes_through_magic_links(
        self,
    ) -> None:
        state, log, _ = self.run_task(
            "import os; fd=os.open('data/truth.txt',os.O_RDONLY); "
            "os.open('/proc/self/fd/%d'%fd,os.O_WRONLY)"
        )
        self.assertEqual(state, "FAILED")
        self.assertIn("PermissionError", log)
