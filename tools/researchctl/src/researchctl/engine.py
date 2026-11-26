"""Campaign-specific, evidence-bound execution of one-shot ML workers."""

from __future__ import annotations

import getpass
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import (
    ConfigError,
    ProjectConfig,
    Trial,
    campaign_config,
    campaign_identity,
    canonical_hash,
    expand_trials,
    get_dot_path,
    hash_file,
    interpolate,
    load_campaign,
    load_model_config,
    redact,
    redact_text,
    resolve_project_path,
    set_dot_path,
)
from .contracts import artifact_receipt, check_outputs, valid_receipt, write_json
from .executor import GpuLock, execute, task_temp_directory
from .journal import Journal
from .preflight import require_preflight
from .processes import node_alive, signal_node, start_ticks
from .selection import rank_rows
from .validation import validate_campaign_contract

_BACKGROUND: dict[int, subprocess.Popen] = {}


def reap_background() -> None:
    """Reap completed controllers when researchctl is used as a Python API."""
    for pid, process in list(_BACKGROUND.items()):
        if process.poll() is not None:
            del _BACKGROUND[pid]


def campaign_id(campaign: Mapping[str, Any]) -> str:
    value = str(campaign.get("campaign", {}).get("id", ""))
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ConfigError("campaign.id must be a directory-safe identifier")
    return value


def journal_paths(config: ProjectConfig) -> tuple[Path, Path]:
    root = resolve_project_path(config.root, config.project["paths"]["runs"])
    return root / "journal.sqlite", root / "journal.jsonl"


def open_journal(config: ProjectConfig) -> Journal:
    return Journal(journal_paths(config)[0])


def stage_trials(
    campaign: Mapping[str, Any], trials: list[Trial], stage: str
) -> list[Trial]:
    options = campaign["stages"].get(stage, {})
    variants = set(options.get("variants", []))
    selected = [trial for trial in trials if not variants or trial.variant in variants]
    limit = options.get("max_trials")
    return selected if limit is None else selected[: int(limit)]


def _base_context(config: ProjectConfig) -> dict[str, Any]:
    context = {
        "project.root": str(config.root),
        "runtime.python": str(config.project["runtime"]["python"]),
    }
    for key, value in config.project["paths"].items():
        if isinstance(value, str):
            context[f"paths.{key}"] = str(resolve_project_path(config.root, value))
    context["runtime.comet"] = bool(
        config.project.get("comet", {}).get("enabled", False)
    )
    return context


def _stage(config: ProjectConfig, identifier: str) -> Mapping[str, Any]:
    for stage in config.pipeline["pipeline"]["stages"]:
        if stage["id"] == identifier:
            return stage
    raise ConfigError(f"Unknown stage: {identifier}")


def _output_dir(
    config: ProjectConfig,
    campaign: Mapping[str, Any],
    identity: Mapping[str, str],
    stage: str,
    trial: str,
) -> Path:
    return (
        resolve_project_path(config.root, config.project["paths"]["runs"])
        / campaign_id(campaign)
        / identity["campaign_hash"][:12]
        / stage
        / trial
    )


def node_context(
    config: ProjectConfig,
    campaign: Mapping[str, Any],
    identity: Mapping[str, str],
    stage: str,
    trial: Trial,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve trial configuration; evaluation reuses the exact training config."""
    spec = _stage(config, stage)
    output = _output_dir(config, campaign, identity, stage, trial.trial_id)
    context = {
        **_base_context(config),
        "campaign.id": campaign_id(campaign),
        "campaign.hash": identity["campaign_hash"],
        "stage.id": stage,
        "trial.id": trial.trial_id,
        "trial.variant": trial.variant,
        "trial.seed": trial.seed,
        "trial.output_dir": str(output),
        "trial.config": str(output / "config.resolved.yaml"),
        "runtime.inputs": str(output / "inputs.json"),
        "runtime.dataset_hash": identity["dataset_fingerprint"],
    }
    if spec.get("kind", "trials") == "campaign":
        context["trial.model_run_dir"] = str(output)
        return {}, context
    if spec.get("uses"):
        model, parent = node_context(
            config, campaign, identity, str(spec["uses"]), trial
        )
        context["trial.model_run_dir"] = parent["trial.model_run_dir"]
        return model, context
    variant = next(item for item in campaign["variants"] if item["id"] == trial.variant)
    model = load_model_config(resolve_project_path(config.root, variant["base_config"]))
    fields = campaign.get(
        "config_contract",
        {
            "seed": "experiment.random_seed",
            "name": "experiment.name",
            "run_root": "paths.run_root",
        },
    )
    options = campaign["stages"].get(stage, {})
    preserve = bool(options.get("preserve_config", False))
    allowed = set(campaign["allowed_change_scope"])
    if preserve:
        if options.get("overrides") or options.get("bindings"):
            raise ConfigError("preserve_config cannot mutate the existing run")
        if fields.get("seed") and get_dot_path(model, fields["seed"]) != trial.seed:
            raise ConfigError(
                "Existing checkpoint seed differs from the declared trial"
            )
        for path, value in trial.parameters.items():
            if get_dot_path(model, path) != value:
                raise ConfigError(f"Existing run cannot change {path}")
    else:
        changes = {**trial.parameters, **options.get("overrides", {})}
        if fields.get("seed"):
            changes[fields["seed"]] = trial.seed
        bindings = {**campaign.get("bindings", {}), **options.get("bindings", {})}
        changes.update(
            {
                path: interpolate(value, context) if isinstance(value, str) else value
                for path, value in bindings.items()
            }
        )
        for path, value in changes.items():
            if path not in allowed:
                raise ConfigError(f"Mutation outside allowed_change_scope: {path}")
            set_dot_path(model, path, value)
        comet = model.get("comet_logging")
        if isinstance(comet, dict):
            comet["using"] = (
                context["runtime.comet"] and spec.get("purpose", stage) != "smoke"
            )
            comet["api_key_env"] = config.project.get("comet", {}).get(
                "api_key_env", "COMET_API_KEY"
            )
            comet.pop("api_key", None)
            # A generated trial must not reuse a previous Comet experiment key.
            comet["experiment_key"] = canonical_hash(
                {
                    "campaign": identity["campaign_hash"],
                    "stage": stage,
                    "trial": trial.trial_id,
                }
            )[:32]
    if fields.get("run_dir"):
        run_dir = resolve_project_path(
            config.root, str(get_dot_path(model, fields["run_dir"]))
        )
    elif fields.get("run_root"):
        root = resolve_project_path(
            config.root, str(get_dot_path(model, fields["run_root"]))
        )
        run_dir = root / str(get_dot_path(model, fields["name"]))
    else:
        run_dir = output
    context["trial.model_run_dir"] = str(run_dir.resolve())
    return model, context


def write_resolved_config(path: Path, model: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(redact_text(yaml.safe_dump(redact(model), sort_keys=False)))
    temporary.replace(path)


def plan_campaign(
    config: ProjectConfig, identifier: str
) -> tuple[dict[str, Any], dict[str, str], list[Trial], list[dict[str, Any]]]:
    """Validate and expand a campaign without launching tasks or writing files."""
    _, campaign = load_campaign(config, identifier)
    config = campaign_config(config, campaign)
    validate_campaign_contract(config, campaign)
    identity = campaign_identity(config, campaign)
    trials = expand_trials(campaign, identity["campaign_hash"])
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ConfigError("Duplicate variant/parameter/seed combinations")
    nodes = []
    for spec in config.pipeline["pipeline"]["stages"]:
        stage = str(spec["id"])
        options = campaign["stages"].get(stage, {})
        if not options.get("enabled", spec.get("enabled", True)):
            continue
        kind = spec.get("kind", "trials")
        candidates = (
            stage_trials(campaign, trials, stage)
            if kind == "trials"
            else [Trial("campaign", "campaign", {}, 0)]
        )
        for trial in candidates:
            node = {
                "node_id": canonical_hash(
                    {
                        "campaign_hash": identity["campaign_hash"],
                        "stage": stage,
                        "trial_id": trial.trial_id,
                    }
                )[:20],
                "campaign_id": campaign_id(campaign),
                "campaign_hash": identity["campaign_hash"],
                "trial_id": trial.trial_id,
                "variant": trial.variant,
                "seed": trial.seed,
                "parameters": trial.parameters,
                "stage": stage,
                "kind": kind,
                "purpose": spec.get("purpose", stage),
                "uses": spec.get("uses"),
                "task": str(options.get("task", spec["task"])),
                "requires_approval": bool(spec.get("requires_approval", False)),
            }
            model, context = node_context(config, campaign, identity, stage, trial)
            _writable_roots(
                config, config.tasks["tasks"][node["task"]], context, create=False
            )
            task = config.tasks["tasks"][node["task"]]
            node.update(
                {
                    "command": [interpolate(str(arg), context) for arg in task["argv"]],
                    "output_dir": context["trial.output_dir"],
                    "model_run_dir": context["trial.model_run_dir"],
                    "resolved_config_hash": canonical_hash(model),
                    "environment_fingerprint": identity["environment_fingerprint"],
                    "gpu": (
                        task.get("resources", {}).get("gpu")
                        if task.get("resources", {}).get("gpu_count")
                        else None
                    ),
                }
            )
            nodes.append(node)
    enabled = {node["stage"] for node in nodes}
    for node in nodes:
        if node["uses"] and node["uses"] not in enabled:
            raise ConfigError(
                f"Stage {node['stage']} requires enabled stage {node['uses']}"
            )
    nodes = _filtered_nodes(nodes, config.profile.get("execution", {}).get("max_stage"))
    return campaign, identity, trials, nodes


def _writable_roots(
    config: ProjectConfig,
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    create: bool,
) -> list[str]:
    declared = [
        resolve_project_path(config.root, item).resolve()
        for item in config.safety["filesystem"]["writable_roots"]
    ]
    candidates = [context["trial.output_dir"], context["trial.model_run_dir"]]
    candidates.append(str(task_temp_directory(context)))
    candidates.extend(
        interpolate(str(item), context) for item in task.get("writable_paths", [])
    )
    roots = []
    for value in candidates:
        path = Path(value).resolve()
        if not any(path == root or root in path.parents for root in declared):
            raise ConfigError(f"Task write path outside safety policy: {path}")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        roots.append(str(path))
    if task.get("resources", {}).get("shared_memory", False):
        if not config.safety["filesystem"].get("allow_posix_shm", False):
            raise ConfigError(
                "Task requests POSIX shared memory without safety authorization"
            )
        roots.append("/dev/shm")
    return sorted(set(roots))


def _filtered_nodes(
    nodes: list[dict[str, Any]], until: str | None
) -> list[dict[str, Any]]:
    if until is None:
        return nodes
    if until not in {node["stage"] for node in nodes}:
        raise ConfigError(f"Unknown/disabled --until stage: {until}")
    last = max(i for i, node in enumerate(nodes) if node["stage"] == until)
    return nodes[: last + 1]


def preflight_config(
    config: ProjectConfig, campaign: Mapping[str, Any], nodes: list[dict[str, Any]]
) -> ProjectConfig:
    """Check only tasks and inputs relevant to this requested campaign segment."""
    config = campaign_config(config, campaign)
    names = {node["task"] for node in nodes}
    return replace(
        config, tasks={"tasks": {name: config.tasks["tasks"][name] for name in names}}
    )


def _selection_fingerprint(rows: list[Mapping[str, Any]]) -> str:
    return canonical_hash(
        [
            {
                "node": row["node_id"],
                "metrics": row["metrics_json"],
                "result_sha256": hash_file(
                    Path(json.loads(row["artifacts_json"])["result"])
                ),
            }
            for row in sorted(rows, key=lambda row: row["node_id"])
        ]
    )


def _source_rows(
    journal: Journal, campaign: Mapping[str, Any], identity: Mapping[str, str]
) -> list[Mapping[str, Any]]:
    source = campaign.get("selection", {}).get("source_stage", "tuning")
    rows = journal.nodes(campaign_id(campaign), identity["campaign_hash"], source)
    if not rows or any(row["state"] != "COMPLETE" for row in rows):
        raise ConfigError(f"All {source} nodes must complete before approval")
    for row in rows:
        path = Path(json.loads(row["artifacts_json"])["result"])
        result = json.loads(path.read_text())
        if not valid_receipt(path, result["input_hash"]):
            raise ConfigError(
                "Approval source artifacts changed; resume the campaign first"
            )
    return rows


def approve_campaign(
    config: ProjectConfig,
    identifier: str,
    stage: str,
    selected_trials: list[str] | None = None,
) -> list[str]:
    """Record an explicit, evidence-bound approval; ablation campaigns retain all trials."""
    campaign, identity, trials, nodes = plan_campaign(config, identifier)
    config = campaign_config(config, campaign)
    targets = [
        node for node in nodes if node["stage"] == stage and node["requires_approval"]
    ]
    if not targets:
        raise ConfigError(f"Stage {stage} is not an approval-gated stage")
    with GpuLock(journal_paths(config)[0].parent / ".locks", "scheduler"):
        journal = open_journal(config)
        try:
            rows = _source_rows(journal, campaign, identity)
            source_stage = campaign.get("selection", {}).get("source_stage", "tuning")
            expected_source = {
                node["node_id"] for node in nodes if node["stage"] == source_stage
            }
            if {row["node_id"] for row in rows} != expected_source:
                raise ConfigError(
                    "The complete pre-main stage must finish before approval"
                )
            mode = campaign.get("selection", {}).get("mode", "rank")
            available = {row["trial_id"] for row in rows}
            expected = {node["trial_id"] for node in targets}
            if mode == "all":
                if not expected <= available:
                    raise ConfigError(
                        "Every main trial requires successful pre-main evidence"
                    )
                selected = sorted(expected)
            else:
                maximum = int(campaign.get("selection", {}).get("max_trials", 1))
                ranked = rank_rows(rows, config.metrics["metric_contract"])
                selected = [str(row["trial_id"]) for row in ranked[:maximum]]
            if selected_trials is not None:
                if (
                    not selected_trials
                    or not set(selected_trials) <= available & expected
                ):
                    raise ConfigError(
                        "Explicit selections must have completed pre-main evidence"
                    )
                if mode == "all" and set(selected_trials) != expected:
                    raise ConfigError(
                        "A fixed ablation matrix requires approval of all declared trials"
                    )
                selected = sorted(set(selected_trials))
            journal.approve(
                campaign_id(campaign),
                identity["campaign_hash"],
                stage,
                selected,
                {**identity, "selection_fingerprint": _selection_fingerprint(rows)},
                getpass.getuser(),
            )
            journal.export_jsonl(journal_paths(config)[1])
            return selected
        finally:
            journal.close()


def run_campaign(
    config: ProjectConfig, identifier: str, until: str | None = None
) -> int:
    """Execute the campaign with one controller, one task per journal node."""
    campaign, identity, trials, all_nodes = plan_campaign(config, identifier)
    config = campaign_config(config, campaign)
    nodes = _filtered_nodes(all_nodes, until)
    checks = require_preflight(preflight_config(config, campaign, nodes))
    lookup = {trial.trial_id: trial for trial in trials}
    root = journal_paths(config)[0].parent
    with GpuLock(root / ".locks", "scheduler"):
        journal = open_journal(config)
        write_json(
            root / ".controllers" / f"{campaign_id(campaign)}.json",
            {
                "pid": os.getpid(),
                "process_start_ticks": start_ticks(os.getpid()),
                "campaign_hash": identity["campaign_hash"],
                "preflight": [asdict(check) for check in checks],
            },
        )
        predecessors: list[dict[str, Any]] = []
        try:
            for node in nodes:
                journal.ensure_node(node)
                task = config.tasks["tasks"][node["task"]]
                trial = lookup.get(
                    node["trial_id"], Trial("campaign", "campaign", {}, 0)
                )
                model, context = node_context(
                    config, campaign, identity, node["stage"], trial
                )
                if node["requires_approval"] or node.get("uses") == "main":
                    stage = node["stage"] if node["requires_approval"] else "main"
                    approval = journal.approval(
                        campaign_id(campaign), identity["campaign_hash"], stage
                    )
                    if approval is None:
                        print(
                            f"Blocked before {stage}: researchctl approve --campaign {campaign_id(campaign)} --stage {stage}"
                        )
                        return 2
                    rows = _source_rows(journal, campaign, identity)
                    if json.loads(approval["identity_json"]).get(
                        "selection_fingerprint"
                    ) != _selection_fingerprint(rows):
                        raise ConfigError(
                            "Pre-main evidence changed; approval must be renewed"
                        )
                    if node["trial_id"] not in json.loads(
                        approval["selected_trials_json"]
                    ):
                        continue
                existing = journal.node(node["node_id"])
                if existing and existing["state"] == "RUNNING" and node_alive(existing):
                    raise ConfigError(
                        "An orphaned task is still alive; inspect/cancel it before resume"
                    )
                uses = node.get("uses")
                if uses and not any(
                    item["stage"] == uses and item["trial_id"] == trial.trial_id
                    for item in predecessors
                ):
                    raise ConfigError(
                        f"Missing completed {uses} node for trial {trial.trial_id}"
                    )
                input_hash = canonical_hash(
                    {"identity": identity, "model": model, "parents": predecessors}
                )
                result_path = Path(context["trial.output_dir"]) / "result.json"
                if (
                    existing
                    and existing["state"] == "COMPLETE"
                    and valid_receipt(result_path, input_hash)
                ):
                    check_outputs(task, context, config.metrics["metric_contract"])
                    predecessors.append(_predecessor(node, result_path))
                    continue
                if existing and existing["state"] == "RUNNING":
                    journal.transition(
                        node["node_id"],
                        "FAILED",
                        failure_reason="Recovered interrupted controller; task no longer alive",
                    )
                journal.transition(
                    node["node_id"],
                    "PENDING",
                    pid=None,
                    pgid=None,
                    process_start_ticks=None,
                )
                context["runtime.writable_roots"] = _writable_roots(
                    config, task, context, create=True
                )
                quota_roots = [
                    Path(path)
                    for path in context["runtime.writable_roots"]
                    if path != "/dev/shm"
                ]
                context["runtime.quota_roots"] = [
                    str(path)
                    for path in quota_roots
                    if not any(other in path.parents for other in quota_roots)
                ]
                if model:
                    write_resolved_config(Path(context["trial.config"]), model)
                write_json(
                    Path(context["runtime.inputs"]),
                    {
                        "campaign_hash": identity["campaign_hash"],
                        "nodes": predecessors,
                        "dataset_fingerprint": identity["dataset_fingerprint"],
                    },
                )
                environment = dict(config.project["runtime"].get("environment", {}))
                comet = config.project.get("comet", {})
                environment.update(
                    {
                        "RESEARCHCTL_COMET_PROJECT": str(
                            comet.get("project_name", "mfqev2")
                        ),
                        "RESEARCHCTL_COMET_WORKSPACE": str(comet.get("workspace", "")),
                    }
                )
                result = execute(
                    node_id=node["node_id"],
                    task=task,
                    context=context,
                    environment=environment,
                    journal=journal,
                    limits=config.resources["resources"],
                    lock_root=root / ".locks",
                    safety=config.safety,
                    metric_contract=config.metrics["metric_contract"],
                )
                artifacts = dict(result.artifacts)
                if model:
                    artifacts["resolved_config"] = context["trial.config"]
                write_json(
                    result_path,
                    {
                        "schema_version": "1.1",
                        "status": result.state.lower(),
                        "scope": node["purpose"],
                        "research_result": node["purpose"]
                        in {"main", "evaluation", "aggregate"},
                        "metrics": result.metrics,
                        "artifacts": artifacts,
                        "artifact_hashes": artifact_receipt(artifacts),
                        "input_hash": input_hash,
                        "provenance": {
                            **identity,
                            "trial_id": trial.trial_id,
                            "seed": trial.seed,
                            "resolved_config_hash": canonical_hash(model),
                        },
                        "failure_reason": result.failure_reason,
                    },
                )
                journal.transition(
                    node["node_id"],
                    result.state,
                    exit_code=result.exit_code,
                    metrics=result.metrics,
                    artifacts={**artifacts, "result": str(result_path)},
                    failure_reason=result.failure_reason,
                )
                journal.export_jsonl(journal_paths(config)[1])
                if result.state != "COMPLETE":
                    return 1
                # Detect edits between nodes, including policy and metric changes.
                _, now, _, _ = plan_campaign(config, identifier)
                if now != identity:
                    raise ConfigError(
                        "Source/config/data changed while running; campaign stopped"
                    )
                predecessors.append(_predecessor(node, result_path))
            return 0
        finally:
            journal.export_jsonl(journal_paths(config)[1])
            journal.close()


def _predecessor(node: Mapping[str, Any], result: Path) -> dict[str, Any]:
    return {
        "node_id": node["node_id"],
        "trial_id": node["trial_id"],
        "variant": node["variant"],
        "seed": node["seed"],
        "stage": node["stage"],
        "purpose": node["purpose"],
        "result": str(result),
        "sha256": hash_file(result),
    }


def cancel_campaign(config: ProjectConfig, identifier: str) -> int:
    """Cancel verified task groups across identities, even after config edits."""
    _, campaign = load_campaign(config, identifier)
    journal = open_journal(config)
    cancelled = 0
    try:
        for row in journal.nodes(campaign_id(campaign)):
            if row["state"] == "RUNNING":
                signal_node(row)
                journal.transition(
                    row["node_id"], "CANCELLED", failure_reason="Cancelled by operator"
                )
                cancelled += 1
        journal.export_jsonl(journal_paths(config)[1])
    finally:
        journal.close()
    controller = (
        journal_paths(config)[0].parent
        / ".controllers"
        / f"{campaign_id(campaign)}.json"
    )
    if controller.exists():
        record = json.loads(controller.read_text())
        pid = int(record["pid"])
        if (
            pid != os.getpid()
            and record.get("process_start_ticks")
            and start_ticks(pid) == record["process_start_ticks"]
        ):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return cancelled


def launch_campaign(
    config: ProjectConfig, identifier: str, profile: str, until: str | None = None
) -> dict[str, Any]:
    """Start a persistent controller; the child still enforces all run gates."""
    reap_background()
    campaign, identity, _, nodes = plan_campaign(config, identifier)
    require_preflight(preflight_config(config, campaign, _filtered_nodes(nodes, until)))
    root = journal_paths(config)[0].parent
    with GpuLock(root / ".locks", "launch"):
        controller = root / ".controllers" / f"{campaign_id(campaign)}.json"
        if controller.exists():
            record = json.loads(controller.read_text())
            if (
                record.get("process_start_ticks")
                and start_ticks(int(record["pid"])) == record["process_start_ticks"]
            ):
                return {"state": "already_running", **record}
        log = root / ".controllers" / f"{campaign_id(campaign)}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            "-m",
            "researchctl.cli",
            "--project-root",
            str(config.root),
            "--profile",
            profile,
            "run",
            "--campaign",
            identifier,
        ]
        if until:
            argv.extend(["--until", until])
        with log.open("ab") as handle:
            process = subprocess.Popen(
                argv,
                cwd=config.root,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _BACKGROUND[process.pid] = process
        record = {
            "pid": process.pid,
            "process_start_ticks": start_ticks(process.pid),
            "campaign_hash": identity["campaign_hash"],
            "log": str(log),
        }
        write_json(controller, record)
        return {"state": "launched", **record}
