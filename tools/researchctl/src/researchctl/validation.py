"""Static validation for the portable project contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .config import (
    ConfigError,
    PLACEHOLDER,
    ProjectConfig,
    is_secret_key,
    resolve_project_path,
)


BASE_PLACEHOLDERS = {
    "project.root",
    "runtime.python",
    "campaign.id",
    "campaign.hash",
    "stage.id",
    "trial.id",
    "trial.variant",
    "trial.seed",
    "trial.output_dir",
    "trial.config",
    "trial.model_run_dir",
    "runtime.inputs",
    "runtime.dataset_hash",
}


def _inside(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def validate_project_contract(config: ProjectConfig) -> None:
    """Reject unsafe or internally inconsistent project configuration."""
    tasks = config.tasks["tasks"]
    if any(
        is_secret_key(key) for key in config.project["runtime"].get("environment", {})
    ):
        raise ConfigError(
            "Credentials must come from the host environment, not project.yaml"
        )
    if not isinstance(tasks, Mapping) or not tasks:
        raise ConfigError("tasks.yaml must define at least one task")
    allowed_placeholders = BASE_PLACEHOLDERS | {
        f"paths.{key}"
        for key, value in config.project.get("paths", {}).items()
        if isinstance(value, (str, Path))
    }
    for name, task in tasks.items():
        if not isinstance(task, Mapping):
            raise ConfigError(f"Task {name} must be a mapping")
        unknown = set(task) - {
            "executor",
            "argv",
            "timeout_seconds",
            "shutdown_grace_seconds",
            "max_output_gb",
            "resources",
            "outputs",
            "collectors",
            "image",
            "network",
            "writable_paths",
            "required_paths",
            "result_contract",
        }
        if unknown:
            raise ConfigError(f"Task {name} has unknown keys: {sorted(unknown)}")
        argv = task.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ConfigError(f"Task {name}.argv must be a non-empty string list")
        if Path(argv[0]).name in {
            "sh",
            "bash",
            "dash",
            "zsh",
            "fish",
            "ksh",
            "csh",
            "tcsh",
            "env",
            "pwsh",
            "powershell",
        }:
            raise ConfigError(
                f"Task {name}: shell/env wrappers are forbidden; use argv and runtime.environment"
            )
        if str(task.get("executor", "host")) not in {"host", "docker"}:
            raise ConfigError(f"Task {name} has an unsupported executor")
        network = task.get("network", "deny")
        if network not in {"deny", "comet"}:
            raise ConfigError(f"Task {name}: network must be deny or comet")
        if network == "comet" and name not in config.safety.get("network", {}).get(
            "comet_tasks", []
        ):
            raise ConfigError(f"Task {name} is not authorized for Comet network access")
        if int(task.get("resources", {}).get("gpu_count", 0)) not in {0, 1}:
            raise ConfigError("Each task supports at most one GPU")
        unknown = set(task.get("resources", {})) - {"gpu_count", "gpu", "shared_memory"}
        if unknown:
            raise ConfigError(f"Unknown task resource limits: {sorted(unknown)}")
        if int(task.get("timeout_seconds", 1)) <= 0:
            raise ConfigError(f"Task {name} timeout must be positive")
        values: list[str] = list(argv)
        values.extend(str(item) for item in task.get("outputs", {}).get("required", []))
        values.extend(str(item.get("path", "")) for item in task.get("collectors", []))
        values.extend(str(item) for item in task.get("writable_paths", []))
        values.extend(str(item) for item in task.get("required_paths", []))
        if task.get("result_contract"):
            values.append(str(task["result_contract"]["path"]))
        for value in values:
            unknown = set(PLACEHOLDER.findall(value)) - allowed_placeholders
            if unknown:
                raise ConfigError(
                    f"Task {name} uses unknown placeholders: {sorted(unknown)}"
                )

    stages = config.pipeline["pipeline"].get("stages")
    if not isinstance(stages, list) or not stages:
        raise ConfigError("pipeline.stages must be a non-empty list")
    stage_ids = [str(stage.get("id", "")) for stage in stages]
    if any(not identifier for identifier in stage_ids) or len(stage_ids) != len(
        set(stage_ids)
    ):
        raise ConfigError("Pipeline stage IDs must be non-empty and unique")
    for stage in stages:
        unknown = set(stage) - {
            "id",
            "kind",
            "task",
            "purpose",
            "uses",
            "requires_approval",
            "enabled",
        }
        if unknown:
            raise ConfigError(f"Unknown stage keys: {sorted(unknown)}")
        if str(stage.get("task", "")) not in tasks:
            raise ConfigError(
                f"Pipeline stage {stage['id']} references an unknown task"
            )
        if stage.get("kind", "trials") not in {"trials", "campaign"}:
            raise ConfigError(f"Invalid stage kind: {stage['id']}")
        uses = stage.get("uses")
        if uses and (
            uses not in stage_ids
            or stage_ids.index(uses) >= stage_ids.index(stage["id"])
        ):
            raise ConfigError(f"Stage {stage['id']} must use an earlier stage")
        if stage.get("purpose", stage["id"]) == "main" or stage[
            "id"
        ] in config.safety.get("approval", {}).get("required_stages", ["main"]):
            if not stage.get("requires_approval", False):
                raise ConfigError(f"Stage {stage['id']} requires human approval")

    primary = config.metrics["metric_contract"].get("primary", {})
    if primary.get("direction") not in {"minimize", "maximize"}:
        raise ConfigError("Primary metric direction must be minimize or maximize")
    if not primary.get("name"):
        raise ConfigError("Primary metric name is required")
    if bool(config.safety.get("execution", {}).get("shell_commands_allowed", True)):
        raise ConfigError("researchctl v1 requires shell_commands_allowed: false")
    if config.safety.get("network", {}).get("default", "deny") != "deny":
        raise ConfigError(
            "Network default must be deny; authorize Comet tasks explicitly"
        )
    if config.safety.get("publication", {}).get("automatic_submission", False):
        raise ConfigError(
            "Automatic publication is outside researchctl's execution contract"
        )
    if int(config.resources["resources"].get("max_parallel_nodes", 1)) != 1:
        raise ConfigError("researchctl v1 supports max_parallel_nodes: 1 only")

    writable = [
        resolve_project_path(config.root, item).resolve()
        for item in config.safety["filesystem"].get("writable_roots", [])
    ]
    if not writable:
        raise ConfigError("At least one writable root is required")
    forbidden = [
        config.root / ".git",
        config.root / ".research",
        config.root / ".agents",
    ]
    forbidden.extend(
        resolve_project_path(config.root, item)
        for item in config.safety["filesystem"].get("forbidden_write_paths", [])
    )
    if config.safety["filesystem"].get("dataset_read_only", True):
        dataset = config.project["paths"].get("datasets")
        if dataset:
            forbidden.append(resolve_project_path(config.root, dataset))
    for root in writable:
        if root == config.root or root == Path("/"):
            raise ConfigError(f"Writable root is too broad: {root}")
        if any(
            _inside(root, [path.resolve()]) or _inside(path.resolve(), [root])
            for path in forbidden
        ):
            raise ConfigError(f"Writable root overlaps a protected path: {root}")
    for key in ("runs", "outputs", "temp"):
        value = config.project["paths"].get(key)
        if value is not None and not _inside(
            resolve_project_path(config.root, value).resolve(), writable
        ):
            raise ConfigError(f"paths.{key} must be inside a configured writable root")


def validate_campaign_contract(
    config: ProjectConfig, campaign: Mapping[str, Any]
) -> None:
    """Validate all changes and stages before creating any process or output."""
    validate_project_contract(config)
    allowed = set(campaign["allowed_change_scope"])
    mutations = set(campaign.get("matrix", {})) | set(campaign.get("bindings", {}))
    for variant in campaign["variants"]:
        mutations.update(variant.get("parameters", {}))
    stage_ids = {stage["id"] for stage in config.pipeline["pipeline"]["stages"]}
    for stage, options in campaign["stages"].items():
        unknown = set(options) - {
            "enabled",
            "task",
            "max_trials",
            "variants",
            "overrides",
            "bindings",
            "preserve_config",
        }
        if unknown:
            raise ConfigError(f"Unknown campaign stage options: {sorted(unknown)}")
        if stage not in stage_ids:
            raise ConfigError(f"Unknown campaign stage: {stage}")
        mutations.update(options.get("overrides", {}))
        mutations.update(options.get("bindings", {}))
        if options.get("task") and options["task"] not in config.tasks["tasks"]:
            raise ConfigError(f"Unknown task in stage {stage}")
        if options.get("max_trials") is not None and int(options["max_trials"]) <= 0:
            raise ConfigError("Stage max_trials must be positive")
    if mutations - allowed:
        raise ConfigError(
            f"Mutation outside allowed_change_scope: {sorted(mutations - allowed)}"
        )
    contract = campaign.get("config_contract", {})
    unknown = set(contract) - {"seed", "name", "run_dir", "run_root"}
    if unknown or (contract.get("run_dir") and contract.get("run_root")):
        raise ConfigError("Invalid config_contract field mapping")
    mode = campaign.get("selection", {}).get("mode", "rank")
    if mode not in {"all", "rank"}:
        raise ConfigError("selection.mode must be all or rank")
    if int(campaign.get("selection", {}).get("max_trials", 1)) <= 0:
        raise ConfigError("selection.max_trials must be positive")
