"""Configuration loading, validation, interpolation, and provenance."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from . import __version__


SCHEMA_VERSION = "1.0"
SECRET_KEYS = {"api_key", "password", "secret", "token"}
PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.-]+)\}")
_HASH_CACHE: dict[str, tuple[tuple[int, ...], str]] = {}
NODE_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "FAILED",
    "TIMED_OUT",
    "CANCELLED",
    "QUOTA_EXCEEDED",
}


class ConfigError(ValueError):
    """Raised when a portable research configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    """Fully loaded project-level automation configuration."""

    root: Path
    config_root: Path
    project: dict[str, Any]
    tasks: dict[str, Any]
    pipeline: dict[str, Any]
    metrics: dict[str, Any]
    resources: dict[str, Any]
    safety: dict[str, Any]
    profile: dict[str, Any]


@dataclass(frozen=True)
class Trial:
    """One deterministic point in a campaign search matrix."""

    trial_id: str
    variant: str
    parameters: dict[str, Any]
    seed: int


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and reject non-mapping roots."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Missing configuration: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def redact(value: Any) -> Any:
    """Return a recursively redacted, serializable copy."""
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if is_secret_key(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return copy.deepcopy(value)


def is_secret_key(key: str) -> bool:
    """Recognize credential values, while retaining environment variable names."""
    name = key.lower()
    return not name.endswith("_env") and (
        name in SECRET_KEYS
        or name.endswith(("_api_key", "_token", "_password", "_secret"))
    )


def secret_values(environment: Mapping[str, str] | None = None) -> list[str]:
    """Return environment credential values for log redaction."""
    return sorted(
        {
            value
            for key, value in (environment or os.environ).items()
            if is_secret_key(key) and value
        },
        key=len,
        reverse=True,
    )


def redact_text(value: str) -> str:
    """Remove environment credentials from diagnostic messages."""
    for secret in secret_values():
        value = value.replace(secret, "<redacted>")
    return value


def load_model_config(path: Path, active: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve YAML/JSON model inheritance without importing an ML environment."""
    path = path.resolve()
    if path in active:
        raise ConfigError(f"Cyclic base_config: {path}")
    value = read_yaml(path)
    base = value.pop("base_config", None)
    if base is None:
        return value
    parent = Path(str(base))
    if not parent.is_absolute():
        parent = path.parent / parent
    return deep_merge(load_model_config(parent, (*active, path)), value)


def canonical_hash(value: Any) -> str:
    """Hash one JSON-compatible value after secret redaction."""
    payload = json.dumps(
        redact(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    path = path.resolve()
    stat = path.stat()
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HASH_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if stamp != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ConfigError(f"File changed while hashing: {path}")
    value = digest.hexdigest()
    _HASH_CACHE[str(path)] = (stamp, value)
    return value


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing a portable ``.research`` folder."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".research/project.yaml").is_file():
            return candidate
    raise ConfigError("No .research/project.yaml found in this directory or parents")


def _require_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ConfigError(f"{label} is missing required keys: {', '.join(missing)}")


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {', '.join(unknown)}")


def load_project(root: Path, profile_name: str) -> ProjectConfig:
    """Load and validate all project-level configuration files."""
    config_root = root / ".research"
    project = read_yaml(config_root / "project.yaml")
    tasks = read_yaml(config_root / "tasks.yaml")
    pipeline = read_yaml(config_root / "pipeline.yaml")
    metrics = read_yaml(config_root / "metrics.yaml")
    resources = read_yaml(config_root / "resources.yaml")
    safety = read_yaml(config_root / "safety.yaml")
    profile = read_yaml(config_root / "profiles" / f"{profile_name}.yaml")
    _reject_unknown(
        profile, {"schema_version", "project", "resources", "execution"}, "profile"
    )
    _reject_unknown(profile.get("execution", {}), {"max_stage"}, "profile.execution")
    _reject_unknown(tasks, {"schema_version", "tasks"}, "tasks.yaml")
    _reject_unknown(pipeline, {"schema_version", "pipeline"}, "pipeline.yaml")
    for label, value in (
        ("project.yaml", project),
        ("tasks.yaml", tasks),
        ("pipeline.yaml", pipeline),
        ("metrics.yaml", metrics),
        ("resources.yaml", resources),
        ("safety.yaml", safety),
        (f"profiles/{profile_name}.yaml", profile),
    ):
        if str(value.get("schema_version")) != SCHEMA_VERSION:
            raise ConfigError(
                f"{label} must declare schema_version: '{SCHEMA_VERSION}'"
            )

    _require_keys(project, {"project", "runtime", "paths"}, "project.yaml")
    _reject_unknown(
        project,
        {
            "schema_version",
            "project",
            "runtime",
            "paths",
            "provenance",
            "comet",
            "external_runs",
        },
        "project.yaml",
    )
    _require_keys(tasks, {"tasks"}, "tasks.yaml")
    _require_keys(pipeline, {"pipeline"}, "pipeline.yaml")
    _require_keys(metrics, {"metric_contract"}, "metrics.yaml")
    _require_keys(resources, {"resources"}, "resources.yaml")
    _require_keys(safety, {"execution", "filesystem"}, "safety.yaml")

    merged_resources = deep_merge(resources, profile.get("resources", {}))
    merged_project = deep_merge(project, profile.get("project", {}))
    return ProjectConfig(
        root=root,
        config_root=config_root,
        project=merged_project,
        tasks=tasks,
        pipeline=pipeline,
        metrics=metrics,
        resources=merged_resources,
        safety=safety,
        profile=profile,
    )


def find_campaign(config_root: Path, identifier: str) -> Path:
    """Resolve a campaign path or unique campaign ID."""
    supplied = Path(identifier)
    candidates = [
        supplied,
        config_root / supplied,
        config_root / "campaigns" / supplied,
    ]
    if supplied.suffix not in {".yaml", ".yml"}:
        candidates.extend(
            [
                config_root / "campaigns" / f"{identifier}.yaml",
                config_root / "campaigns" / f"{identifier}.yml",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = []
    for path in sorted((config_root / "campaigns").glob("*.y*ml")):
        value = read_yaml(path)
        if str(value.get("campaign", {}).get("id")) == identifier:
            matches.append(path)
    if len(matches) == 1:
        return matches[0].resolve()
    raise ConfigError(f"Cannot resolve campaign: {identifier}")


def load_campaign(
    config: ProjectConfig, identifier: str
) -> tuple[Path, dict[str, Any]]:
    """Load and validate one campaign."""
    path = find_campaign(config.config_root, identifier)
    campaign = read_yaml(path)
    if str(campaign.get("schema_version")) != SCHEMA_VERSION:
        raise ConfigError(f"{path} must declare schema_version: '{SCHEMA_VERSION}'")
    _require_keys(
        campaign,
        {"campaign", "variants", "matrix", "allowed_change_scope", "stages"},
        str(path),
    )
    _reject_unknown(
        campaign,
        {
            "schema_version",
            "campaign",
            "variants",
            "matrix",
            "allowed_change_scope",
            "bindings",
            "stages",
            "dataset_manifests",
            "selection",
            "pipeline",
            "config_contract",
            "metric_contract",
            "required_paths",
        },
        str(path),
    )
    if not isinstance(campaign["variants"], list) or not campaign["variants"]:
        raise ConfigError("campaign.variants must be a non-empty list")
    if not isinstance(campaign["matrix"], Mapping):
        raise ConfigError("campaign.matrix must be a mapping")
    return path, campaign


def campaign_config(
    config: ProjectConfig, campaign: Mapping[str, Any]
) -> ProjectConfig:
    """Apply allowed campaign contracts; safety and resource policy stay immutable."""
    pipeline = {
        "pipeline": {
            "stages": campaign.get("pipeline", config.pipeline["pipeline"]["stages"])
        }
    }
    metrics = {
        "metric_contract": campaign.get(
            "metric_contract", config.metrics["metric_contract"]
        )
    }
    return replace(config, pipeline=pipeline, metrics=metrics)


def project_source_fingerprint(config: ProjectConfig) -> str:
    """Hash configured source files, including untracked project code."""
    patterns = config.project.get("provenance", {}).get(
        "include", ["src/**/*.py", "scripts/**/*.py", "configs/**/*.y*ml"]
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in config.root.glob(str(pattern)) if path.is_file())
    rows = [
        (str(path.relative_to(config.root)), hash_file(path)) for path in sorted(paths)
    ]
    return canonical_hash(rows)


def git_commit(root: Path) -> str:
    """Return HEAD when available, otherwise an explicit marker."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-repository"


def dataset_fingerprint(config: ProjectConfig, campaign: Mapping[str, Any]) -> str:
    """Hash campaign-declared dataset manifests."""
    rows = []
    for raw_path in campaign.get("dataset_manifests", []):
        path = (config.root / str(raw_path)).resolve()
        if not path.is_file():
            raise ConfigError(f"Missing dataset manifest: {path}")
        rows.append((str(path.relative_to(config.root)), hash_file(path)))
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
            for key in ("gt_lmdb", "lq_lmdb"):
                if isinstance(value, dict) and key in value:
                    database = Path(value[key])
                    if not database.is_absolute():
                        database = path.parent / database
                    storage = database / "data.mdb"
                    if not storage.is_file():
                        raise ConfigError(f"Missing dataset storage: {storage}")
                    stat = storage.stat()
                    rows.append(
                        (str(storage.resolve()), stat.st_size, stat.st_mtime_ns)
                    )
    return canonical_hash(rows)


def instructions_fingerprint(root: Path) -> str:
    """Bind approvals to the agent instructions as well as execution policy."""
    paths = {root / "AGENTS.md", root / "scripts/AGENTS.md"}
    for folder in (".agents", ".research/prompts"):
        paths.update((root / folder).glob("*.md"))
    return canonical_hash(
        [
            (str(path.relative_to(root)), hash_file(path))
            for path in sorted(paths)
            if path.is_file()
        ]
    )


def environment_fingerprint(config: ProjectConfig) -> str:
    """Fingerprint interpreter and installed package metadata without executing it."""
    python = str(config.project["runtime"]["python"])
    executable = Path(shutil.which(python) or python)
    prefix = executable.parent.parent
    paths = set(prefix.glob("lib/python*/site-packages/*.dist-info/METADATA"))
    paths.update(prefix.glob("conda-meta/*.json"))
    paths.update({prefix / "pyvenv.cfg", Path("/proc/driver/nvidia/version")})
    for task in config.tasks["tasks"].values():
        context = {
            f"paths.{key}": str(resolve_project_path(config.root, value))
            for key, value in config.project["paths"].items()
            if isinstance(value, str)
        }
        paths.update(
            resolve_project_path(config.root, interpolate(name, context))
            for name in task.get("required_paths", [])
        )
    return canonical_hash(
        {
            "python": str(executable),
            "binary": hash_file(executable) if executable.is_file() else "missing",
            "packages": [
                (str(path), hash_file(path)) for path in sorted(paths) if path.is_file()
            ],
            "environment": config.project["runtime"].get("environment", {}),
        }
    )


def campaign_identity(
    config: ProjectConfig, campaign: Mapping[str, Any]
) -> dict[str, str]:
    """Build the immutable identity used by trials and approvals."""
    identity = {
        "campaign_config_hash": canonical_hash(campaign),
        "source_fingerprint": project_source_fingerprint(config),
        "source_commit": git_commit(config.root),
        "dataset_fingerprint": dataset_fingerprint(config, campaign),
        "policy_fingerprint": canonical_hash(
            {
                "project": config.project,
                "tasks": config.tasks,
                "pipeline": config.pipeline,
                "metrics": config.metrics,
                "resources": config.resources,
                "safety": config.safety,
                "profile": config.profile,
            }
        ),
        "base_configs_fingerprint": canonical_hash(
            [
                (
                    variant["id"],
                    load_model_config(
                        resolve_project_path(config.root, variant["base_config"])
                    ),
                )
                for variant in campaign["variants"]
            ]
        ),
        "instructions_fingerprint": instructions_fingerprint(config.root),
        "environment_fingerprint": environment_fingerprint(config),
        "controller_version": __version__,
        "policy_files_fingerprint": canonical_hash(
            [
                (str(path.relative_to(config.config_root)), hash_file(path))
                for path in sorted(
                    {
                        *config.config_root.glob("*.yaml"),
                        *config.config_root.glob("*.yml"),
                        *(config.config_root / "profiles").glob("*.yaml"),
                        *(config.config_root / "profiles").glob("*.yml"),
                    }
                )
            ]
        ),
    }
    identity["campaign_hash"] = canonical_hash(identity)
    return identity


def expand_trials(campaign: Mapping[str, Any], campaign_hash: str) -> list[Trial]:
    """Expand variants and a Cartesian parameter matrix deterministically."""
    matrix = campaign["matrix"]
    keys = sorted(matrix)
    values = []
    for key in keys:
        choices = matrix[key]
        if not isinstance(choices, list) or not choices:
            raise ConfigError(f"Matrix value for {key} must be a non-empty list")
        values.append(choices)
    trials = []
    products = itertools.product(*values) if keys else [()]
    combinations = [dict(zip(keys, items)) for items in products]
    for variant in campaign["variants"]:
        if not isinstance(variant, Mapping):
            raise ConfigError("Each campaign variant must be a mapping")
        _require_keys(variant, {"id", "base_config"}, "campaign variant")
        seeds = variant.get("seeds", campaign.get("campaign", {}).get("seeds", [0]))
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("Trial seeds must be a non-empty list")
        fixed = variant.get("parameters", {})
        if not isinstance(fixed, Mapping):
            raise ConfigError("variant.parameters must be a mapping")
        for combination in combinations:
            for seed in seeds:
                parameters = {**fixed, **combination}
                payload = {
                    "campaign_hash": campaign_hash,
                    "variant": str(variant["id"]),
                    "parameters": parameters,
                    "seed": int(seed),
                }
                trial_id = canonical_hash(payload)[:16]
                trials.append(
                    Trial(trial_id, str(variant["id"]), parameters, int(seed))
                )
    return trials


def set_dot_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Set one existing-or-new nested mapping value."""
    parts = path.split(".")
    if not all(parts):
        raise ConfigError(f"Invalid dot path: {path}")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot descend through non-mapping path: {path}")
        current = child
    current[parts[-1]] = value


def get_dot_path(value: Mapping[str, Any], path: str) -> Any:
    """Read a dot-separated mapping path."""
    current: Any = value
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ConfigError(f"Missing value at path: {path}")
    return current


def interpolate(text: str, context: Mapping[str, Any]) -> str:
    """Substitute only explicitly registered placeholders."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise ConfigError(f"Unknown placeholder: {{{key}}}")
        return str(context[key])

    rendered = PLACEHOLDER.sub(replace, text)
    if "{" in rendered or "}" in rendered:
        raise ConfigError(f"Malformed placeholder expression: {text}")
    return rendered


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve one path relative to the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def command_exists(command: str) -> bool:
    """Return whether an executable is available."""
    return (
        Path(command).is_file() if "/" in command else shutil.which(command) is not None
    )
