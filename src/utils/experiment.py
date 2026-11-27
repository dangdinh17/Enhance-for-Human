"""Experiment configuration, locking and fail-soft Comet logging helpers."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping

import yaml


SECRET_KEYS = {"api_key", "password", "secret", "token"}


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, optionally inheriting a local base config."""
    def load(source: Path, active: set[Path]) -> dict[str, Any]:
        resolved = source.resolve()
        if resolved in active:
            raise ValueError(f"Cyclic base_config reference: {source}")
        with source.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"YAML root must be a mapping: {source}")
        base_name = value.pop("base_config", None)
        if base_name is None:
            return value
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = source.parent / base_path
        base = load(base_path, active | {resolved})
        def merge(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            for key, item in override.items():
                if isinstance(item, dict) and isinstance(target.get(key), dict):
                    merge(target[key], item)
                else:
                    target[key] = item
            return target
        return merge(base, value)

    return load(Path(path), set())


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve ``value`` relative to a repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def redact_secrets(value: Any) -> Any:
    """Return a deep copy with credential-like mapping values redacted."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).lower() in SECRET_KEYS
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return copy.deepcopy(value)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a configuration after removing credential values."""
    normalized = redact_secrets(config)
    comet = normalized.get("comet_logging") if isinstance(normalized, dict) else None
    if isinstance(comet, dict) and "api_key_env" in comet:
        comet.setdefault("api_key", "<redacted>")
        comet.pop("api_key_env", None)
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_redacted_yaml(path: Path, config: Mapping[str, Any]) -> None:
    """Atomically write a configuration snapshot without credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(redact_secrets(config), sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_training_manifest(
    path: str | Path,
    project_root: Path,
) -> list[Path]:
    """Load and validate the ordered list of training configuration paths."""
    manifest_path = Path(path)
    manifest = load_yaml_mapping(manifest_path)
    raw_experiments = manifest.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("Training manifest must contain a non-empty experiments list")
    config_paths = [
        resolve_project_path(project_root, str(item)) for item in raw_experiments
    ]
    if len(set(config_paths)) != len(config_paths):
        raise ValueError("Training manifest contains duplicate config paths")
    for config_path in config_paths:
        if not config_path.is_file():
            raise FileNotFoundError(f"Training config not found: {config_path}")
    return config_paths


def validate_named_experiment(config: Mapping[str, Any]) -> tuple[str, str]:
    """Validate the stable local and Comet identity of a named experiment."""
    experiment = config["experiment"]
    name = str(experiment["name"])
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("experiment.name must be a non-empty directory-safe name")
    comet = config.get("comet_logging", {})
    if not isinstance(comet, Mapping):
        raise ValueError("comet_logging must be a mapping")
    key = str(comet.get("experiment_key", ""))
    if bool(comet.get("using", False)):
        if len(key) != 32 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError(
                "comet_logging.experiment_key must be 32 lowercase hexadecimal "
                "characters"
            )
    return name, key


def initialize_experiment_directory(
    directory: Path,
    config: Mapping[str, Any],
    qp: int,
    branch: str | None = None,
    *,
    variant: str | None = None,
) -> None:
    """Create or verify immutable experiment identity metadata."""
    if (branch is None) == (variant is None):
        raise ValueError("Define exactly one experiment branch or variant")
    name, experiment_key = validate_named_experiment(config)
    config_hash = canonical_config_hash(config)
    selector = {"branch": branch} if branch is not None else {"variant": variant}
    status_path = directory / "status.json"
    status_exists = status_path.is_file()
    if status_exists:
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        expected = {
            "experiment_name": name,
            "experiment_key": experiment_key,
            "qp": qp,
            **selector,
            "config_sha256": config_hash,
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"Experiment identity mismatch for {directory}: field {key}"
                )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    atomic_write_redacted_yaml(directory / "config.resolved.yaml", config)
    if not status_exists:
        atomic_write_json(
            status_path,
            {
                "state": "initialized",
                "experiment_name": name,
                "experiment_key": experiment_key,
                "qp": qp,
                **selector,
                "config_sha256": config_hash,
            },
        )


class ExperimentLock(AbstractContextManager):
    """Advisory non-blocking lock preventing duplicate training processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"Another process is already using experiment {self.path.parent}"
            ) from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class CometMetricLogger:
    """Small fail-soft wrapper around Comet ML for one experiment."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_directory: Path,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.directory = experiment_directory
        self._experiment: Any = None
        self.enabled = False
        self._initialize()

    def _resolve_key(self) -> None:
        """Persist one generated experiment key per run, reused after resume."""
        comet = self.config["comet_logging"]
        path = self.directory / "comet.json"
        saved = json.loads(path.read_text()) if path.is_file() else {}
        previous = str(saved.get("experiment_key", ""))
        configured = str(comet.get("experiment_key", ""))
        if previous:
            for field in ("project_name", "workspace"):
                if saved.get(field) != str(comet.get(field, "")):
                    raise ValueError(f"Comet resume {field} differs from this run")
            if configured and configured != previous:
                raise ValueError("Comet experiment key differs from this run")
        key = configured or previous or uuid.uuid4().hex
        if len(key) != 32 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError(
                "Comet experiment key must be 32 lowercase hexadecimal characters"
            )
        comet["experiment_key"] = key
        self._write_state("initializing")

    def _safe_error(self, error: Exception) -> str:
        text = str(error)
        comet = self.config.get("comet_logging", {})
        if isinstance(comet, Mapping):
            api_key_env = str(comet.get("api_key_env", "COMET_API_KEY"))
            api_key = str(comet.get("api_key", "")) or os.environ.get(api_key_env, "")
            if api_key:
                text = text.replace(api_key, "<redacted>")
        return text

    def _write_state(self, state: str, error: str | None = None) -> None:
        comet = self.config.get("comet_logging", {})
        experiment = self.config.get("experiment", {"name": self.config.get("name", "NewQE")})
        state_path = self.directory / "comet.json"
        payload: dict[str, Any] = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else {}
        )
        payload.update({
            "status": state,
            "experiment_name": str(experiment["name"]),
            "experiment_key": str(comet.get("experiment_key", "")),
            "project_name": str(comet.get("project_name", "")),
            "workspace": str(comet.get("workspace", "")),
        })
        if error is not None:
            payload["error"] = error
        else:
            payload.pop("error", None)
        atomic_write_json(state_path, payload)

    def _initialize(self) -> None:
        comet = self.config.get("comet_logging", {})
        if not isinstance(comet, Mapping) or not bool(comet.get("using", False)):
            return
        self._resolve_key()
        try:
            api_key = (
                str(comet.get("api_key", ""))
                or os.environ.get(str(comet.get("api_key_env", "COMET_API_KEY")), "")
            )
            online = bool(comet.get("online", True))
            if online and not api_key:
                raise ValueError(
                    "Set COMET_API_KEY through Kaggle Secrets "
                    "or the notebook key prompt"
                )
            import comet_ml

            # Only values explicitly logged below belong to the experiment.  In
            # particular, keep the repository source, git diff, host details and
            # captured console output local; the Comet credential is resolved only from the configured environment
            # variable and must never be swept up by automatic logging.
            comet_experiment_config = comet_ml.ExperimentConfig(
                log_code=False,
                log_graph=False,
                parse_args=False,
                display_summary_level=0,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_details=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                auto_output_logging=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_log_co2=False,
                offline_directory=str(self.directory / "comet-offline"),
            )
            self._experiment = comet_ml.start(
                api_key=api_key or None,
                project_name=str(comet["project_name"]),
                workspace=str(comet["workspace"]),
                experiment_key=str(comet["experiment_key"]),
                mode="get_or_create",
                online=online,
                experiment_config=comet_experiment_config,
            )
            self._experiment.set_name(str(self.config.get("experiment", {"name": self.config.get("name", "NewQE")})["name"]))
            self._experiment.log_parameters(redact_secrets(self.config))
            self.enabled = True
            self._write_state("active")
            if online:
                print(f"Comet: {self._experiment.get_url()}", flush=True)
        except Exception as error:  # Comet failures must not stop GPU training.
            safe_error = self._safe_error(error)
            print(f"Warning: Comet disabled: {safe_error}", flush=True)
            self._write_state("failed", safe_error)
            self._experiment = None
            self.enabled = False

    def log_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        step: int,
    ) -> None:
        """Log numeric metrics, disabling Comet after the first failure."""
        if not self.enabled or self._experiment is None:
            return
        try:
            values = {
                f"{phase}/{name}": float(value)
                for name, value in metrics.items()
                if name not in {"iteration", "elapsed_seconds"}
                and isinstance(value, (int, float))
            }
            self._experiment.log_metrics(values, step=step)
        except Exception as error:  # See class fail-soft contract.
            safe_error = self._safe_error(error)
            print(f"Warning: Comet disabled: {safe_error}", flush=True)
            self._write_state("failed", safe_error)
            self.enabled = False

    def close(self) -> None:
        """Finish the active Comet session without propagating service errors."""
        if self._experiment is None:
            return
        try:
            self._experiment.end()
            if self.enabled:
                self._write_state("closed")
        except Exception as error:  # See class fail-soft contract.
            safe_error = self._safe_error(error)
            print(f"Warning: failed to close Comet: {safe_error}", flush=True)
            self._write_state("failed", safe_error)
        finally:
            self.enabled = False
            self._experiment = None
