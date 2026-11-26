"""Project, environment, storage, Docker, GPU, and secret preflight checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from .config import (
    ConfigError,
    ProjectConfig,
    command_exists,
    interpolate,
    resolve_project_path,
)
from .sandbox import check_support, restrict_host


@dataclass(frozen=True)
class Check:
    """One named preflight result."""

    name: str
    ok: bool
    detail: str
    required: bool = True


_CUDA_PROBE = """import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")
tensor = torch.ones(1, device="cuda:0")
if tensor.item() != 1:
    raise RuntimeError("CUDA computation failed")
torch.cuda.synchronize(0)
print(json.dumps({"device": torch.cuda.get_device_name(0)}))
"""


def _check_cuda(python: str, gpu: int, config: ProjectConfig | None = None) -> Check:
    """Verify the project's PyTorch can compute on the GPU used by training."""
    try:
        with tempfile.TemporaryDirectory(prefix="researchctl-cuda-probe-") as temporary:
            roots = [Path(temporary), Path("/dev/null"), Path("/dev/shm")]
            roots.extend(Path("/dev").glob("nvidia*"))
            environment = dict(os.environ)
            if config:
                environment.update(config.project["runtime"].get("environment", {}))
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "TMPDIR": temporary,
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            result = subprocess.run(
                [python, "-c", _CUDA_PROBE],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=environment,
                preexec_fn=(
                    (
                        lambda: restrict_host(
                            roots,
                            True,
                            proc_runtime=bool(
                                config.safety.get("filesystem", {}).get(
                                    "allow_proc_runtime", False
                                )
                            ),
                        )
                    )
                    if config
                    else None
                ),
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("gpu-cuda", False, f"GPU {gpu}: {exc}")
    detail = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return Check(
        "gpu-cuda", result.returncode == 0, f"GPU {gpu}: {detail or 'no output'}"
    )


def run_preflight(config: ProjectConfig) -> list[Check]:
    """Run non-mutating environment checks for configured tasks."""
    checks: list[Check] = []
    python = str(config.project["runtime"]["python"])
    checks.append(Check("python", command_exists(python), python))

    task_values = list(config.tasks["tasks"].values())
    if any(task.get("executor", "host") == "host" for task in task_values):
        try:
            checks.append(Check("host-isolation", True, check_support()))
        except (ConfigError, OSError) as exc:
            checks.append(Check("host-isolation", False, str(exc)))
    for name, task in config.tasks["tasks"].items():
        argv = task.get("argv", [])
        executable = str(argv[0]) if argv else ""
        if executable and "{" not in executable:
            checks.append(Check(f"task:{name}", command_exists(executable), executable))

    if any(str(task.get("executor", "host")) == "docker" for task in task_values):
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        checks.append(
            Check(
                "docker",
                result.returncode == 0,
                result.stderr.strip() or result.stdout.strip(),
            )
        )

    gpu_tasks = [
        task
        for task in task_values
        if int(task.get("resources", {}).get("gpu_count", 0)) > 0
    ]
    if gpu_tasks:
        default_gpu = int(config.resources["resources"].get("default_gpu", 0))
        gpu_indices = sorted(
            {
                int(task.get("resources", {}).get("gpu", default_gpu))
                for task in gpu_tasks
            }
        )
        checks.extend(_check_cuda(python, gpu, config) for gpu in gpu_indices)
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            detail = result.stderr.strip() or result.stdout.strip()
            checks.append(
                Check("gpu-nvml", result.returncode == 0, detail, required=False)
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(Check("gpu-nvml", False, str(exc), required=False))

    run_root = resolve_project_path(config.root, config.project["paths"]["runs"])
    existing = next(
        (path for path in (run_root, *run_root.parents) if path.exists()), config.root
    )
    free_gb = shutil.disk_usage(existing).free / 1024**3
    minimum = float(config.resources["resources"].get("stop_before_disk_full_gb", 20))
    checks.append(
        Check(
            "disk",
            free_gb >= minimum,
            f"{free_gb:.1f} GiB free; requires {minimum:g} GiB",
        )
    )

    comet = config.project.get("comet", {})
    if bool(comet.get("enabled", False)) and any(
        task.get("network") == "comet" for task in task_values
    ):
        variable = str(comet.get("api_key_env", "COMET_API_KEY"))
        checks.append(
            Check(
                "comet-secret",
                bool(os.environ.get(variable)),
                f"environment variable {variable}",
            )
        )

    for raw_path in config.project.get("paths", {}).get("required", []):
        path = resolve_project_path(config.root, raw_path)
        checks.append(Check(f"path:{raw_path}", path.exists(), str(path)))
    for name, task in config.tasks["tasks"].items():
        for raw_path in task.get("required_paths", []):
            context = {
                f"paths.{key}": str(resolve_project_path(config.root, value))
                for key, value in config.project["paths"].items()
                if isinstance(value, str)
            }
            path = resolve_project_path(config.root, interpolate(raw_path, context))
            checks.append(Check(f"task-input:{name}", path.exists(), str(path)))
    return checks


def require_preflight(config: ProjectConfig) -> list[Check]:
    """Run checks and raise when any mandatory check fails."""
    checks = run_preflight(config)
    failures = [check for check in checks if check.required and not check.ok]
    if failures:
        raise ConfigError(
            "Preflight failed: "
            + "; ".join(f"{check.name}: {check.detail}" for check in failures)
        )
    return checks
