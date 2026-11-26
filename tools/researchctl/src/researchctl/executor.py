"""Constrained host and Docker task execution."""

from __future__ import annotations

import fcntl
import hashlib
import os
import resource
import selectors
import shutil
import signal
import subprocess
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, interpolate, secret_values
from .contracts import check_outputs
from .journal import Journal
from .processes import start_ticks
from .sandbox import restrict_host


@dataclass(frozen=True)
class ExecutionResult:
    """Normalized terminal state for one command node."""

    state: str
    exit_code: int | None
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    failure_reason: str | None = None


def task_temp_directory(context: Mapping[str, Any]) -> Path:
    """Keep IPC socket names short, with one policy-checked directory per node."""
    output = Path(str(context["trial.output_dir"]))
    root = context.get("paths.temp")
    if root is None:
        return output / "tmp"
    identifier = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    return Path(str(root)) / f"ipc-{identifier}"


class GpuLock(AbstractContextManager):
    """Exclusive advisory lock for one host GPU."""

    def __init__(self, lock_root: Path, gpu: str) -> None:
        self.path = lock_root / f"gpu-{gpu}.lock"
        self.handle: Any = None

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"Resource is already locked: {self.path.stem}"
            ) from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class _RedactedLog:
    """Redact credential values even when a value spans pipe read boundaries."""

    def __init__(self, handle: Any, secrets: list[str]) -> None:
        self.handle = handle
        self.secrets = [value.encode() for value in secrets]
        self.pending = b""
        self.tail = max((len(value) for value in self.secrets), default=1) - 1

    def write(self, block: bytes, final: bool = False) -> None:
        self.pending += block
        # Keep a raw suffix; never split a potential secret across writes.
        limit = len(self.pending) if final else max(0, len(self.pending) - self.tail)
        for secret in self.secrets:
            start = self.pending.rfind(secret, 0, limit + len(secret) - 1)
            if 0 <= start < limit < start + len(secret):
                limit = min(limit, start)
        ready, self.pending = self.pending[:limit], self.pending[limit:]
        for secret in self.secrets:
            ready = ready.replace(secret, b"<redacted>")
        self.handle.write(ready)
        self.handle.flush()


def directory_size(path: Path) -> int:
    """Return the current recursive size of a directory in bytes."""
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _limit_resources(limits: Mapping[str, Any]) -> None:
    """Apply child-only POSIX resource limits before exec."""
    mappings = {
        "max_processes": resource.RLIMIT_NPROC,
        "max_open_files": resource.RLIMIT_NOFILE,
        "max_file_size_bytes": resource.RLIMIT_FSIZE,
        "max_address_space_bytes": resource.RLIMIT_AS,
        "max_cpu_seconds": resource.RLIMIT_CPU,
    }
    for key, kind in mappings.items():
        if key in limits and limits[key] is not None:
            value = int(limits[key])
            resource.setrlimit(kind, (value, value))


def _docker_command(
    argv: list[str], task: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    """Build a non-networked, non-root Docker command for a CPU task."""
    image = str(task.get("image", ""))
    if not image:
        raise ConfigError("Docker task must define image")
    project_root = str(context["project.root"])
    output_dir = str(context["trial.output_dir"])

    def container_path(argument: str) -> str:
        for host, container in ((output_dir, "/output"), (project_root, "/project")):
            if argument == host:
                return container
            prefix = host + os.sep
            if argument.startswith(prefix):
                return container + "/" + argument[len(prefix) :]
        return argument

    argv = [container_path(argument) for argument in argv]
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        _container_name(context),
        "--network=none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--mount",
        f"type=bind,src={project_root},dst=/project,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "--workdir",
        "/project",
        image,
        *argv,
    ]


def _container_name(context: Mapping[str, Any]) -> str:
    return (
        "researchctl-"
        + hashlib.sha256(str(context["trial.output_dir"]).encode()).hexdigest()[:20]
    )


def execute(
    *,
    node_id: str,
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    environment: Mapping[str, str],
    journal: Journal,
    limits: Mapping[str, Any],
    lock_root: Path,
    safety: Mapping[str, Any] | None = None,
    metric_contract: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Execute one task without a shell and normalize its result."""
    argv = [interpolate(str(item), context) for item in task.get("argv", [])]
    if not argv:
        raise ConfigError("Task argv must be a non-empty list")
    output_dir = Path(str(context["trial.output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    timeout = int(
        task.get("timeout_seconds", limits.get("default_timeout_seconds", 3600))
    )
    quota_gb = float(task.get("max_output_gb", limits.get("max_node_gb", 20)))
    quota_bytes = int(quota_gb * 1024**3)
    gpu_count = int(task.get("resources", {}).get("gpu_count", 0))
    gpu = str(task.get("resources", {}).get("gpu", limits.get("default_gpu", 0)))
    executor = str(task.get("executor", "host"))
    if executor == "docker":
        if gpu_count:
            raise ConfigError("GPU Docker execution is not supported in v1")
        argv = _docker_command(argv, task, context)
    elif executor != "host":
        raise ConfigError(f"Unsupported executor: {executor}")

    child_env = os.environ.copy()
    child_env.update({str(key): str(value) for key, value in environment.items()})
    temp = task_temp_directory(context)
    temp.mkdir(parents=True, exist_ok=True)
    child_env.update(
        {
            "TMPDIR": str(temp),
            "XDG_CACHE_HOME": str(temp / "cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RESEARCHCTL_MANAGED": "1",
            "RESEARCHCTL_DATASET_HASH": str(context.get("runtime.dataset_hash", "")),
            "RESEARCHCTL_COMET_ENABLED": str(
                int(bool(context.get("runtime.comet", False)))
            ),
        }
    )
    if gpu_count:
        child_env["CUDA_VISIBLE_DEVICES"] = gpu

    def child_setup() -> None:
        _limit_resources(limits.get("rlimits", {}))
        if safety is not None and executor == "host":
            roots = [
                Path(path)
                for path in context.get("runtime.writable_roots", [output_dir])
            ]
            roots.append(Path("/dev/null"))
            if gpu_count:
                roots.extend(Path("/dev").glob("nvidia*"))
            deny_network = not (
                task.get("network", "deny") == "comet"
                and context.get("runtime.comet", False)
            )
            restrict_host(
                roots,
                deny_network,
                proc_runtime=bool(
                    safety.get("filesystem", {}).get("allow_proc_runtime", False)
                ),
            )

    lock = GpuLock(lock_root, gpu) if gpu_count else nullcontext()

    with lock, stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        if gpu_count and safety is not None:
            # Advisory locks cannot detect jobs started outside researchctl.
            # When NVML works, prevent overlapping those external jobs as well.
            try:
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-compute-apps=pid,process_name",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                busy = [
                    line
                    for line in query.stdout.splitlines()
                    if line.strip() and "gnome-remote-desktop" not in line
                ]
                if query.returncode == 0 and busy:
                    return ExecutionResult(
                        "FAILED", None, {}, {}, "GPU has external compute jobs"
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass  # CUDA preflight remains authoritative when NVML is broken.
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(context["project.root"]),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=child_setup,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ExecutionResult(
                "FAILED", None, {}, {}, f"Cannot start isolated task: {exc}"
            )
        pgid = os.getpgid(process.pid)
        journal.transition(
            node_id,
            "RUNNING",
            pid=process.pid,
            pgid=pgid,
            process_start_ticks=start_ticks(process.pid),
        )
        started = time.monotonic()
        terminal_state = None
        failure_reason = None
        stopped_at = None
        next_budget_check = 0.0
        requested: list[int] = []

        def request_stop(signum: int, frame: Any) -> None:
            del frame
            requested.append(signum)

        handlers = {
            s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)
        }
        streams = selectors.DefaultSelector()
        secrets = secret_values(child_env)
        logs = {
            process.stdout: _RedactedLog(stdout, secrets),
            process.stderr: _RedactedLog(stderr, secrets),
        }
        for stream in logs:
            streams.register(stream, selectors.EVENT_READ)
        try:
            while process.poll() is None or streams.get_map():
                for key, _ in streams.select(timeout=0.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    logs[key.fileobj].write(block, final=not block)
                    if not block:
                        streams.unregister(key.fileobj)
                        key.fileobj.close()
                now = time.monotonic()
                row = journal.node(node_id)
                if requested or (row and row["state"] == "CANCELLED"):
                    terminal_state = "CANCELLED"
                    failure_reason = "Cancelled by operator"
                elif terminal_state is None and now - started > timeout:
                    terminal_state = "TIMED_OUT"
                    failure_reason = f"Task exceeded timeout of {timeout} seconds"
                if terminal_state is None and now >= next_budget_check:
                    next_budget_check = now + 1
                    size = sum(
                        directory_size(Path(root))
                        for root in context.get("runtime.quota_roots", [output_dir])
                    )
                    minimum = float(limits.get("stop_before_disk_full_gb", 0))
                    if size > quota_bytes:
                        terminal_state = "QUOTA_EXCEEDED"
                        failure_reason = (
                            f"Task exceeded output quota of {quota_gb:g} GiB"
                        )
                    elif shutil.disk_usage(output_dir).free < minimum * 1024**3:
                        terminal_state = "QUOTA_EXCEEDED"
                        failure_reason = "Stopped before filling the filesystem"
                if terminal_state and stopped_at is None:
                    stopped_at = now
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                grace = float(task.get("shutdown_grace_seconds", 120))
                if stopped_at is not None and now - stopped_at > grace:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Do not leave descendants holding the log pipes after the
                # task leader exits (including an interrupted adapter).
                if (
                    process.poll() is not None
                    and streams.get_map()
                    and stopped_at is None
                ):
                    stopped_at = now
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        finally:
            streams.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
            process.wait()
            if executor == "docker":
                subprocess.run(
                    ["docker", "rm", "-f", _container_name(context)],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
        return_code = process.wait()

    artifacts = {"stdout": str(stdout_path), "stderr": str(stderr_path)}
    if terminal_state:
        return ExecutionResult(
            terminal_state, return_code, {}, artifacts, failure_reason
        )
    if return_code != 0:
        return ExecutionResult(
            "FAILED", return_code, {}, artifacts, f"Command exited with {return_code}"
        )
    try:
        metrics, collected = check_outputs(task, context, metric_contract)
        artifacts.update(collected)
    except (ConfigError, OSError, KeyError, ValueError) as error:
        return ExecutionResult("FAILED", return_code, {}, artifacts, str(error))
    return ExecutionResult("COMPLETE", return_code, metrics, artifacts)
