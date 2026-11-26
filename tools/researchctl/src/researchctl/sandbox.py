"""Linux host isolation: Landlock write permissions and seccomp networking."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from typing import Iterable

from .config import ConfigError

_LIBC = ctypes.CDLL(None, use_errno=True)
_CREATE, _ADD, _RESTRICT = 444, 445, 446
_WRITE = 2 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _ArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def landlock_abi() -> int:
    """Return the supported ABI, or fail rather than silently skipping isolation."""
    abi = _LIBC.syscall(_CREATE, 0, 0, 1)
    if abi < 1:
        raise ConfigError("Host isolation requires Linux Landlock support")
    return int(abi)


def _seccomp() -> ctypes.CDLL:
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise ConfigError("Network isolation requires libseccomp.so.2") from exc
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ArgCompare),
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    return lib


def check_support() -> str:
    """Check prerequisites without restricting the calling process."""
    abi = landlock_abi()
    _seccomp()
    return f"Landlock ABI {abi}; seccomp available"


def restrict_host(
    writable_roots: Iterable[Path], deny_network: bool, proc_runtime: bool = False
) -> None:
    """Restrict this child and its descendants before executing a task.

    Reads remain available for Python, CUDA and datasets. Filesystem mutations
    are confined to declared roots. IPv4/IPv6 sockets are denied when requested;
    Unix sockets remain available for local multiprocessing and CUDA services.
    """
    abi = landlock_abi()
    access = _WRITE | (8192 if abi >= 2 else 0) | (16384 if abi >= 3 else 0)
    ruleset = _Ruleset(access)
    fd = _LIBC.syscall(_CREATE, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot create Landlock ruleset")
    try:
        for root in writable_roots:
            path_fd = os.open(str(root), os.O_PATH | os.O_CLOEXEC)
            try:
                rights = access if root.is_dir() else (2 | (16384 if abi >= 3 else 0))
                rule = _PathRule(rights, path_fd)
                if _LIBC.syscall(_ADD, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(
                        ctypes.get_errno(), f"Cannot allow writable root {root}"
                    )
            finally:
                os.close(path_fd)
        if proc_runtime:
            # NVIDIA initializes driver threads through /proc/self/task/*/comm.
            # Descendant PIDs are not known when installing this inherited rule.
            # Permit writes to existing proc runtime files, never creation,
            # removal or rename. Persistent dataset/source writes stay denied.
            proc_fd = os.open("/proc", os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathRule(2 | (16384 if abi >= 3 else 0), proc_fd)
                if _LIBC.syscall(_ADD, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "Cannot authorize proc runtime")
            finally:
                os.close(proc_fd)
        if _LIBC.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot set no_new_privs")
        if _LIBC.syscall(_RESTRICT, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot enforce Landlock")
    finally:
        os.close(fd)
    if deny_network:
        lib = _seccomp()
        ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
        if not ctx:
            raise OSError("Cannot initialize seccomp")
        try:
            syscall = lib.seccomp_syscall_resolve_name(b"socket")
            for family in (2, 10, 17):  # IPv4, IPv6, packet sockets
                comparison = _ArgCompare(0, 4, family, 0)  # SCMP_CMP_EQ
                code = lib.seccomp_rule_add_array(
                    ctx,
                    0x00050000 | errno.EPERM,
                    syscall,
                    1,
                    ctypes.byref(comparison),
                )
                if code != 0:
                    raise OSError(f"Cannot configure seccomp: {code}")
            if lib.seccomp_load(ctx) != 0:
                raise OSError("Cannot load seccomp")
        finally:
            lib.seccomp_release(ctx)
