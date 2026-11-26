"""Command-line interface for researchctl."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from importlib.resources import files
from collections import Counter
from pathlib import Path
from typing import Any

from . import __version__
from .credentials import (
    credential_path,
    load_credentials,
    read_credentials,
    save_comet_key,
)
from .config import (
    ConfigError,
    find_project_root,
    load_project,
    redact,
    redact_text,
    resolve_project_path,
)
from .orchestrator import (
    approve_campaign,
    cancel_campaign,
    plan_campaign,
    run_campaign,
    journal_paths,
    preflight_config,
)
from .engine import launch_campaign, reap_background
from .journal import Journal
from .preflight import run_preflight
from .processes import node_alive, start_ticks
from .validation import validate_project_contract


def parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    root = argparse.ArgumentParser(prog="researchctl")
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--project-root", type=Path)
    root.add_argument("--profile", default="local_1gpu")
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Copy the bundled .research template")
    init.add_argument("--destination", type=Path, default=Path.cwd())

    credentials = commands.add_parser(
        "credentials", help="Manage user-private credentials"
    )
    credentials.add_argument("action", choices=("set-comet", "status"))

    commands.add_parser("validate", help="Validate portable project configuration")
    preflight = commands.add_parser(
        "preflight", help="Validate runtime dependencies and resources"
    )
    preflight.add_argument("--campaign")
    for name in ("plan", "run", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", required=True)
        if name in {"run", "launch"}:
            command.add_argument("--until", help="Stop after this stage (e.g. smoke)")
    approve = commands.add_parser("approve")
    approve.add_argument("--campaign", required=True)
    approve.add_argument("--stage", default="main")
    approve.add_argument(
        "--trial", action="append", help="Explicit trial ID; repeat to select multiple"
    )
    resume = commands.add_parser("resume")
    resume.add_argument("--campaign", required=True)
    resume.add_argument("--until")
    status = commands.add_parser("status")
    status.add_argument("--campaign")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--campaign", required=True)
    return root


def _project(args: argparse.Namespace):
    root = args.project_root.resolve() if args.project_root else find_project_root()
    config = load_project(root, args.profile)
    if args.command not in {"status", "cancel"}:
        validate_project_contract(config)
    return config


def _copy_resources(source: Any, destination: Path) -> None:
    """Recursively copy a packaged Traversable resource tree."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _copy_resources(item, target)
        else:
            target.write_bytes(item.read_bytes())


def _init(destination: Path) -> int:
    target = destination.resolve() / ".research"
    if target.exists():
        raise ConfigError(f"Destination already exists: {target}")
    template = files("researchctl").joinpath("templates/research")
    _copy_resources(template, target)
    print(target)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert expected failures to concise diagnostics."""
    args = parser().parse_args(argv)
    try:
        if args.command == "credentials":
            if args.action == "set-comet":
                if not sys.stdin.isatty():
                    raise ConfigError(
                        "Use an interactive terminal to enter the API key privately"
                    )
                key = getpass.getpass("Comet API key (hidden): ")
                path = save_comet_key(key)
                print(f"Saved COMET_API_KEY to {path} (mode 0600; value hidden)")
            else:
                stored = read_credentials()
                print(
                    json.dumps(
                        {
                            "path": str(credential_path()),
                            "comet_stored": bool(stored.get("COMET_API_KEY")),
                        }
                    )
                )
            return 0
        load_credentials()
        if args.command == "init":
            return _init(args.destination)
        config = _project(args)
        if args.command == "validate":
            campaigns = []
            for path in sorted((config.config_root / "campaigns").glob("*.y*ml")):
                campaign, identity, trials, nodes = plan_campaign(config, str(path))
                campaigns.append(
                    {
                        "id": campaign["campaign"]["id"],
                        "hash": identity["campaign_hash"],
                        "trials": len(trials),
                        "nodes": len(nodes),
                    }
                )
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "project": config.project["project"]["id"],
                        "campaigns": campaigns,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "preflight":
            if args.campaign:
                campaign, _, _, nodes = plan_campaign(config, args.campaign)
                config = preflight_config(config, campaign, nodes)
            checks = run_preflight(config)
            for check in checks:
                label = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
                print(f"{label} {check.name}: {redact_text(check.detail)}")
            return 0 if all(check.ok or not check.required for check in checks) else 1
        if args.command == "plan":
            campaign, identity, trials, nodes = plan_campaign(config, args.campaign)
            print(
                redact_text(
                    json.dumps(
                        redact(
                            {
                                "campaign": campaign["campaign"]["id"],
                                "campaign_hash": identity["campaign_hash"],
                                "trials": [trial.__dict__ for trial in trials],
                                "nodes": nodes,
                            }
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
            )
            return 0
        if args.command in {"run", "resume"}:
            return run_campaign(config, args.campaign, until=args.until)
        if args.command == "launch":
            print(
                json.dumps(
                    launch_campaign(config, args.campaign, args.profile, args.until)
                )
            )
            return 0
        if args.command == "approve":
            selected = approve_campaign(config, args.campaign, args.stage, args.trial)
            print(json.dumps({"stage": args.stage, "selected_trials": selected}))
            return 0
        if args.command == "status":
            reap_background()
            path = journal_paths(config)[0]
            rows = []
            if path.exists():
                journal = Journal(path, readonly=True)
                try:
                    rows = journal.nodes(campaign_id=args.campaign)
                finally:
                    journal.close()
            for row in rows:
                if row["state"] == "RUNNING":
                    row["process_alive"] = node_alive(row)
            external = []
            for specification in config.project.get("external_runs", []):
                if args.campaign and args.campaign not in {
                    specification["id"],
                    specification.get("campaign"),
                }:
                    continue
                path = resolve_project_path(config.root, specification["status"])
                record = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"state": "not_started"}
                )
                record = {"id": specification["id"], "managed": False, **record}
                pid = record.get("pid")
                ticks = start_ticks(int(pid)) if pid else None
                record["process_alive"] = (
                    bool(ticks and ticks == record.get("process_start_ticks"))
                    if record.get("process_start_ticks")
                    else None
                )
                record["progress"] = {
                    str(path): json.loads(path.read_text())
                    for path in config.root.glob(
                        specification.get("progress_glob", "__no_progress__")
                    )
                }
                external.append(record)
            counts = Counter(str(row["state"]) for row in rows)
            print(
                redact_text(
                    json.dumps(
                        redact(
                            {"counts": counts, "nodes": rows, "external_runs": external}
                        ),
                        indent=2,
                        default=dict,
                    )
                )
            )
            return 0
        if args.command == "cancel":
            count = cancel_campaign(config, args.campaign)
            print(json.dumps({"cancelled": count}))
            return 0
    except (ConfigError, OSError, RuntimeError, KeyError, ValueError) as error:
        print(f"researchctl: {redact_text(str(error))}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
