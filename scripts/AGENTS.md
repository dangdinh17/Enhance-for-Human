# Script organization

- Keep reusable data/train/inference/evaluation/tracking/reporting pipeline entry
  points directly in `scripts/`.
- Put scripts created only for a conversation's smoke checks, diagnostics or
  one-off operations in `scripts/temp/<YYYYMMDD_task_or_run>/`. Use one folder
  per conversation test/run and group that run's helper scripts inside it.
- Keep unit/regression tests in the repository's `tests/` directory.
- Determine a script's role from callers and behavior. A filename containing
  `test` may be a shared production evaluation helper and should stay in `scripts/`.
- When relocating a script, update callers, imports, project-root resolution and
  documentation. Preserve experiment outputs and source-provenance history;
  check active queues before changing paths used by their frozen snapshots.
