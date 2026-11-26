# Enhance for Human automation entry point

Read `.agents/AGENTS.md` and `scripts/AGENTS.md` before changes.
Kaggle training calls scripts/train_mfqev2_window.py directly with one
configs/kaggle/QP*.yml config per notebook. No orchestration service or approval
CLI is required. Keep the selected QP's config and source fixed when resuming.
Check runtime and run bounded diagnostics before changing training behavior.
Preserve shared Enhance_for_Machine/MDFI sources, datasets and checkpoints.
Shared dataset links are read-only; writes belong to Human cache/run/output roots.
Smoke metrics are diagnostics, never research results. Verify artifacts for claims.
Do not store credentials in source, configuration or reports.
