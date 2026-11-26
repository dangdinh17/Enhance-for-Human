# Enhance for Human automation entry point

Read `.agents/AGENTS.md` and `scripts/AGENTS.md` before changes.
All experiment orchestration uses installed researchctl and `.research/` contracts.
Run validate, plan, preflight and bounded smoke before explicit main approval.
Implementing a campaign does not approve it. Never invoke production trainers or
encoders outside researchctl. Use researchctl status before scheduling GPU work.
Preserve shared Enhance_for_Machine/MDFI sources, datasets and checkpoints.
Shared dataset links are read-only; writes belong to Human cache/run/output roots.
Smoke metrics are diagnostics, never research results. Verify artifacts for claims.
Do not store credentials in source, configuration or reports.
