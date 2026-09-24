# scripts/

One-off scripts, each runnable as `uv run python scripts/<name>.py`.
No reusable logic here: anything used twice moves into `src/watchdog_agent/`.

Planned:
- `e0_data_audit.py` (SP1): data audit, writes `results/e0_audit_v2.json`.
