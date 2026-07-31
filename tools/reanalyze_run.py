#!/usr/bin/env python3
"""Re-measure a saved run and rebuild its report, without dialling anything.

The analyzer is pure -- it reads saved audio and nothing else -- so a fix to a
measurement rule can and should be applied to runs already on disk rather than
paid for again in phone calls. Three analyzer fixes on 2026-07-30 each needed
this, and each time only `result.json` could be regenerated: `bench.json` kept
the aggregates from the original run, so the headline said one thing while the
per-call data said another.

This does the whole tail of a bench run:
  1. re-runs the analyzer over every call (rewrites result.json)
  2. rebuilds bench.json from the new results -- percentiles, turn curve,
     discard breakdown, honesty block, instrument block
  3. re-renders report.html

The receipts are read back from the run directory rather than re-fetched, so
this stays credential-free and describes the configuration that actually ran,
not whatever the vendor is set to now.

Usage:
    .venv/bin/python tools/reanalyze_run.py runs/bench-bland-20260730-150314
    .venv/bin/python tools/reanalyze_run.py runs/bench-*          # several
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

from analyzer.measure import analyze_run                  # noqa: E402
from harness.bench import finalize_run                    # noqa: E402


def vendor_of(run_dir: Path) -> str:
    """The vendor slug, from bench.json if present, else the run id."""
    bench = run_dir / "bench.json"
    if bench.exists():
        recorded = json.loads(bench.read_text()).get("vendor")
        if recorded:
            return recorded
    # bench-<vendor>-YYYYMMDD-HHMMSS
    parts = run_dir.name.split("-")
    return parts[1] if len(parts) > 2 else run_dir.name


def load_json(path: Path, what: str) -> dict:
    if not path.exists():
        sys.exit(f"{path} is missing -- {what} cannot be rebuilt without it")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN_DIR")
    args = parser.parse_args()

    failures = 0
    for run_dir in args.runs:
        if not run_dir.is_dir():
            print(f"not a directory: {run_dir}", file=sys.stderr)
            failures += 1
            continue

        print(f"\n{'=' * 70}\n{run_dir}\n{'=' * 70}")
        applied = load_json(run_dir / "applied_config.json", "the vendor receipt")
        caller = load_json(run_dir / "caller_config.json", "the caller receipt")

        results = analyze_run(run_dir, write=True)
        if not results:
            print("no call directories with a recording -- nothing to rebuild")
            failures += 1
            continue

        finalize_run(run_dir, results, vendor_of(run_dir), applied, caller)
        print(f"\nrebuilt: {run_dir / 'bench.json'}")
        print(f"         {run_dir / 'report.html'}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
