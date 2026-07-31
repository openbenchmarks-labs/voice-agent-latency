#!/usr/bin/env python3
"""Fetch what each stored run cost, from the vendors' own billing APIs.

A TOOL, not part of the bench. It reads run directories and vendor accounts and
writes costs.json into each run; it never dials, never changes a vendor's
configuration, and never touches the latency numbers.

Why a separate pass at all, when the bench collects costs live: billing lags.
Vapi and Bland post a price within seconds, Telnyx's detail record appears a
little later, and a run measured while a platform was still settling ends up
with unmatched calls. Re-running this is free and idempotent -- it re-reads the
window, re-matches, and rewrites the manifest.

Usage:
    .venv/bin/python tools/backfill_costs.py --all --dry-run
    .venv/bin/python tools/backfill_costs.py --run runs/bench-vapi-...
    .venv/bin/python tools/backfill_costs.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

from harness import costs as C                            # noqa: E402
from harness.config import settings                       # noqa: E402
from vendors import get_vendor                            # noqa: E402

def vendor_of(run_dir: Path) -> str | None:
    bench = run_dir / "bench.json"
    if bench.is_file():
        return (json.loads(bench.read_text()) or {}).get("vendor")
    return run_dir.name.split("-")[1] if "-" in run_dir.name else None


def money(value, currency: str | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f} {currency or ''}".strip()


def report(payload: dict) -> None:
    summary = payload["summary"]
    print(f"  matched {summary['calls_matched']} call(s), "
          f"priced {summary['calls_priced']}")
    if summary["calls_priced"]:
        print(f"  total {money(summary['total_cost'], summary['currency'])} "
              f"over {summary['total_duration_s']:.0f}s "
              f"-> {money(summary['cost_per_minute'], summary['currency'])}/min")
        if (summary["cost_per_billed_minute"]
                and summary["cost_per_billed_minute"] != summary["cost_per_minute"]):
            print(f"  billed {summary['total_billed_s']:.0f}s "
                  f"-> {money(summary['cost_per_billed_minute'], summary['currency'])}"
                  f"/billed-min")
    if payload["unmatched_calls"]:
        print(f"  UNMATCHED: {', '.join(payload['unmatched_calls'])} "
              f"(billing lag, or the vendor record is outside the window)")
    for note in summary["notes"]:
        print(f"  note: {note}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], metavar="DIR",
                        help="Run directory to price (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help=f"Every bench-* run under {settings.runs_dir}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and match, but write no manifest")
    args = parser.parse_args()

    if args.all:
        run_dirs = sorted(p for p in settings.runs_dir.glob("bench-*") if p.is_dir())
    else:
        run_dirs = [Path(r).resolve() for r in args.run]
    if not run_dirs:
        return parser.error("nothing to do: pass --run DIR or --all")

    caller = settings.plivo_from_number
    if not caller:
        print("warning: PLIVO_FROM_NUMBER is unset, so matches cannot be "
              "confirmed by caller id -- pairing on time alone")

    written = []
    for run_dir in run_dirs:
        vendor = vendor_of(run_dir)
        print(f"\n{run_dir.name} [{vendor}]")
        if not vendor:
            print("  no vendor in bench.json -- skipped")
            continue
        try:
            adapter = get_vendor(vendor)
        except Exception as exc:  # noqa: BLE001 -- one bad vendor must not stop the sweep
            print(f"  cannot load the {vendor} adapter: {type(exc).__name__}: {exc}")
            continue
        if not hasattr(adapter, "call_costs"):
            print(f"  the {vendor} adapter reports no costs")
            continue
        try:
            payload = C.collect(vendor, adapter, run_dir, caller)
        except Exception as exc:  # noqa: BLE001 -- billing APIs fail independently
            print(f"  billing API failed: {type(exc).__name__}: {exc}")
            continue

        report(payload)
        if not args.dry_run:
            C.write_manifest(run_dir, payload)
            written.append(run_dir)

    if args.dry_run:
        print("\n--dry-run: no manifests written")
        return 0
    print(f"\n{C.MANIFEST_NAME} written for {len(written)} run(s)")
    if written:
        print("Next: `tools/export_snapshot.py` carries these into the "
              "per-call artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
