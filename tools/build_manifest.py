#!/usr/bin/env python3
"""Rebuild manifest.json — the flat index of everything published here.

One file that answers, without opening fifty others: which platforms are on the
board, what they scored, which runs produced that, and where every recording is.

Reads only `data/`, writes only `manifest.json`. No credentials, no network.
Run it after `tools/export_snapshot.py` adds or replaces a run.

Usage:
    .venv/bin/python tools/build_manifest.py
    .venv/bin/python tools/build_manifest.py --check     # CI-style: is it current?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = DATA / "voice-runs"
MANIFEST = ROOT / "manifest.json"

LIVE_BENCHMARK = "https://openbenchmarks.com/voice-agent-latency"
JSON_API = "https://openbenchmarks.com/api/benchmarks/voice-agent-latency"

SCHEMA_VERSION = 1


def build(generated_at: str) -> dict:
    snapshot_path = DATA / "latest-voice.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {}
    vendors = snapshot.get("vendors") or {}

    runs, calls = [], []
    analyzer_versions: set[str] = set()
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.is_dir() else []:
        bench_path = run_dir / "bench.json"
        bench = json.loads(bench_path.read_text()) if bench_path.is_file() else {}
        artifacts = sorted(run_dir.glob("call-*.json"))
        runs.append({
            "run_id": run_dir.name,
            "vendor": bench.get("vendor"),
            "calls": len(artifacts),
            "turns_usable": bench.get("turns_usable"),
            "turn_attempts": bench.get("turn_attempts"),
            # The two receipts that pin what was measured: the agent's live
            # config, and the caller's script and voice.
            "vendor_config_sha256": bench.get("vendor_config_sha256"),
            "caller_config_sha256": (bench.get("caller_config") or {}).get("sha256"),
        })
        for path in artifacts:
            entry = json.loads(path.read_text())
            result = entry.get("result") or {}
            if result.get("analyzer_version"):
                analyzer_versions.add(result["analyzer_version"])
            recording = entry.get("recording") or {}
            cost = entry.get("cost") or {}
            calls.append({
                "run_id": run_dir.name,
                "call_id": result.get("call_id") or path.stem,
                "artifact": str(path.relative_to(ROOT)),
                # A URL alone is clickable; the checksum is what makes it
                # checkable, and it is what tools/verify_run.py refuses on.
                "recording_url": recording.get("url"),
                "recording_sha256": recording.get("sha256"),
                "cost": cost.get("cost"),
                "cost_currency": cost.get("currency"),
            })

    leaderboard = []
    ranked = sorted(vendors.items(),
                    key=lambda kv: ((kv[1].get("report") or {})
                                    .get("ttfab_onset_ms", {}).get("p50") or 9e9))
    for slug, entry in ranked:
        report = entry.get("report") or {}
        onset = report.get("ttfab_onset_ms") or {}
        cost = entry.get("cost") or {}
        leaderboard.append({
            "rank": len(leaderboard) + 1,
            "provider_slug": slug,
            "provider_name": entry.get("provider_name"),
            "ttfab_onset_p50_ms": onset.get("p50"),
            "ttfab_onset_p95_ms": onset.get("p95"),
            "cost_per_billed_minute": cost.get("cost_per_billed_minute"),
            "cost_currency": cost.get("currency"),
            "turns_usable": report.get("turns_usable"),
            "turn_attempts": report.get("turn_attempts"),
            "latest_run_id": entry.get("run_id"),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": (snapshot.get("dataset") or {}).get("slug"),
        "dataset_name": (snapshot.get("dataset") or {}).get("name"),
        "generated_at": generated_at,
        # Which analyzer produced these numbers. Re-measuring with a different
        # one can legitimately move them, so "the same number" is only defined
        # relative to a version.
        "analyzer_versions": sorted(analyzer_versions),
        "live_benchmark": LIVE_BENCHMARK,
        "json_api": JSON_API,
        "leaderboard": leaderboard,
        "runs": runs,
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if manifest.json is out of date, "
                             "without rewriting it")
    args = parser.parse_args()

    if args.check:
        # Compare everything except the timestamp, which changes on every run and
        # would make --check permanently fail.
        current = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        fresh = build(current.get("generated_at", ""))
        if current == fresh:
            print(f"manifest.json is current "
                  f"({len(fresh['runs'])} runs, {len(fresh['calls'])} calls)")
            return 0
        print("manifest.json is STALE — run tools/build_manifest.py")
        print(f"  on disk: {len(current.get('runs') or [])} runs, "
              f"{len(current.get('calls') or [])} calls")
        print(f"  actual:  {len(fresh['runs'])} runs, {len(fresh['calls'])} calls")
        return 1

    manifest = build(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest.json: {len(manifest['leaderboard'])} platforms, "
          f"{len(manifest['runs'])} runs, {len(manifest['calls'])} calls, "
          f"analyzer {', '.join(manifest['analyzer_versions']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
