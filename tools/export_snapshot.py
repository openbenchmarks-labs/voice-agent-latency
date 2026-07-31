"""Export bench runs into the publishable data/ layout.

Turns runner output (runs/<run_id>/bench.json + call-*/), which is gitignored
and host-local, into the committed per-call audit trail:

    data/latest-voice.json               one entry per vendor (newest run wins)
    data/voice-runs/<run_id>/bench.json  the full report, verbatim
    data/voice-runs/<run_id>/call-XXX.json  slim per-call audit artifacts

Audio WAVs are never copied into data/. Where a run's recordings have been
published to object storage, each slim artifact gains a `recording` block
(public URL + sha256) alongside the `recording_available: true` flag; where they
have not, the run exports exactly the same, minus the links.

Usage (from the repo root):

    python tools/export_snapshot.py \
        --run runs/bench-telnyx-20260729-182644

Runs live wherever the harness ran. Either run this there and copy the two
data/ outputs over, or copy the run directory first -- the exporter only reads
JSON, never audio.

The instrument block comes from bench.json's own `instrument` key when the run
recorded one, else stated as absent. It is NEVER derived from a
raw sweep instrument.json: the raw intercept contains the rig's own VAD
hangover, and the decomposition into path overhead is a vetted judgement with
provenance (see harness/instrument.py), not arithmetic to redo here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run directly (`python tools/export_snapshot.py`) the repo root is not on the
# path, so `harness` would not import. Every other tool here does the same.
# No venv guard: this reads and writes JSON only, and requiring the analyzer's
# numpy/onnxruntime stack would refuse to run where exporting works fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import instrument                            # noqa: E402
from harness.config import data_root                      # noqa: E402

# The repo's own data/ directory -- the same resolver the scenarios use.
# `--data-dir` overrides it, which is how you export somewhere else without
# touching the committed audit trail.
DATA_DIR = data_root() / "data"
# The published dataset's identity. `--dataset-slug` / `--dataset-name` override
# it, which is what you want when exporting your own runs rather than adding to
# this board -- otherwise your snapshot claims to be part of ours.
DATASET = {"slug": "voice-2026-q3",
           "name": "OpenBenchmarks Voice Agent Latency - 2026 Q3"}

# Display names for vendor slugs coming out of bench.json.
#
# These are not cosmetic. The name travels into `providers.name` on the first
# backfill, and `ensure_provider` matches on slug -- so whatever lands there
# first is what the public board shows. A slug missing from this map falls back
# to `.title()`, which turns "vapi" into "Vapi" and "elevenlabs" into
# "Elevenlabs". Add a vendor here in the same commit that adds its adapter.
VENDOR_NAMES = {
    "telnyx": "Telnyx",
    "vapi": "Vapi",
    "retell": "Retell AI",
    "bland": "Bland AI",
    "elevenlabs": "ElevenLabs",
}

RUN_ID_STAMP = re.compile(r"-(\d{8})-(\d{6})$")


def run_started_at(run_id: str) -> str | None:
    """The run id embeds its start time (bench-<vendor>-YYYYMMDD-HHMMSS)."""
    m = RUN_ID_STAMP.search(run_id)
    if not m:
        return None
    stamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return stamp.replace(tzinfo=timezone.utc).isoformat()


def instrument_block(report: dict) -> dict:
    """What characterised this run -- or, honestly, that nothing did.

    There is no fallback value on purpose. No characterisation exists for the
    current measurement path, so a run that recorded none publishes the absence
    rather than a borrowed figure that would look reassuring and be wrong.
    """
    if report.get("instrument"):
        return dict(report["instrument"])
    gap = instrument.SELF_REPORTED_LATENCY_GAP
    return {
        "valid": False,
        "host": None,
        "sweep_run": None,
        "noise_sd_ms": None,
        "slope": None,
        "slope_gate_passed": False,
        "path_overhead_ms": None,
        "path_overhead_subtracted": False,
        "note": "no delay-sweep characterisation exists for this measurement "
                "path; recording-path overhead and instrument noise are "
                "unmeasured here",
        "self_reported_latency_gap_ms": gap.floor_ms,
        "self_reported_latency_gap_run": gap.bench_run,
    }


def two_tap_findings() -> list[dict]:
    """The self-reported-latency evidence, attributed to the platform whose run
    produced it (the bench_run id names the platform)."""
    gap = instrument.SELF_REPORTED_LATENCY_GAP
    m = re.match(r"bench-([a-z0-9]+)-", gap.bench_run)
    if not m:
        return []
    return [{
        "provider_slug": m.group(1),
        "bench_run": gap.bench_run,
        "n_turns": gap.n_turns,
        "floor_ms": gap.floor_ms,
        "floor_sd_ms": gap.floor_sd_ms,
        "note": gap.note,
    }]


def recordings_manifest(run_dir: Path) -> dict:
    """Published recording URLs, keyed by call id.

    Written by whatever published the run's audio, which is a separate step and
    may not have happened at all -- an unpublished run exports exactly as it
    did before, without the links.
    """
    path = run_dir / "recordings.json"
    if not path.is_file():
        return {}
    return (json.loads(path.read_text()) or {}).get("calls") or {}


def costs_manifest(run_dir: Path) -> dict:
    """Per-call billing figures, keyed by call id.

    Written by tools/backfill_costs.py, which runs separately and may not have
    run at all -- an unpriced run exports exactly as it did before.
    """
    path = run_dir / "costs.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()) or {}


def slim_call(call_dir: Path, published: dict | None = None,
              priced: dict | None = None) -> dict | None:
    """One call's audit artifact: measurement + provenance, no local paths."""
    result_path = call_dir / "result.json"
    if not result_path.is_file():
        return None
    slim = {
        "result": json.loads(result_path.read_text()),
        "recording_available": True,
    }
    meta_path = call_dir / "metadata.json"
    if meta_path.is_file():
        slim["metadata"] = json.loads(meta_path.read_text())
    entry = (published or {}).get(call_dir.name)
    if entry:
        # The carrier tape this call's numbers were measured from. sha256 rides
        # along so the link is checkable rather than merely clickable.
        slim["recording"] = {
            "url": entry.get("url"),
            "sha256": entry.get("sha256"),
            "bytes": entry.get("bytes"),
            "source": entry.get("source"),
        }
    cost = (priced or {}).get(call_dir.name)
    if cost:
        # The vendor's own billing figure for THIS call, with the caveats that
        # qualify it. Kept per call rather than only as a run total so a
        # disputed rate can be traced to the calls it was computed from.
        slim["cost"] = {
            "vendor_call_id": cost.get("vendor_call_id"),
            "cost": cost.get("cost"),
            "currency": cost.get("currency"),
            "duration_s": cost.get("duration_s"),
            "billed_s": cost.get("billed_s"),
            "source": cost.get("source"),
            "notes": cost.get("notes") or [],
        }
    return slim


def export_run(run_dir: Path, data_dir: Path) -> dict:
    """Write the run's artifacts under data/ and return its vendor entry."""
    report = json.loads((run_dir / "bench.json").read_text())
    run_id, vendor = report["run_id"], report["vendor"]

    out = data_dir / "voice-runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(report, indent=2) + "\n")

    published = recordings_manifest(run_dir)
    costs = costs_manifest(run_dir)
    priced = costs.get("calls") or {}
    n_calls = 0
    for call_dir in sorted(run_dir.glob("call-*")):
        slim = slim_call(call_dir, published, priced)
        if slim is None:
            continue
        (out / f"{call_dir.name}.json").write_text(
            json.dumps(slim, indent=2) + "\n")
        n_calls += 1

    carrier = None
    first_meta = next(iter(sorted(run_dir.glob("call-*/metadata.json"))), None)
    if first_meta:
        carrier = json.loads(first_meta.read_text()).get("carrier")

    return {
        "provider_slug": vendor,
        "provider_name": VENDOR_NAMES.get(vendor, vendor.title()),
        "carrier": carrier,
        "run_id": run_id,
        "started_at": run_started_at(run_id),
        "calls_exported": n_calls,
        "report": report,
        # Run-level cost summary. bench.json carries one for runs measured
        # after costs landed; costs.json is the source for everything priced
        # afterwards, and it wins because it is the fresher fetch.
        "cost": costs.get("summary") or report.get("cost"),
    }


def promotes(entry: dict, current: dict | None) -> bool:
    """Should `entry` become this vendor's snapshot entry?

    "The newest run per vendor becomes that vendor's entry" means comparing
    timestamps, not trusting export order. Re-exporting an older run -- to
    restore one, or to re-analyze it -- otherwise DEMOTES the vendor to it
    silently, and the board then shows one run's numbers under a label naming
    another. That happened on 2026-07-31, exporting a 1-call smoke test after
    the 9-call run it preceded. ISO-8601 compares lexicographically.
    """
    if current is None:
        return True
    return (entry.get("started_at") or "") >= (current.get("started_at") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, dest="runs",
                        metavar="RUN_DIR",
                        help="bench run directory (repeatable); the newest run "
                             "per vendor becomes that vendor's entry")
    parser.add_argument("--dataset-slug", default=DATASET["slug"],
                        help=f"dataset identity for your own runs "
                             f"(default {DATASET['slug']})")
    parser.add_argument("--dataset-name", default=DATASET["name"],
                        help="human-readable dataset name")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"where to write the artifacts (default {DATA_DIR})")
    args = parser.parse_args()

    dataset = {"slug": args.dataset_slug, "name": args.dataset_name}
    snapshot_path = args.data_dir / "latest-voice.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text())
        # An explicit --dataset-* on an existing snapshot is a rename, not a
        # no-op: exporting your own runs into a copy of ours should not leave it
        # claiming to be ours.
        if dataset != DATASET:
            snapshot["dataset"] = dataset
    else:
        snapshot = {"benchmark": "voice-agent-latency", "dataset": dataset,
                    "instrument": None, "vendors": {}, "runs": []}

    for run_arg in args.runs:
        run_dir = Path(run_arg)
        if not (run_dir / "bench.json").is_file():
            print(f"error: {run_dir} has no bench.json -- not a bench run "
                  f"(sweeps validate the instrument; they are not exported)",
                  file=sys.stderr)
            return 1
        entry = export_run(run_dir, args.data_dir)
        current = snapshot["vendors"].get(entry["provider_slug"])
        if promotes(entry, current):
            snapshot["vendors"][entry["provider_slug"]] = entry
        else:
            print(f"  kept {current['run_id']} as the {entry['provider_slug']} "
                  f"entry: it is newer than {entry['run_id']}")
        registry = [r for r in snapshot["runs"]
                    if r["run_id"] != entry["run_id"]]
        registry.append({
            "run_id": entry["run_id"],
            "provider_slug": entry["provider_slug"],
            "started_at": entry["started_at"],
            "attempts": entry["report"].get("attempts"),
            "usable": entry["report"].get("usable"),
            "turn_attempts": entry["report"].get("turn_attempts"),
            "turns_usable": entry["report"].get("turns_usable"),
        })
        snapshot["runs"] = sorted(registry, key=lambda r: r["run_id"])
        print(f"exported {entry['run_id']}: {entry['calls_exported']} call(s), "
              f"{entry['report'].get('turns_usable')} usable turn(s) "
              f"[{entry['provider_slug']}]")

    # The instrument is a property of a RUN, not of the snapshot. It already
    # travels inside each vendor's `report.instrument`, and the backfill reads
    # it from there. A single snapshot-level block used to be overwritten by
    # whichever run was exported last and then attached to every vendor -- so
    # vendor A's board row would cite vendor B's characterisation. Kept only as
    # a legacy hint for the first vendor exported, never as the source of truth.
    first = next(iter(snapshot["vendors"].values()), None)
    snapshot["instrument"] = instrument_block(first["report"]) if first else None

    # Merged, not regenerated: `two_tap_findings()` derives the one historical
    # Telnyx finding from a module constant, so assigning it would wipe any
    # other vendor's finding on the next export.
    existing = {(f.get("provider_slug"), f.get("bench_run")): f
                for f in snapshot.get("two_tap_findings") or []}
    for finding in two_tap_findings():
        existing[(finding.get("provider_slug"), finding.get("bench_run"))] = finding
    snapshot["two_tap_findings"] = sorted(
        existing.values(), key=lambda f: (f.get("provider_slug") or "",
                                          f.get("bench_run") or ""))
    snapshot["exported_at"] = datetime.now(timezone.utc).isoformat()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"wrote {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
