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

--dry-run measures everything and writes nothing, reporting what WOULD change:
which calls and turns change verdict, and how the published percentiles move.
An analyzer change is a change to every number derived from it, so the size of
that blast radius should be readable before it lands, not after.

Both modes compute the "after" figures with harness.build_report -- the same
aggregator the live bench uses -- so the diff cannot disagree with the report
that a real run would produce. The "before" side is the committed bench.json,
i.e. what was actually published, not a recomputation of it.

Usage:
    .venv/bin/python tools/reanalyze_run.py --dry-run runs/bench-*    # diff only
    .venv/bin/python tools/reanalyze_run.py runs/bench-bland-20260730-150314
    .venv/bin/python tools/reanalyze_run.py runs/bench-*              # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

from analyzer.measure import ANALYZER_VERSION, analyze_run  # noqa: E402
from harness.bench import build_report, finalize_run        # noqa: E402

#: A TTFAB move smaller than this is detection jitter re-run over the same audio,
#: not a consequence of the change. Reported as a count either way, but only
#: bigger moves are listed per turn -- otherwise a real regression hides inside
#: hundreds of sub-millisecond lines.
TTFAB_NOISE_MS = 0.5


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


def call_dirs(run_dir: Path) -> list[Path]:
    """The same directories, in the same order, that analyze_run walks.

    Pairing old with new by position over this listing rather than by
    `call_id`: a call nobody answered never got a metadata.json, so its
    freshly measured Result carries an empty call_id and would look like a
    different call on every run.
    """
    return sorted(p for p in run_dir.iterdir() if p.is_dir())


def old_result(call_dir: Path) -> dict | None:
    """The committed result.json for one call, or None if it was never measured.

    Read from disk rather than recomputed with the previous analyzer: the point
    of the comparison is what was PUBLISHED, and the old code is gone.
    """
    path = call_dir / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def _p(block, point: str):
    """One percentile out of a report's percentile block, tolerating absence."""
    return (block or {}).get(point)


# How many turns of this call reach the pooled percentiles. Mirrors build_report:
# a call-level discard takes every turn with it, and a turn needs both a clean
# verdict and an actual measurement.
def _measured_old(prev: dict) -> int:
    if prev.get("discard_reason") is not None:
        return 0
    return sum(1 for t in (prev.get("turns") or [])
               if not t.get("discard_reason") and t.get("ttfab_onset_ms") is not None)


def _measured_new(result) -> int:
    return sum(1 for t in result.usable_turns if t.ttfab_onset_ms is not None)


def diff_run(run_dir: Path, results: list, before: dict, after: dict) -> dict:
    """Print what changed, and return the totals for the run-level summary."""
    dirs = call_dirs(run_dir)
    previous = [old_result(d) for d in dirs]
    versions = {p.get("analyzer_version") for p in previous if p} or {"?"}
    print(f"analyzer {'/'.join(sorted(v or '?' for v in versions))} "
          f"-> {ANALYZER_VERSION}")

    call_changes: list[str] = []
    turn_verdicts: list[str] = []
    turn_moves: list[str] = []
    unmeasured = 0

    for call_dir, prev, result in zip(dirs, previous, results):
        name = call_dir.name
        if prev is None:
            # Never measured in the first place -- a call nobody answered has no
            # recording. Counting these as changes would report a diff on every
            # run forever, which is how a diff stops being read.
            unmeasured += 1
            continue

        was, now = prev.get("discard_reason"), result.discard_reason
        if was != now:
            gained = _measured_new(result) - _measured_old(prev)
            call_changes.append(
                f"  {name:<10} {was or '(kept)':<22} -> "
                f"{now or '(kept)':<22} {gained:+d} turns")

        by_index = {t.get("index"): t for t in (prev.get("turns") or [])}
        for turn in result.turns:
            old_turn = by_index.get(turn.index)
            if old_turn is None:
                continue
            if old_turn.get("discard_reason") != turn.discard_reason:
                turn_verdicts.append(
                    f"  {name}/turn {turn.index}: "
                    f"{old_turn.get('discard_reason') or '(kept)'} -> "
                    f"{turn.discard_reason or '(kept)'}")
            a, b = old_turn.get("ttfab_onset_ms"), turn.ttfab_onset_ms
            if a is not None and b is not None and abs(b - a) > TTFAB_NOISE_MS:
                turn_moves.append(
                    f"  {name}/turn {turn.index}: "
                    f"{a:.1f} -> {b:.1f} ms ({b - a:+.1f})")

    def section(title: str, lines: list[str], limit: int = 12) -> None:
        print(f"\n{title}: {len(lines)}")
        for line in lines[:limit]:
            print(line)
        if len(lines) > limit:
            print(f"  ... and {len(lines) - limit} more")

    section("calls changing verdict", call_changes)
    section("turns changing verdict", turn_verdicts)
    section(f"turns whose TTFAB moved >{TTFAB_NOISE_MS} ms", turn_moves)
    if unmeasured:
        print(f"\nnever measured (no recording, so no verdict either way): "
              f"{unmeasured}")

    rows = [
        ("usable calls", before.get("usable"), after.get("usable"), 0),
        ("usable turns", before.get("turns_usable"), after.get("turns_usable"), 0),
        ("turn attempts", before.get("turn_attempts"), after.get("turn_attempts"), 0),
        ("TTFAB p50 (ms)", _p(before.get("ttfab_onset_ms"), "p50"),
         _p(after.get("ttfab_onset_ms"), "p50"), 1),
        ("TTFAB p95 (ms)", _p(before.get("ttfab_onset_ms"), "p95"),
         _p(after.get("ttfab_onset_ms"), "p95"), 1),
    ]
    print("\npublished figures")
    for label, was, now, places in rows:
        if was is None and now is None:
            continue
        delta = ("" if was is None or now is None
                 else f"  ({now - was:+.{places}f})")
        fmt = lambda v: "—" if v is None else f"{v:.{places}f}"   # noqa: E731
        flag = "" if was == now else "   <-- moved"
        print(f"  {label:<16} {fmt(was):>9} -> {fmt(now):>9}{delta}{flag}")

    return {
        "run": run_dir.name,
        "calls_changed": len(call_changes),
        "turns_changed": len(turn_verdicts),
        "turns_moved": len(turn_moves),
        "turns_usable_before": before.get("turns_usable"),
        "turns_usable_after": after.get("turns_usable"),
        "p50_before": _p(before.get("ttfab_onset_ms"), "p50"),
        "p50_after": _p(after.get("ttfab_onset_ms"), "p50"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and diff, write nothing")
    args = parser.parse_args()

    failures = 0
    summary: list[dict] = []
    for run_dir in args.runs:
        if not run_dir.is_dir():
            print(f"not a directory: {run_dir}", file=sys.stderr)
            failures += 1
            continue

        print(f"\n{'=' * 70}\n{run_dir}  ({vendor_of(run_dir)})\n{'=' * 70}")
        applied = load_json(run_dir / "applied_config.json", "the vendor receipt")
        caller = load_json(run_dir / "caller_config.json", "the caller receipt")

        # Measured first and written second, so a dry run and a real run see
        # exactly the same numbers -- the only difference is whether they land.
        results = analyze_run(run_dir, write=False)
        if not results:
            print("no call directories with a recording -- nothing to rebuild")
            failures += 1
            continue

        before = (json.loads((run_dir / "bench.json").read_text())
                  if (run_dir / "bench.json").exists() else {})
        after = build_report(results, vendor_of(run_dir), run_dir.name, applied)
        summary.append(diff_run(run_dir, results, before, after))

        if args.dry_run:
            print("\ndry run -- nothing written")
            continue

        # Re-measure with write=True rather than serialising `results` here, so
        # the on-disk artifacts are produced by the one code path that writes
        # them and cannot drift into a second format.
        results = analyze_run(run_dir, write=True)
        finalize_run(run_dir, results, vendor_of(run_dir), applied, caller)
        print(f"\nrebuilt: {run_dir / 'bench.json'}")
        print(f"         {run_dir / 'report.html'}")

    if len(summary) > 1:
        print(f"\n{'=' * 70}\nall runs\n{'=' * 70}")
        print(f"{'run':<38} {'turns':>13} {'p50 ms':>17}")
        for row in summary:
            turns = f"{row['turns_usable_before']} -> {row['turns_usable_after']}"
            p50 = ("—" if row["p50_before"] is None or row["p50_after"] is None
                   else f"{row['p50_before']:.1f} -> {row['p50_after']:.1f}")
            print(f"{row['run']:<38} {turns:>13} {p50:>17}")
        print(f"\ncalls changing verdict: "
              f"{sum(r['calls_changed'] for r in summary)}   "
              f"turns changing verdict: "
              f"{sum(r['turns_changed'] for r in summary)}   "
              f"turns moved: {sum(r['turns_moved'] for r in summary)}")
        if args.dry_run:
            print("\ndry run -- re-run without --dry-run to apply")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
