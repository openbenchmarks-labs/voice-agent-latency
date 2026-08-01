"""Measure every vendor at once, one question set at a time.

    python -m harness.sync_bench --calls 100
    python -m harness.sync_bench --vendors telnyx,vapi --calls 10

WHY THIS EXISTS. The sequential bench measures Telnyx at 10:00 and Retell at
11:40. Anything that moved in between -- carrier load, a platform's own capacity,
our network, time of day -- lands on one vendor and not the other, and there is
no way afterwards to tell that apart from a platform being slower. Dialling every
vendor within the same second with the SAME three questions removes the confound
by construction rather than by argument.

HOW. For question set i, place one call per vendor concurrently, wait for all of
them to finish, then move to set i+1. Sets stay in lockstep, so vendor A's set 7
and vendor B's set 7 happened at the same moment. Calls inside a set are
concurrent; sets are strictly sequential.

WHAT IT WRITES. One ordinary run directory per vendor -- `runs/bench-<vendor>-<stamp>/`
exactly as the sequential bench produces, so the analyzer, the exporter and
`reanalyze_run.py` all work on them unchanged. Plus one coordinator directory,
`runs/sync-<stamp>/`, holding `sets.jsonl`: per set, per vendor, when each call
was placed and when it ended, and how far apart the placements were. That file is
the evidence that the calls really were simultaneous; without it "we ran them
together" is a claim rather than a record.

CONCURRENCY. Every webhook URL carries its call's token (see Registry in
bench.py), so the carrier's POSTs stay attributable with several calls live. The
one thing that must never appear on this path is `carrier.hangup_all()` -- it
would kill the other vendors' in-flight calls. `harness/onecall.py` uses per-call
hangup for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from analyzer.measure import analyze_run
from carriers import get_carrier
from vendors import get_vendor
from vendors.base import stack_summary
from vendors.registry import load_vendor_config, spec_from_config

from .bench import MODE, Registry, build_bench_app, finalize_run
from .config import settings
from .dialog import build_run_scripts, channel_map_is_provisional, run_plan_receipt
from .onecall import place_and_measure
from .serving import add_server_args, describe, server_kwargs, server_problems

log = logging.getLogger(__name__)

ALL_VENDORS = ("telnyx", "vapi", "retell", "bland", "elevenlabs")

#: Breather between SETS, not between calls. Placing the next set instantly after
#: the previous one would have some platforms still tearing down the last call.
PAUSE_BETWEEN_SETS_S = 3.0


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds")


class Vendor:
    """One vendor's whole run: its target, receipts, scripts and results."""

    def __init__(self, slug: str, *, calls: int, cases: list[str] | None,
                 skip_verify: bool):
        self.slug = slug
        self.adapter = get_vendor(slug)
        self.spec = spec_from_config(load_vendor_config(slug))
        self.problems = self.adapter.verify_agent(self.spec)
        self.skip_verify = skip_verify
        self.target = None if self.problems and not skip_verify else \
            self.adapter.dial_target().value
        self.applied = self.adapter.applied_config().as_dict()
        self.scripts = build_run_scripts(calls, cases)
        self.caller_receipt = run_plan_receipt(self.scripts)
        self.results: list = []
        self.run_id = ""
        self.run_dir: Path | None = None

    def prepare(self, stamp: str) -> None:
        self.run_id = f"bench-{self.slug}-{stamp}"
        self.run_dir = settings.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "applied_config.json").write_text(
            json.dumps(self.applied, indent=2, default=str) + "\n")
        (self.run_dir / "caller_config.json").write_text(
            json.dumps(self.caller_receipt, indent=2, default=str) + "\n")

    def attach(self, run_id: str) -> list[str]:
        """Point at an EXISTING run instead of starting one. Returns refusals.

        A repaired call lands beside calls measured hours earlier and is pooled
        with them, so it is only honest if it was asked the same questions under
        the same vendor configuration. Both are receipts already on disk, so
        both are checked rather than assumed -- and a mismatch refuses the
        repair rather than warning about it, because the damage (a run whose
        calls silently disagree about what they measured) is invisible
        afterwards and permanent.
        """
        self.run_id = run_id
        self.run_dir = settings.runs_dir / run_id
        problems: list[str] = []
        if not self.run_dir.is_dir():
            return [f"{run_id} does not exist"]

        stored_caller = json.loads(
            (self.run_dir / "caller_config.json").read_text())
        if stored_caller["sha256"] != self.caller_receipt["sha256"]:
            problems.append(
                f"caller plan is {self.caller_receipt['sha256'][:12]}, the run "
                f"was measured with {stored_caller['sha256'][:12]} -- the "
                f"question rotation changed, so set N is no longer set N")

        stored_applied = json.loads(
            (self.run_dir / "applied_config.json").read_text())
        if stored_applied["sha256"] != self.applied["sha256"]:
            problems.append(
                f"vendor config is {self.applied['sha256'][:12]}, the run was "
                f"measured with {stored_applied['sha256'][:12]} -- the repaired "
                f"calls would not be the same instrument as their siblings")
        return problems


def run_set(vendors: list[Vendor], index: int, carrier, registry,
            gate: threading.Barrier, *, stamp: str = "",
            rotation: int = 1) -> list[dict]:
    """One question set across every vendor, concurrently. Returns per-vendor rows."""
    rows: list[dict] = []
    lock = threading.Lock()

    def one(v: Vendor) -> None:
        call_id = f"call-{index:03d}"
        row = {"vendor": v.slug, "call_id": call_id}
        try:
            # Every thread waits here, so the dial itself is what happens
            # simultaneously -- not the setup before it. Without the barrier the
            # first vendor is dialled while the last is still resolving.
            gate.wait(timeout=60.0)
            result, call = place_and_measure(
                call_id=call_id, run_dir=v.run_dir, script=v.scripts[index],
                vendor=v.slug, run_id=v.run_id, target=v.target,
                carrier=carrier, registry=registry, applied=v.applied,
                caller_receipt=v.caller_receipt, mode=MODE,
                verify_skipped=bool(v.problems and v.skip_verify),
                # set_index, not call_index: with a 50-set rotation and 100
                # calls, call 7 and call 57 ask the same questions, and
                # comparing platforms means grouping by the set.
                extra={"set_index": index % rotation if rotation else index,
                       "sync_session": stamp},
            )
            row.update(call.timing())
            if result is None:
                row["outcome"] = "not_answered"
            else:
                v.results.append(result)
                row["outcome"] = ("usable" if result.usable
                                  else f"discarded:{result.discard_reason}")
                row["ttfab_onset_ms"] = result.ttfab_onset_ms
        except threading.BrokenBarrierError:
            row["outcome"] = "aborted:barrier"
        except Exception as exc:  # noqa: BLE001 -- one vendor must not stop the set
            row["outcome"] = f"failed:{type(exc).__name__}"
            row["error"] = str(exc)[:300]
            log.warning("%s set %d failed: %s", v.slug, index, exc)
        with lock:
            rows.append(row)

    with ThreadPoolExecutor(max_workers=len(vendors)) as pool:
        list(pool.map(one, vendors))
    rows.sort(key=lambda r: r["vendor"])
    return rows


def supersede(run_dir: Path, call_id: str, stamp: str) -> Path | None:
    """Move a call out of the run before its slot is re-dialled.

    Moved, never deleted: the old recording is the evidence for whatever went
    wrong, and a repair that destroys it makes the failure unauditable.

    The destination is OUTSIDE the run directory on purpose. `analyze_run` treats
    every subdirectory of a run as a call, so a `call-018.old` left in place
    would be measured as an extra call and quietly join the percentiles.
    """
    src = run_dir / call_id
    if not src.exists():
        return None
    dest = settings.runs_dir / "superseded" / run_dir.name / f"{call_id}-{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)
    return dest


def run_repair(args, sets: list[int]) -> int:
    """Re-dial specific question sets into an existing run, in place.

    The alternative -- a fresh run of just the broken sets -- produces a second
    run directory whose calls have to be merged by hand into the first, and the
    merge is where the mistakes live. Here the slot keeps its identity: set 18's
    repair is call-018, in the same run, asked the same questions, and the
    rebuilt report is derived from every call on disk rather than from the
    handful just re-dialled.
    """
    # Accepts a path or a bare run id, so the value a user copies out of the
    # console resolves whichever directory they are standing in.
    coord = Path(args.repair)
    if not coord.is_dir():
        coord = settings.runs_dir / Path(args.repair).name
    manifest_path = coord / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path} -- is that a coordinator dir?")
        return 2
    manifest = json.loads(manifest_path.read_text())

    calls = manifest["calls_per_vendor"]
    bad = [i for i in sets if not 0 <= i < calls]
    if bad:
        print(f"sets out of range for a {calls}-call run: {bad}")
        return 2

    wanted = [s.strip() for s in args.vendors.split(",") if s.strip()]
    slugs = [s for s in manifest["vendors"] if s in wanted]
    missing = [s for s in wanted if s not in manifest["vendors"]]
    if missing and args.vendors != ",".join(ALL_VENDORS):
        print(f"not part of {coord.name}: {', '.join(missing)}")
        return 2
    if not slugs:
        print("no vendors selected that are part of this run")
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"REPAIR {coord.name}: sets {', '.join(str(i) for i in sets)} "
          f"x {len(slugs)} vendors = {len(sets) * len(slugs)} calls")
    print(f"caller plan on record: {manifest['caller_plan_sha256'][:16]}\n")

    vendors: list[Vendor] = []
    for slug in slugs:
        try:
            v = Vendor(slug, calls=calls, cases=None,
                       skip_verify=args.skip_verify)
        except Exception as exc:  # noqa: BLE001
            print(f"  {slug:11} UNAVAILABLE  {type(exc).__name__}: {str(exc)[:90]}")
            return 2
        refusals = v.attach(manifest["vendors"][slug]["run_id"])
        if v.problems and not args.skip_verify:
            refusals += [str(p)[:110] for p in v.problems]
        if refusals:
            print(f"  {slug:11} REFUSED")
            for r in refusals:
                print(f"      - {r}")
            return 2
        print(f"  {slug:11} ready        {v.run_dir.name}  "
              f"receipt {v.applied['sha256'][:12]}")
        vendors.append(v)

    # Every check has passed for every vendor before anything is moved: a repair
    # that refuses halfway leaves the run in a state nobody planned.
    rotation = vendors[0].caller_receipt["rotation_length"]
    for v in vendors:
        for i in sets:
            moved = supersede(v.run_dir, f"call-{i:03d}", stamp)
            if moved:
                print(f"  {v.slug:11} set {i:>3} -> {moved.parent.name}/{moved.name}")

    registry = Registry()
    app = build_bench_app(registry)
    config = uvicorn.Config(app, **server_kwargs(args))
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.5)
    print(f"\nserving {describe(args)}\n")

    records: list[dict] = []
    try:
        for n, i in enumerate(sets, start=1):
            cases_i = [t.case_id for t in vendors[0].scripts[i].turns if t.case_id]
            print(f"[repair {n}/{len(sets)}] set {i}: {', '.join(cases_i)}")
            gate = threading.Barrier(len(vendors))
            t0 = time.time()
            rows = run_set(vendors, i, get_carrier(), registry, gate,
                           stamp=stamp, rotation=rotation)
            placed = [r["placed_at_epoch"] for r in rows
                      if r.get("placed_at_epoch") is not None]
            spread = round(max(placed) - min(placed), 3) if len(placed) > 1 else None
            records.append({
                "set_index": i,
                "call_id": f"call-{i:03d}",
                "cases": cases_i,
                "set_started_at": iso(t0),
                "set_ended_at": iso(time.time()),
                "placement_spread_s": spread,
                "repair_of": manifest["stamp"],
                # Which vendors were re-dialled TOGETHER. Repairing a subset is
                # legitimate, but those calls are then simultaneous only with
                # each other -- not with the siblings they are pooled beside.
                "simultaneous_with": sorted(v.slug for v in vendors),
                "vendors": rows,
            })
            for r in rows:
                ttfab = r.get("ttfab_onset_ms")
                print(f"    {r['vendor']:11} {r['outcome']:22}"
                      + (f" ttfab={ttfab:.0f}ms" if ttfab is not None else ""))
            if spread is not None:
                print(f"    placements within {spread * 1000:.0f} ms")
            if n < len(sets):
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted -- rebuilding from whatever is on disk")
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    with (coord / "repairs.jsonl").open("a") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    # Rebuilt from EVERY call in the run, not from the handful just re-dialled.
    # finalize_run(v.results) would silently rewrite bench.json as if the run
    # were six calls long.
    print("\nrebuilding reports from all calls on disk ...")
    for v in vendors:
        results = analyze_run(v.run_dir, write=True)
        finalize_run(v.run_dir, results, v.slug, v.applied, v.caller_receipt)
        print(f"  {v.slug:11} {len(results)} calls -> {v.run_dir / 'bench.json'}")

    print(f"\nrepair record: {coord / 'repairs.jsonl'}")
    print("\nRun tools/backfill_costs.py to price these calls, and "
          "tools/export_snapshot.py to publish them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vendors", default=",".join(ALL_VENDORS),
                        help=f"comma-separated (default all: {','.join(ALL_VENDORS)})")
    parser.add_argument("--calls", type=int, default=10,
                        help="calls PER VENDOR. Call i asks question set "
                             "i mod rotation_length, for every vendor.")
    parser.add_argument("--cases", default=None,
                        help="comma-separated case ids; makes every call "
                             "identical (debugging)")
    parser.add_argument("--pause", type=float, default=PAUSE_BETWEEN_SETS_S,
                        help="seconds between SETS")
    parser.add_argument("--port", type=int, default=8000)
    add_server_args(parser)
    parser.add_argument("--skip-verify", action="store_true",
                        help="bench a config that differs from vendors.yaml "
                             "(recorded in every report as unverified)")
    parser.add_argument("--repair", metavar="COORD_DIR", default=None,
                        help="re-dial specific sets INTO an existing run "
                             "(runs/sync-*), replacing those calls in place and "
                             "rebuilding its reports. Use with --sets.")
    parser.add_argument("--sets", default=None,
                        help="comma-separated set indices to repair, e.g. "
                             "18,61,62. Required with --repair.")
    args = parser.parse_args()

    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(message)s")

    if bool(args.repair) != bool(args.sets):
        print("--repair and --sets go together: --repair says which run, "
              "--sets says which calls in it")
        return 2

    exposure = server_problems(args)
    if exposure:
        print("SERVER CONFIG FAILED -- webhooks would never reach us:")
        for problem in exposure:
            print(f"  - {problem}")
        return 2
    settings.require_carrier()

    if args.repair:
        try:
            # Sorted and de-duplicated: the sets are dialled in order so the log
            # reads chronologically, and asking for 18 twice would supersede the
            # first repair with the second.
            sets = sorted({int(s) for s in args.sets.split(",") if s.strip()})
        except ValueError:
            print(f"--sets wants integers, got {args.sets!r}")
            return 2
        if not sets:
            print("--sets is empty -- nothing to repair")
            return 2
        return run_repair(args, sets)

    slugs = [s.strip() for s in args.vendors.split(",") if s.strip()]
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else None

    print("resolving vendors ...")
    vendors: list[Vendor] = []
    for slug in slugs:
        try:
            v = Vendor(slug, calls=args.calls, cases=cases,
                       skip_verify=args.skip_verify)
        except Exception as exc:  # noqa: BLE001
            print(f"  {slug:11} UNAVAILABLE  {type(exc).__name__}: {str(exc)[:90]}")
            continue
        if v.problems and not args.skip_verify:
            print(f"  {slug:11} REFUSED      config differs from vendors.yaml:")
            for p in v.problems:
                print(f"      - {str(p)[:110]}")
            continue
        print(f"  {slug:11} ready        dials {v.target}  "
              f"receipt {v.applied['sha256'][:12]}  "
              f"model {stack_summary(v.applied.get('defaults_used') or {})['model']}")
        vendors.append(v)

    if not vendors:
        print("\nno vendors are ready -- nothing to run")
        return 2
    # A comparison needs something to compare. One vendor is the sequential
    # bench with extra machinery, and the whole point here is simultaneity.
    if len(vendors) < 2:
        print(f"\nonly {vendors[0].slug} is ready. Synchronised mode compares "
              f"vendors against each other; use `python -m harness.bench "
              f"--vendor {vendors[0].slug}` for a single vendor.")
        return 2

    rotation = vendors[0].caller_receipt["rotation_length"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for v in vendors:
        v.prepare(stamp)
    coord = settings.runs_dir / f"sync-{stamp}"
    coord.mkdir(parents=True, exist_ok=True)

    from .dialog import case_rotation
    configured = len(case_rotation())
    print(f"\nSYNCHRONISED BENCH  {len(vendors)} vendors x {args.calls} calls")
    # Two different numbers, and conflating them is how a short run looks like a
    # narrower instrument than it is: `configured` is how many sets the rotation
    # defines, `rotation` is how many this run actually reaches.
    each = args.calls // rotation if rotation else 0
    print(f"question sets: {rotation} of {configured} configured asked this run"
          + (f", each {each}x per vendor" if each > 1 else ""))
    if rotation < configured:
        print(f"  (a short run only reaches the first {rotation} sets -- "
              f"--calls {configured} covers them all once)")
    print(f"caller receipt: {vendors[0].caller_receipt['sha256'][:16]} "
          f"(identical for every vendor)")
    if channel_map_is_provisional():
        print("WARNING: stereo channel order is assumed, not probed")
    print(f"coordinator: {coord}")
    for v in vendors:
        print(f"  {v.slug:11} -> {v.run_dir}")

    registry = Registry()
    app = build_bench_app(registry)
    config = uvicorn.Config(app, **server_kwargs(args))
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.5)
    print(f"\nserving {describe(args)}\n")

    sets_path = coord / "sets.jsonl"
    try:
        for i in range(args.calls):
            cases_i = [t.case_id for t in vendors[0].scripts[i].turns if t.case_id]
            print(f"[set {i + 1}/{args.calls}] {', '.join(cases_i)}")
            # len(vendors) parties: every call thread waits, nobody dials early.
            gate = threading.Barrier(len(vendors))
            t0 = time.time()
            rows = run_set(vendors, i, get_carrier(), registry, gate,
                           stamp=stamp, rotation=rotation)

            placed = [r["placed_at_epoch"] for r in rows
                      if r.get("placed_at_epoch") is not None]
            spread = round(max(placed) - min(placed), 3) if len(placed) > 1 else None
            record = {
                "set_index": i,
                "call_id": f"call-{i:03d}",
                "cases": cases_i,
                "set_started_at": iso(t0),
                "set_ended_at": iso(time.time()),
                # How far apart the placements actually landed. This is the
                # number that says whether "simultaneous" is true.
                "placement_spread_s": spread,
                "vendors": rows,
            }
            with sets_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

            for r in rows:
                ttfab = r.get("ttfab_onset_ms")
                print(f"    {r['vendor']:11} {r['outcome']:22}"
                      + (f" ttfab={ttfab:.0f}ms" if ttfab is not None else ""))
            if spread is not None:
                print(f"    placements within {spread * 1000:.0f} ms")
            if i + 1 < args.calls:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted -- finalising what completed so far")
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    print()
    reports = {}
    for v in vendors:
        if not v.results:
            print(f"{v.slug}: no completed calls -- nothing to report")
            continue
        reports[v.slug] = finalize_run(v.run_dir, v.results, v.slug, v.applied,
                                       v.caller_receipt)

    (coord / "manifest.json").write_text(json.dumps({
        "kind": "synchronised",
        "stamp": stamp,
        "calls_per_vendor": args.calls,
        "rotation_length": rotation,
        "caller_plan_sha256": vendors[0].caller_receipt["sha256"],
        "vendors": {v.slug: {"run_id": v.run_id,
                             "vendor_config_sha256": v.applied["sha256"],
                             "calls_completed": len(v.results)}
                    for v in vendors},
        "sets_log": "sets.jsonl",
    }, indent=2) + "\n")
    print(f"\ncoordinator record: {sets_path}")
    print("\nRun tools/backfill_costs.py to price these calls, and "
          "tools/export_snapshot.py to publish them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
