"""Measure a vendor. The thing everything else was built for.

    python3 -m harness.bench --vendor telnyx --calls 10
    python3 -m harness.bench --vendor telnyx --calls 2 --cases price-basic,contract

Per run:
  1. resolve the vendor, GATE on verify_agent -- the bench refuses to measure an
     agent whose live config disagrees with config/vendors.yaml, because the
     published receipt has to describe the text that actually ran
  2. write applied_config.json + caller_config.json once: both halves of the
     receipt, hashed. With a live caller the stimulus is no longer a committed
     WAV, so the caller's script and voice are provenance too
  3. N sequential calls, each a scripted Plivo dialog (harness/dialog.py) over a
     stereo recording
  4. analyze each call as it lands (a discard is a result, not a retry -- discard
     rate is a headline number here)
  5. print + save the report, with an honesty block stating what is not yet
     certified about the instrument

Every reported number comes from the analyzer reading the recording. This module
places calls and collects evidence; nothing it timestamps is a
measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

from analyzer.measure import measure_call
from carriers import get_carrier
from vendors import get_vendor
from vendors.base import stack_summary
from vendors.registry import load_vendor_config, spec_from_config

from . import instrument
from .config import settings
from .dialog import (
    ANSWER_TIMEOUT_S,
    CALL_DEADLINE_S,
    DialogSession,
    build_run_scripts,
    channel_map,
    channel_map_is_provisional,
    fetch_recording,
    run_plan_receipt,
)
from .serving import (
    add_server_args,
    describe,
    server_kwargs,
    server_problems,
)

log = logging.getLogger(__name__)

PAUSE_BETWEEN_CALLS_S = 3.0

#: How this run measured. Stamped into every metadata.json; the analyzer
#: dispatches on it and the instrument has no characterisation for it.
MODE = "scripted_dialog"

# The method wants n>=100 per vendor across >=5 days and 4 time-of-day buckets before
# anything is publishable. Anything below this is a sample, and the report says so.
PUBLISHABLE_N = 100


class Registry:
    """One active call at a time (sequential bench)."""

    def __init__(self):
        self.current: DialogSession | None = None


def build_bench_app(registry: Registry) -> FastAPI:
    """The caller side of the conversation, served as Plivo XML.

    Every route authenticates: these webhooks do not merely observe the call,
    they drive it -- an unauthenticated POST to the dialog action could put
    words in our mouth or hang us up mid-measurement.
    """
    carrier = get_carrier()
    app = FastAPI(title="bench")

    @app.post("/webhooks/answer")
    async def answer(request: Request) -> Response:
        params = await carrier.verify_webhook(request)
        call = registry.current
        if call is None:
            return Response(status_code=409)
        return Response(content=call.answer_xml(params),
                        media_type="application/xml")

    @app.post("/webhooks/dialog/{token}/{step}")
    async def dialog(token: str, step: str, request: Request) -> Response:
        params = await carrier.verify_webhook(request)
        call = registry.current
        if call is None or token != call.token:
            return Response(status_code=409)
        return Response(content=call.handle_action(step, params),
                        media_type="application/xml")

    @app.post("/webhooks/hangup")
    async def hangup(request: Request) -> dict:
        params = await carrier.verify_webhook(request)
        call = registry.current
        if call is not None:
            # Keep the whole payload. Filtering it down to two fields is what
            # made a run of silent calls undiagnosable (2026-07-30): the cause,
            # the durations and the SIP detail all live in here, and by the
            # time you want them the call is long gone. Note it is `Duration`,
            # not `CallDuration` -- the latter is always absent.
            call.note_hangup(params)
        return {"ok": True}

    @app.post("/webhooks/recording")
    async def recording(request: Request) -> dict:
        params = await carrier.verify_webhook(request)
        if registry.current is not None:
            registry.current.note_recording_callback(params)
        return {"ok": True}

    return app


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _percentiles(values: list[float], points=(50, 90, 95, 99)) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 1) for p in points}


def _turns_of(result) -> list:
    """Every turn of a call, including single-turn calls recorded before turns
    existed (their top-level fields are the one turn)."""
    turns = getattr(result, "turns", None)
    if turns:
        return turns
    return [result]


def build_report(results: list, vendor_name: str, run_id: str,
                 applied: dict) -> dict:
    """Turn per-call results into the reported figures.

    Latency figures are POOLED OVER TURNS, not over calls: a 4-turn call
    contributes four measurements. Discards are counted per turn too, because one
    bad turn should not waste the other three.
    """
    # A call-level discard invalidates the whole conversation; otherwise each turn
    # stands or falls on its own.
    all_turns = []
    for r in results:
        if r.discard_reason is not None:
            continue
        all_turns.extend(_turns_of(r))

    usable_turns = [t for t in all_turns if getattr(t, "usable", False)]
    onset = [t.ttfab_onset_ms for t in usable_turns if t.ttfab_onset_ms is not None]
    content = [t.ttfab_content_ms for t in usable_turns
               if t.ttfab_content_ms is not None]

    # TTFG is a property of the CALL (the greeting happens once), so it stays
    # call-scoped.
    usable_calls = [r for r in results if r.usable]
    ttfg = [r.ttfg_ms for r in usable_calls if r.ttfg_ms is not None]

    discards: dict[str, int] = {}
    for r in results:
        if r.discard_reason is not None:
            # Whole call lost: count it once, and note how many turns went with it.
            discards[r.discard_reason] = discards.get(r.discard_reason, 0) + 1
            continue
        for turn in _turns_of(r):
            if not getattr(turn, "usable", False):
                key = turn.discard_reason or "unknown"
                discards[key] = discards.get(key, 0) + 1

    turn_counts = sorted({getattr(t, "index", 1) or 1 for t in all_turns})
    per_turn = {}
    for index in turn_counts:
        at_index = [t for t in all_turns if (getattr(t, "index", 1) or 1) == index]
        good = [t for t in at_index if getattr(t, "usable", False)]
        values = [t.ttfab_onset_ms for t in good if t.ttfab_onset_ms is not None]
        per_turn[str(index)] = {
            "attempts": len(at_index),
            "usable": len(good),
            "n": len(values),
            **_percentiles(values, points=(50, 90, 95)),
        }

    report = {
        "run_id": run_id,
        "vendor": vendor_name,
        "attempts": len(results),
        "usable": len(usable_calls),
        "turns_per_call": max(turn_counts) if turn_counts else 1,
        "turn_attempts": len(all_turns),
        "turns_usable": len(usable_turns),
        "per_turn": per_turn,
        "discards": discards,
        "ttfab_onset_ms": _percentiles(onset),
        "ttfab_onset_n": len(onset),
        "ttfab_content_ms": _percentiles(content, points=(50, 95)),
        "ttfab_content_n": len(content),
        # Labelled at every point of use: this is measured from the start of the
        # recording, which is when our record_start took effect -- NOT carrier
        # answer.
        "ttfg_from_recording_start_ms": _percentiles(ttfg, points=(50, 95)),
        "negative_ttfab_count": sum(1 for v in onset if v < 0),
        "vendor_config_sha256": applied.get("sha256"),
        "vendor_defaults_used": applied.get("defaults_used"),
        "vendor_unsupported": applied.get("unsupported"),
    }

    if onset:
        p50 = float(np.percentile(onset, 50))
        p95 = float(np.percentile(onset, 95))
        report["consistency_p95_over_p50"] = round(p95 / p50, 3) if p50 else None
        report["beyond_2x_median_fraction"] = round(
            sum(1 for v in onset if v > 2 * p50) / len(onset), 3
        )

    # Does this vendor's TTS wait for the whole answer before
    # speaking? If so, longer replies start later and TTFAB correlates with
    # response duration. Reported, never corrected for.
    pairs = [(t.vendor_response_duration_ms, t.ttfab_onset_ms) for t in usable_turns
             if t.vendor_response_duration_ms and t.ttfab_onset_ms is not None]
    if len(pairs) >= 4:
        durations = np.array([p[0] for p in pairs], dtype=np.float64)
        onsets = np.array([p[1] for p in pairs], dtype=np.float64)
        if durations.std() > 0 and onsets.std() > 0:
            report["buffering_correlation_r"] = round(
                float(np.corrcoef(durations, onsets)[0, 1]), 3
            )
            report["buffering_n"] = len(pairs)

    return report


def print_report(report: dict, honesty: list[str]) -> None:
    turns_per_call = report.get("turns_per_call", 1)
    print(f"\n=== BENCH: {report['vendor']}  ({report['run_id']}) ===\n")
    print(f"calls             {report['usable']}/{report['attempts']} usable")
    if turns_per_call > 1:
        print(f"turns             {report.get('turns_usable', 0)}/"
              f"{report.get('turn_attempts', 0)} usable "
              f"({turns_per_call} per call)")
    if report["discards"]:
        breakdown = ", ".join(f"{k}: {v}" for k, v in
                              sorted(report["discards"].items(), key=lambda kv: -kv[1]))
        print(f"discarded         {sum(report['discards'].values())}  ({breakdown})")
    else:
        print("discarded         0")

    def line(label: str, stats: dict, n: int | None = None, note: str = "") -> None:
        if not stats:
            return
        cells = "  ".join(f"{k} {v:>8.1f}" for k, v in stats.items())
        suffix = f"  (n={n})" if n is not None else ""
        print(f"{label:<18}{cells} ms{suffix}{note}")

    print()
    line("TTFAB-onset", report["ttfab_onset_ms"], report["ttfab_onset_n"])
    line("TTFAB-content", report["ttfab_content_ms"], report["ttfab_content_n"])
    line("TTFG", report["ttfg_from_recording_start_ms"], None,
         "   [from recording start, not carrier answer]")

    per_turn = report.get("per_turn") or {}
    if len(per_turn) > 1:
        indices = sorted(per_turn, key=int)
        width = 11
        print("\nTURN CURVE" + "".join(f"{'turn ' + i:>{width}}" for i in indices))
        for label, key in (("p50", "p50"), ("p90", "p90"), ("p95", "p95")):
            cells = "".join(
                f"{(f'{per_turn[i][key]:,.0f} ms' if per_turn[i].get(key) is not None else '-'):>{width}}"
                for i in indices)
            print(f"{label:<10}{cells}")
        cells = "".join(f"{f'{per_turn[i]['usable']}/{per_turn[i]['attempts']}':>{width}}"
                        for i in indices)
        print(f"{'usable':<10}{cells}")

    if report.get("negative_ttfab_count"):
        print(f"\nnegative TTFAB    {report['negative_ttfab_count']} turn(s) "
              "(vendor started before we finished -- barge-in, a result not an error)")

    if "consistency_p95_over_p50" in report:
        print(f"\nconsistency       p95/p50 = {report['consistency_p95_over_p50']}"
              f"   beyond 2x median: {report['beyond_2x_median_fraction']}")
    if "buffering_correlation_r" in report:
        r = report["buffering_correlation_r"]
        verdict = ("suggests TTS waits for the full reply"
                   if r > 0.5 else "no strong sign of full-reply buffering")
        print(f"buffering check   r = {r:+.3f} (response duration vs TTFAB, "
              f"n={report['buffering_n']}) -- {verdict}")

    used = report.get("vendor_defaults_used") or {}
    if used:
        summary = stack_summary(used)
        print(f"\nvendor stack      model {summary['model'] or '—'}")
        print(f"                  stt   {summary['stt'] or '—'}")
        print(f"                  voice {summary['voice'] or '—'}")
        # Each platform's own knob names: see vendors.base.stack_summary for why
        # these are not normalised into one vocabulary.
        knobs = ", ".join(f"{k}={v}" for k, v in summary["endpointing"].items())
        print(f"                  endpointing {knobs or '—'}")
        if summary["idle"]:
            print(f"                  {summary['idle']}")
    if report.get("vendor_unsupported"):
        print(f"                  cannot pin: {', '.join(report['vendor_unsupported'])}")
    print(f"config receipt    sha256 {str(report.get('vendor_config_sha256'))[:16]}...")

    print("\nNOT YET PUBLISHABLE")
    for item in honesty:
        print(f"  - {item}")


def finalize_run(run_dir: Path, results: list, vendor: str, applied: dict,
                 caller_receipt: dict) -> dict:
    """Turn measured calls into bench.json + report.html, and print the report.

    Shared with tools/reanalyze_run.py so that re-measuring a saved run after an
    analyzer change produces exactly what the original bench would have, rather
    than a fresh HTML page over stale aggregates. Every derived figure lives
    here; nothing is computed twice.
    """
    from report import write_html

    report = build_report(results, vendor, run_dir.name, applied)
    report["mode"] = MODE
    report["caller_config"] = caller_receipt
    report["channel_map_provisional"] = channel_map_is_provisional()
    honesty = honesty_block(report, caller_receipt)
    report["not_yet_publishable"] = honesty
    # Machine-readable alongside the prose. There is no characterisation for
    # this path, and the block says so with no numbers rather than borrowing one
    # -- a stale figure here would be quoted on a public page as if it described
    # these calls.
    report["instrument"] = instrument_block(MODE)

    report["cost"] = collect_costs(run_dir, vendor)

    (run_dir / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
    print_report(report, honesty)
    write_html(report, [asdict(r) for r in results], run_dir, applied=applied)
    return report


def collect_costs(run_dir: Path, vendor: str) -> dict | None:
    """Ask the vendor what this run cost, and summarise it into the report.

    Best-effort by design. Billing lags on some platforms, so a run finished
    seconds ago may price only partially or not at all -- that is a "come back
    later", which tools/backfill_costs.py does, and never a reason to fail a
    run whose latency measurements are already complete. Any failure here is
    printed and swallowed.
    """
    from harness import costs as C

    try:
        adapter = get_vendor(vendor)
        payload = C.collect(vendor, adapter, run_dir, settings.plivo_from_number)
    except Exception as exc:  # noqa: BLE001 -- costs never fail a measured run
        print(f"cost: unavailable ({type(exc).__name__}: {exc}) -- "
              f"`tools/backfill_costs.py --run {run_dir}` can fetch it later")
        return None

    C.write_manifest(run_dir, payload)
    summary = payload["summary"]
    if not summary["calls_priced"]:
        print(f"cost: no billing records matched yet -- "
              f"`tools/backfill_costs.py --run {run_dir}` once they settle")
    else:
        print(f"cost: {summary['total_cost']:.4f} {summary['currency']} over "
              f"{summary['total_duration_s']:.0f}s "
              f"= {summary['cost_per_minute']:.4f}/min "
              f"({summary['calls_priced']}/{len(list(run_dir.glob('call-*')))} "
              f"calls priced)")
    return summary


def instrument_block(mode: str) -> dict:
    """What characterised this run -- or, honestly, that nothing did.

    An absent characterisation must read as "unknown", never as a clean number:
    the backfill leaves instrument_id null when there is no sweep_run, and the
    page renders that as instrument unknown.
    """
    rig = instrument.for_mode(mode)
    if rig is None:
        return {
            "mode": mode,
            "valid": False,
            "note": "no delay-sweep characterisation exists for this measurement "
                    "path; recording-path overhead and instrument noise are "
                    "unmeasured here",
            "path_overhead_subtracted": False,
        }
    return {
        "mode": mode,
        "valid": True,
        "host": rig.host,
        "sweep_run": rig.sweep_run,
        "noise_sd_ms": rig.noise_sd_ms,
        "slope": rig.slope,
        "slope_gate_passed": rig.slope_gate_passed,
        "path_overhead_ms": rig.path_overhead_ms,
        "path_overhead_subtracted": False,
    }


def honesty_block(report: dict, caller_receipt: dict) -> list[str]:
    """What a reader must know before believing these numbers.

    Deliberately part of the output, not a footnote somewhere else: the whole
    argument for this benchmark is that it states its own limits.
    """
    gap = instrument.SELF_REPORTED_LATENCY_GAP
    items = [
        "The instrument is UNCHARACTERISED for this measurement path. "
        "Characterising it means measuring against a far end that answers after "
        "a known delay, and a characterisation is only valid for the host, "
        "carrier and audio path it was measured on -- so no overhead figure is "
        "quoted here and none is subtracted. Recording-path overhead sits "
        "inside every number below, unmeasured. Every figure here is therefore "
        "an upper bound on the platform's own contribution.",
        "t1 -- the end of our own speech -- is found by a speech detector on "
        "the near channel, not by matching a known waveform. It is accurate to "
        "roughly a few tens of ms rather than sample-exact, so small "
        "differences between vendors are not resolvable; the per-turn error "
        "budget is the one Gate A measures on synthetic dialogs.",
        "Turn-taking is Plivo's speech endpointing, not ours. It decides when "
        "the conversation moves on, never when a measurement starts or ends: "
        "both endpoints of every TTFAB are read off the recording afterwards. "
        "Eager or slow endpointing therefore costs discarded turns, not "
        "shifted numbers.",
        f"Our speech is {caller_receipt.get('voice')} text-to-speech reading a "
        f"fixed script, not a human recording (TTS "
        f"endings trip endpointing differently). It is re-rendered per call "
        f"rather than played from one committed file, so the stimulus is "
        f"identical in wording across vendors but not byte-identical across "
        f"calls; t1 is measured from each call's own tape, so this costs no "
        f"accuracy.",
        f"Answer accuracy is a keyword match against the carrier's transcript "
        f"of the vendor's reply -- a coarse check that the agent answered from "
        f"its prompt, reported alongside latency and never mixed into it.",
        f"A platform's own reported latency reads about {gap.floor_ms:.0f} ms "
        f"lower than what we measure from the same calls' audio "
        f"({gap.bench_run}, n={gap.n_turns} calls, sd {gap.floor_sd_ms:.0f} ms). "
        f"It is measured at the platform's edge, not at the caller's ear. Read "
        f"vendor-reported latency with that in mind.",
    ]
    if report.get("channel_map_provisional"):
        items.append(
            "The stereo channel order is assumed, not probed: Plivo does not "
            "document which channel carries which party. An inversion would "
            "swap every t1 and t2, so the analyzer counts our own utterances "
            "on the assumed near channel and discards the call as "
            "channel_map_suspect if the other channel fits better. Run "
            "`python -m harness.probe_dialog` to replace the assumption."
        )
    if report["usable"] < PUBLISHABLE_N:
        items.append(
            f"n={report['usable']} usable call(s). The method wants n>={PUBLISHABLE_N} "
            "spread over >=5 days and 4 time-of-day buckets; this is a sample, "
            "not a distribution."
        )
    per_turn = report.get("per_turn") or {}
    if len(per_turn) > 1:
        smallest = min(v["n"] for v in per_turn.values())
        items.append(
            f"The turn curve has as few as n={smallest} measurements at some turn "
            "indices. Pooled figures are the sturdier read; per-turn percentiles "
            "(especially p95) are directional until each index has many more."
        )
    if report["discards"]:
        items.append(
            "Discarded calls are reported above, never silently dropped -- a vendor "
            "that fails to respond is worse than one that is slightly slower."
        )
    return items


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True, help="a key in config/vendors.yaml")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--cases", default=None,
                        help="comma-separated case ids from "
                             "data/voice-bench/ttfab_scenarios.json. Each becomes "
                             "one measured turn, between the fixed greeting and "
                             "goodbye turns. Default: config/dialog.yaml's "
                             "default_cases")
    parser.add_argument("--pause", type=float, default=PAUSE_BETWEEN_CALLS_S,
                        help="seconds between calls. Raise it if a vendor "
                             "starts answering with silence partway through a "
                             "run -- an agent hammered back to back can stop "
                             "greeting while still accepting the call")
    parser.add_argument("--port", type=int, default=8000)
    add_server_args(parser)
    parser.add_argument("--skip-verify", action="store_true",
                        help="bench a config that differs from vendors.yaml "
                             "(recorded in the report as unverified)")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    # Before anything is dialled: a bind that disagrees with PUBLIC_BASE_URL
    # fails as "the vendor never answered", which is the most expensive kind of
    # wrong to debug.
    exposure = server_problems(args)
    if exposure:
        print("SERVER CONFIG FAILED -- webhooks would never reach us:")
        for problem in exposure:
            print(f"  - {problem}")
        return 2
    settings.require_carrier()
    carrier = get_carrier()

    # ---- vendor gate -------------------------------------------------------- #
    vendor = get_vendor(args.vendor)
    spec = spec_from_config(load_vendor_config(args.vendor))

    problems = vendor.verify_agent(spec)
    if problems and not args.skip_verify:
        print(f"\nVENDOR CONFIG MISMATCH ({args.vendor}) -- refusing to measure.")
        print("The published receipt must describe the config that actually ran.\n")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix in the vendor's portal (or update config/vendors.yaml), then "
              "re-run. Use --skip-verify to bench anyway.")
        return 2

    try:
        target = vendor.dial_target()
    except Exception as exc:  # noqa: BLE001 -- message is the product here
        print(f"\nVENDOR NOT REACHABLE ({args.vendor}):\n  {exc}")
        return 2

    applied = vendor.applied_config().as_dict()

    # ---- the caller ---------------------------------------------------------- #
    # Reported, not applied. The greeting wait used to be scaled from this, and
    # no longer is (harness/dialog.py greeting_timeout_for) -- how long a vendor
    # takes to greet is not a function of when it would re-prompt. It is still
    # worth printing: a vendor that re-prompts sooner than our wait explains an
    # idle_filler discard. Only Telnyx exposes it; stack_summary resolves the
    # other platforms' names for the report.
    idle_reply = (applied.get("defaults_used") or {}).get("user_idle_reply_secs")
    try:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()] \
            if args.cases else None
        # One script per call: the rotation in config/dialog.yaml varies the
        # questions between calls while keeping call i identical across
        # vendors and across re-runs.
        scripts = build_run_scripts(args.calls, case_ids, idle_reply_secs=idle_reply)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"\nCALLER SCRIPT PROBLEM:\n  {exc}")
        return 2
    caller_receipt = run_plan_receipt(scripts)

    # ---- server ------------------------------------------------------------- #
    registry = Registry()

    app = build_bench_app(registry)
    print(describe(args))
    server = uvicorn.Server(uvicorn.Config(app, **server_kwargs(args)))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    if not server.started:
        print(f"FATAL: could not bind :{args.port} -- another server running?")
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"bench-{args.vendor}-{stamp}"
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "applied_config.json").write_text(json.dumps(applied, indent=2,
                                                            default=str) + "\n")
    (run_dir / "caller_config.json").write_text(json.dumps(caller_receipt, indent=2,
                                                           default=str) + "\n")

    print(f"\nBENCH {args.vendor} -> {target.value}   ({args.calls} calls)")
    # Via stack_summary, not the raw key: Bland reports a capability tier
    # rather than an LLM identity, so a direct `defaults_used["model"]` lookup
    # prints "model None" for a receipt that does carry the value.
    print(f"vendor receipt: sha256 {applied['sha256'][:16]}...  "
          f"model {stack_summary(applied.get('defaults_used') or {})['model']}")
    rotation = caller_receipt["rotation_length"]
    print(f"caller receipt: sha256 {caller_receipt['sha256'][:16]}...  "
          f"{scripts[0].n_turns} turns, voice {scripts[0].voice}")
    print(f"question sets : {rotation} in rotation"
          + ("" if rotation > 1 else " (every call asks the same questions)"))
    for entry in caller_receipt["calls"][:rotation]:
        print(f"    call {entry['call_index']}: {', '.join(entry['cases'])}")
    print(f"greeting fallback: {scripts[0].greeting_timeout_s:.1f}s"
          + (f"  (vendor re-prompts at {idle_reply}s)" if idle_reply else ""))
    if channel_map_is_provisional():
        print("WARNING: stereo channel order is assumed, not probed -- run "
              "`python -m harness.probe_dialog` (see the honesty block)")
    print(f"run dir: {run_dir}\n")

    results = []
    for i in range(args.calls):
        call_id = f"call-{i:03d}"
        out_dir = run_dir / call_id
        out_dir.mkdir(parents=True, exist_ok=True)
        script = scripts[i]
        call = DialogSession(call_id=call_id, out_dir=out_dir, script=script)
        registry.current = call

        print(f"[{i + 1}/{args.calls}] ...", end="", flush=True)
        try:
            call.call_sid = carrier.place_call(target.value)
            call.event("placed", call_sid=call.call_sid, vendor=args.vendor)

            # Answered, then finished, are two different waits. A call nobody
            # picks up has nothing to converse about, so it must not sit out
            # the whole conversation deadline.
            if not call.answered.wait(timeout=ANSWER_TIMEOUT_S):
                call.event("never_answered")
                carrier.hangup_all()
                print(" not answered", flush=True)
                continue

            # The dialog ends itself: the last turn's action webhook returns
            # <Hangup/> and sets dialog_done. The deadline is the backstop for
            # a webhook that never arrives.
            call.dialog_done.wait(timeout=CALL_DEADLINE_S)
            # The vendor usually hangs up first, so this is normally a no-op --
            # `carrier.hangup` swallows "call not found" for exactly that
            # reason. Only worth attempting if we have not seen the hangup.
            if call.call_control_id and not call.hangup_seen.is_set():
                carrier.hangup(call.call_control_id)
            call.hangup_seen.wait(timeout=10.0)

            recording = fetch_recording(call, carrier)
            metadata = {
                "call_id": call_id, "run_id": run_id,
                "kind": "measure", "mode": MODE, "vendor": args.vendor,
                "carrier": carrier.name,
                "vendor_config_sha256": applied["sha256"],
                # THIS call's script, not the run's plan -- calls differ.
                "caller_config_sha256": script.receipt()["sha256"],
                "caller_plan_sha256": caller_receipt["sha256"],
                "call_sid": call.call_sid,
                "call_control_id": call.call_control_id,
                "verify_skipped": bool(problems and args.skip_verify),
                "cases": [t.case_id for t in script.turns if t.case_id],
                "turns_requested": script.n_turns,
                "turns_played": call.turns_spoken,
                "greeting_transcript": call.greeting_transcript,
                "greeting_timed_out": call.greeting_timed_out,
                # One entry per caller utterance: what we said, what came back,
                # and whether the answer contained what the case expects. The
                # analyzer pairs turn i's speech-end with turn i's reply.
                "turns": call.turn_metadata(),
                # Pinned, with provenance -- there is no known waveform to
                # re-derive it from. See harness/dialog.py.
                "channel_map": channel_map(),
            }
            if recording is None:
                metadata["recording_missing"] = True
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

            result = measure_call(out_dir)
            (out_dir / "result.json").write_text(
                json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
            results.append(result)

            if result.usable:
                print(f" ttfab={result.ttfab_onset_ms:.1f}ms")
            else:
                print(f" discarded: {result.discard_reason}")
        except Exception as exc:  # noqa: BLE001 -- one bad call must not kill the run
            call.event("bench_call_failed", error=str(exc)[:300])
            print(f" FAILED: {str(exc)[:120]}")
        finally:
            registry.current = None

        if i + 1 < args.calls:
            time.sleep(args.pause)

    server.should_exit = True
    server_thread.join(timeout=5)

    # ---- report ------------------------------------------------------------- #
    if not results:
        print("\nno calls completed -- nothing to report")
        return 1

    report = finalize_run(run_dir, results, args.vendor, applied, caller_receipt)
    print(f"\nsaved: {run_dir / 'bench.json'}")
    print(f"       {run_dir / 'report.html'}")
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
