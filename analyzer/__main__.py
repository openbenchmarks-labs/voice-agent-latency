"""Credential-free CLI over a saved run directory.

    python -m analyzer runs/<run_id>              measure a run, write result.json
    python -m analyzer runs/<run_id> --no-write   measure without touching disk
    python -m analyzer --gate-a                   run Gate A against the fixtures

Nothing here reaches the network or reads a credential. That is the point: the
numbers we publish must be re-derivable by anyone holding the artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .measure import ANALYZER_VERSION, analyze_run


def _print_table(results) -> None:
    header = (
        f"{'call':<24} {'src':<10} {'TTFAB':>10} {'content':>9} "
        f"{'TTFG':>9} {'PSR':>6} {'drift':>7}  status"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        fmt = lambda v, w=10, p=1: (f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}")  # noqa: E731
        status = r.discard_reason or "ok"
        print(
            f"{(r.call_id or '?'):<24} {r.source:<10} {fmt(r.ttfab_onset_ms)} "
            f"{fmt(r.ttfab_content_ms, 9)} {fmt(r.ttfg_ms, 9)} "
            f"{fmt(r.psr, 6, 2)} {fmt(r.drift_ms, 7)}  {status}"
        )

    usable = [r for r in results if r.usable]
    print()
    print(f"{len(usable)}/{len(results)} usable")
    if len(usable) != len(results):
        reasons: dict[str, int] = {}
        for r in results:
            if r.discard_reason:
                reasons[r.discard_reason] = reasons.get(r.discard_reason, 0) + 1
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  discarded {n:>4}  {reason}")


def _gate_a() -> int:
    """Measure every fixture family against its constructed truth.

    The only check in the pipeline that compares the analyzer to a truth we
    built rather than to itself. Three families, because there are three ways a
    call can reach the analyzer:

      reference     one known clip, one reply         (the original Gate A)
      multi-turn    four known clips in a row
      dialog        no known clips at all -- t1 detected, as every live Plivo
                    call now is

    Multi-turn used to be exercised only by pytest, which meant the CLI gate
    could pass while the path most runs take was broken.
    """
    failed = _gate_a_reference()
    failed |= _gate_a_multiturn()
    failed |= _gate_a_dialog()
    print()
    if failed:
        print("GATE A FAILED")
        return 1
    print("GATE A PASSED")
    return 0


def _gate_a_reference() -> int:
    import tempfile

    import numpy as np
    import soundfile as sf

    from .fixtures import build_all
    from .measure import measure_recording

    tolerance_ms = 5.0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        truths = build_all(out)

        print(f"analyzer {ANALYZER_VERSION} -- Gate A: reference mode, "
              f"tolerance +/-{tolerance_ms} ms\n")
        print(f"{'fixture':<20} {'true':>10} {'measured':>10} {'error':>8}  status")
        print("-" * 66)

        worst = 0.0
        failures = []
        for name, truth in sorted(truths.items()):
            audio, rate = sf.read(out / truth["wav"], dtype="int16", always_2d=True)
            reference, _ = sf.read(out / truth["reference_wav"], dtype="int16")
            near = np.ascontiguousarray(audio[:, 0])
            far = np.ascontiguousarray(audio[:, 1])

            result = measure_recording(near, far, reference, rate)
            measured = result.ttfab_onset_ms
            expected = truth["ttfab_ms"]
            wanted_discard = truth["expect_discard"]

            # Branch on the verdict, not on whether a number came out. A discarded
            # call still carries its measured TTFAB on purpose -- it is useful to see
            # what the wrong answer would have been -- so testing `measured is None`
            # would score a correct discard as a passing measurement.
            shown = f"{measured:>10.2f}" if measured is not None else f"{'-':>10}"

            if result.discard_reason is not None:
                ok = result.discard_reason == wanted_discard
                print(f"{name:<20} {expected:>10.2f} {shown} {'-':>8}  "
                      f"discarded: {result.discard_reason}"
                      f"{'' if ok else '  FAIL'}")
                if not ok:
                    failures.append(
                        f"{name}: discarded as {result.discard_reason}, "
                        f"expected {wanted_discard or 'a measurement'}"
                    )
                continue

            if wanted_discard is not None:
                print(f"{name:<20} {expected:>10.2f} {shown} {'-':>8}  "
                      f"FAIL: should have been discarded as {wanted_discard}")
                failures.append(
                    f"{name}: produced a measurement but should have been "
                    f"discarded as {wanted_discard}"
                )
                continue

            if measured is None:
                print(f"{name:<20} {expected:>10.2f} {shown} {'-':>8}  FAIL: no TTFAB")
                failures.append(f"{name}: no TTFAB and no discard reason")
                continue

            error = measured - expected
            worst = max(worst, abs(error))
            ok = abs(error) <= tolerance_ms
            if not ok:
                failures.append(f"{name}: error {error:+.2f} ms")
            print(
                f"{name:<20} {expected:>10.2f} {shown} {error:>+8.2f}  "
                f"{'ok' if ok else 'FAIL'}"
            )

        print()
        print(f"worst |error| = {worst:.2f} ms")
        for f in failures:
            print(f"  FAIL {f}")
        return 1 if failures else 0


def _gate_a_multiturn() -> int:
    """Four constructed gaps in one recording, recovered separately."""
    import tempfile

    import numpy as np
    import soundfile as sf

    from .fixtures.multiturn import build_all_multiturn
    from .measure import measure_turns
    from .resample import resample_int16

    tolerance_ms = 5.0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        truths = build_all_multiturn(out)

        print(f"\nanalyzer {ANALYZER_VERSION} -- Gate A: multi-turn, "
              f"tolerance +/-{tolerance_ms} ms\n")
        print(f"{'fixture':<26} {'turn':>5} {'true':>10} {'measured':>10} "
              f"{'error':>8}  status")
        print("-" * 76)

        worst, failures = 0.0, []
        for name, truth in sorted(truths.items()):
            audio, rate = sf.read(out / truth["wav"], dtype="int16", always_2d=True)
            references = []
            for row in truth["turns"]:
                ref, ref_rate = sf.read(out / row["reference_wav"], dtype="int16")
                references.append(ref if ref_rate == rate
                                  else resample_int16(ref, ref_rate, rate))
            result = measure_turns(np.ascontiguousarray(audio[:, 0]),
                                   np.ascontiguousarray(audio[:, 1]),
                                   references, rate)
            for turn, row in zip(result.turns, truth["turns"]):
                expected = row["ttfab_ms"]
                measured = turn.ttfab_onset_ms
                if turn.discard_reason is not None:
                    print(f"{name:<26} {turn.index:>5} {expected:>10.2f} "
                          f"{'-':>10} {'-':>8}  discarded: {turn.discard_reason}")
                    continue
                if measured is None:
                    failures.append(f"{name} turn {turn.index}: no TTFAB")
                    continue
                error = measured - expected
                worst = max(worst, abs(error))
                ok = abs(error) <= tolerance_ms
                if not ok:
                    failures.append(f"{name} turn {turn.index}: error {error:+.2f} ms")
                print(f"{name:<26} {turn.index:>5} {expected:>10.2f} "
                      f"{measured:>10.2f} {error:>+8.2f}  {'ok' if ok else 'FAIL'}")

        print()
        print(f"worst |error| = {worst:.2f} ms")
        for f in failures:
            print(f"  FAIL {f}")
        return 1 if failures else 0


def _gate_a_dialog() -> int:
    """The live-caller path: no references, t1 detected on the near channel.

    Measured through `measure_call` on a synthetic call directory, so the gate
    exercises the same entry point, mode gate and channel handling a real run
    does -- not an inner function no call ever reaches.
    """
    import tempfile

    from .fixtures.dialog import (
        T1_TOLERANCE_MS,
        TTFAB_TOLERANCE_MS,
        attribution_table,
        build_all_dialog,
    )
    from .measure import measure_call
    from .resample import samples_to_ms

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        truths = build_all_dialog(out)

        print(f"\nanalyzer {ANALYZER_VERSION} -- Gate A: scripted dialog, "
              f"tolerance +/-{TTFAB_TOLERANCE_MS} ms TTFAB, "
              f"+/-{T1_TOLERANCE_MS} ms t1\n")
        print(f"{'fixture':<24} {'turn':>5} {'true':>10} {'measured':>10} "
              f"{'error':>8} {'t1 err':>8}  status")
        print("-" * 84)

        worst_ttfab, worst_t1, failures = 0.0, 0.0, []
        for name, truth in sorted(truths.items()):
            result = measure_call(Path(truth["call_dir"]))
            rate = truth["rate"]

            if result.discard_reason is not None or truth["expect_call_discard"]:
                ok = result.discard_reason == truth["expect_call_discard"]
                print(f"{name:<24} {'-':>5} {'-':>10} {'-':>10} {'-':>8} {'-':>8}  "
                      f"call discarded: {result.discard_reason}"
                      f"{'' if ok else '  FAIL'}")
                if not ok:
                    failures.append(
                        f"{name}: call discarded as {result.discard_reason}, "
                        f"expected {truth['expect_call_discard'] or 'a measurement'}")
                continue

            # Flags a fixture must carry. Checked by prefix so the measured value
            # (a pause in ms, a count) stays free to differ. Without this a
            # fixture whose point is "kept, but recorded" would pass on the
            # verdict alone, and the recording half could silently disappear.
            for prefix in truth.get("expect_flags", ()):
                if not any(f.startswith(prefix) for f in result.flags):
                    failures.append(f"{name}: missing flag {prefix!r} "
                                    f"(flags: {', '.join(result.flags) or 'none'})")
                    print(f"{name:<24} {'-':>5} {'-':>10} {'-':>10} {'-':>8} "
                          f"{'-':>8}  FAIL: missing flag {prefix!r}")

            for turn, row in zip(result.turns, truth["turns"]):
                wanted = row["expect_discard"]
                t1_err = (turn.t1_ms - samples_to_ms(row["t1"], rate)
                          if turn.t1_ms is not None else None)
                if turn.discard_reason is not None:
                    ok = turn.discard_reason == wanted
                    print(f"{name:<24} {turn.index:>5} {'-':>10} {'-':>10} "
                          f"{'-':>8} {'-':>8}  discarded: {turn.discard_reason}"
                          f"{'' if ok else '  FAIL'}")
                    if not ok:
                        failures.append(
                            f"{name} turn {turn.index}: discarded as "
                            f"{turn.discard_reason}, expected {wanted or 'a measurement'}")
                    continue
                if wanted is not None:
                    failures.append(f"{name} turn {turn.index}: should have been "
                                    f"discarded as {wanted}")
                    print(f"{name:<24} {turn.index:>5} {'-':>10} {'-':>10} "
                          f"{'-':>8} {'-':>8}  FAIL: expected {wanted}")
                    continue

                expected, measured = row["ttfab_ms"], turn.ttfab_onset_ms
                if measured is None:
                    failures.append(f"{name} turn {turn.index}: no TTFAB")
                    continue
                error = measured - expected
                worst_ttfab = max(worst_ttfab, abs(error))
                ok = abs(error) <= TTFAB_TOLERANCE_MS
                if t1_err is not None:
                    worst_t1 = max(worst_t1, abs(t1_err))
                    if abs(t1_err) > T1_TOLERANCE_MS:
                        ok = False
                        failures.append(f"{name} turn {turn.index}: "
                                        f"t1 error {t1_err:+.2f} ms")
                if abs(error) > TTFAB_TOLERANCE_MS:
                    failures.append(f"{name} turn {turn.index}: error {error:+.2f} ms")
                print(f"{name:<24} {turn.index:>5} {expected:>10.2f} "
                      f"{measured:>10.2f} {error:>+8.2f} "
                      f"{(f'{t1_err:+.2f}' if t1_err is not None else '-'):>8}  "
                      f"{'ok' if ok else 'FAIL'}")

        print()
        print(f"worst |TTFAB error| = {worst_ttfab:.2f} ms   "
              f"worst |t1 error| = {worst_t1:.2f} ms")
        # The evidence for _refine_offset's attribution choice, recomputed
        # rather than remembered.
        rows = attribution_table(out)
        print("t1 error by turn on the clean fixture: "
              + ", ".join(f"{r['t1_error_ms']:+.2f}" for r in rows) + " ms")
        for f in failures:
            print(f"  FAIL {f}")
        return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analyzer", description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="runs/<run_id>")
    parser.add_argument("--no-write", action="store_true",
                        help="do not write result.json files")
    parser.add_argument("--json", action="store_true",
                        help="emit results as JSON instead of a table")
    parser.add_argument("--gate-a", action="store_true",
                        help="validate the analyzer against the synthetic fixtures")
    args = parser.parse_args(argv)

    if args.gate_a:
        return _gate_a()

    if args.run_dir is None:
        parser.error("run_dir is required unless --gate-a is given")
    if not args.run_dir.is_dir():
        print(f"not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    results = analyze_run(args.run_dir, write=not args.no_write)
    if not results:
        print(f"no call directories under {args.run_dir}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
    else:
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
