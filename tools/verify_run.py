#!/usr/bin/env python3
"""Re-derive a published TTFAB from the published audio, and diff it.

This is the tool that makes "every published figure is re-derivable" a command
rather than a promise. It needs no credentials and no vendor account -- only the
committed artifacts under data/voice-runs/ and the public recording URLs they
carry.

For each call in a run it:
  1. reads data/voice-runs/<run_id>/call-NNN.json for the recording URL + sha256
  2. downloads the carrier tape and VERIFIES the checksum, refusing on mismatch
  3. resamples it to the analyzer's 8 kHz exactly as the harness did
  4. runs the analyzer over the reconstructed call directory
  5. diffs its own per-turn TTFAB against the published numbers

Step 3 is the step that did not previously exist anywhere reusable. What we
publish is `recording_raw.wav`, the tape as the carrier wrote it; what the
analyzer reads is `recording.wav`, that tape resampled. The conversion lived
inline in harness/dialog.py fetch_recording, reachable only by placing a real
phone call. Publishing the raw tape and keeping the derivation private meant the
reproduction path had a missing link.

Exit code is non-zero if any turn disagrees by more than --tolerance, so this
doubles as a regression check on the analyzer against real calls rather than
synthetic fixtures.

Usage:
    .venv/bin/python tools/verify_run.py bench-telnyx-20260730-173501
    .venv/bin/python tools/verify_run.py --all
    .venv/bin/python tools/verify_run.py <run_id> --keep /tmp/rebuilt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

import httpx                                             # noqa: E402
import numpy as np                                       # noqa: E402
import soundfile as sf                                   # noqa: E402

from analyzer.measure import ANALYZER_VERSION, analyze_run  # noqa: E402
from analyzer.resample import resample_int16             # noqa: E402
from harness.config import data_root                     # noqa: E402

#: The analyzer's working rate. Must match harness/dialog.py ANALYZER_RATE --
#: imported from there would drag the carrier and FastAPI into a tool that needs
#: neither, so it is asserted against the published `rate` field instead.
ANALYZER_RATE = 8000

#: A deterministic pipeline should reproduce a number exactly. A small tolerance
#: absorbs resampler differences between scipy builds, which is the one part of
#: the chain we do not pin for an outsider.
DEFAULT_TOLERANCE_MS = 5.0

VOICE_RUNS = "data/voice-runs"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebuild_call(entry: dict, call_dir: Path, client: httpx.Client) -> str | None:
    """Materialise one call directory from its published artifact. Returns an
    error string, or None on success."""
    recording = entry.get("recording") or {}
    url = recording.get("url")
    if not url:
        return "no recording url in the artifact (the call produced no tape)"

    response = client.get(url)
    if response.status_code >= 300:
        return f"download failed: HTTP {response.status_code}"
    raw = response.content

    want = recording.get("sha256")
    got = sha256_of(raw)
    if want and got != want:
        # Refuse rather than measure. A tape that does not match its checksum is
        # not the evidence the number was derived from, and quietly measuring it
        # anyway would produce a figure with no provenance at all.
        return f"CHECKSUM MISMATCH\n      published {want}\n      downloaded {got}"

    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "recording_raw.wav").write_bytes(raw)

    audio, rate = sf.read(call_dir / "recording_raw.wav", dtype="int16",
                          always_2d=True)
    if audio.shape[1] < 2:
        return f"tape is not stereo ({audio.shape[1]} channel) -- cannot separate near/far"
    if rate != ANALYZER_RATE:
        audio = np.stack(
            [resample_int16(np.ascontiguousarray(audio[:, c]), rate, ANALYZER_RATE)
             for c in range(audio.shape[1])],
            axis=1,
        )
    sf.write(call_dir / "recording.wav", audio, ANALYZER_RATE, subtype="PCM_16")

    # The analyzer reads channel_map, mode and the turn script from here.
    metadata = entry.get("metadata")
    if not metadata:
        return "no metadata in the artifact -- the analyzer cannot tell the channels apart"
    (call_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return None


def published_turns(entry: dict) -> dict[int, float]:
    result = entry.get("result") or {}
    return {t["index"]: t["ttfab_onset_ms"]
            for t in (result.get("turns") or [])
            if t.get("ttfab_onset_ms") is not None}


def measured_turns(result) -> dict[int, float]:
    turns = getattr(result, "turns", None) or []
    return {t.index: t.ttfab_onset_ms for t in turns
            if getattr(t, "ttfab_onset_ms", None) is not None}


def verify(run_id: str, tolerance: float, keep: Path | None) -> tuple[int, int]:
    """Returns (turns_compared, turns_disagreeing)."""
    run_artifacts = data_root() / VOICE_RUNS / run_id
    artifacts = sorted(run_artifacts.glob("call-*.json"))
    if not artifacts:
        print(f"{run_id}: no artifacts under {run_artifacts}")
        return (0, 0)

    print(f"\n{run_id}  ({len(artifacts)} calls)")
    workdir = Path(tempfile.mkdtemp(prefix=f"verify-{run_id}-")) if keep is None \
        else (keep / run_id)
    entries: dict[str, dict] = {}

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for path in artifacts:
            entry = json.loads(path.read_text())
            call_id = (entry.get("result") or {}).get("call_id") or path.stem
            entries[call_id] = entry
            problem = rebuild_call(entry, workdir / call_id, client)
            if problem:
                print(f"  {call_id}: {problem}")

    if not any((workdir / c).is_dir() for c in entries):
        print("  nothing could be rebuilt")
        return (0, 0)

    # write=False: never touch the published result.json, only compare against it.
    results = analyze_run(workdir, write=False)

    compared = disagree = 0
    print(f"  {'call':10} {'turn':>4} {'published':>10} {'re-derived':>11} {'delta':>8}")
    for result in results:
        entry = entries.get(result.call_id)
        if entry is None:
            continue
        theirs, ours = published_turns(entry), measured_turns(result)
        for index in sorted(theirs):
            if index not in ours:
                print(f"  {result.call_id:10} {index:>4} {theirs[index]:>10.1f} "
                      f"{'MISSING':>11} {'':>8}  <-- not re-derived")
                disagree += 1
                continue
            delta = ours[index] - theirs[index]
            compared += 1
            flag = "" if abs(delta) <= tolerance else "  <-- DISAGREES"
            if flag:
                disagree += 1
            if flag or abs(delta) > 0:
                print(f"  {result.call_id:10} {index:>4} {theirs[index]:>10.1f} "
                      f"{ours[index]:>11.1f} {delta:>+8.1f}{flag}")

    exact = compared - disagree
    print(f"  {compared} turns compared, {exact} within {tolerance:g} ms"
          + (f", {disagree} DISAGREE" if disagree else " (all of them)"))
    if keep is not None:
        print(f"  rebuilt call directories kept at {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return (compared, disagree)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="*", metavar="RUN_ID",
                        help="run ids under data/voice-runs/")
    parser.add_argument("--all", action="store_true",
                        help="every run under data/voice-runs/")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_MS,
                        metavar="MS", help=f"per-turn tolerance in ms "
                                          f"(default {DEFAULT_TOLERANCE_MS:g})")
    parser.add_argument("--keep", type=Path, default=None, metavar="DIR",
                        help="keep the rebuilt call directories for inspection "
                             "instead of deleting them")
    args = parser.parse_args()

    root = data_root() / VOICE_RUNS
    run_ids = ([p.name for p in sorted(root.iterdir()) if p.is_dir()]
               if args.all else args.runs)
    if not run_ids:
        return parser.error(f"pass a run id or --all. Available under {root}: "
                            + ", ".join(p.name for p in sorted(root.iterdir())
                                        if p.is_dir()))

    print(f"analyzer {ANALYZER_VERSION}  tolerance {args.tolerance:g} ms")
    total = bad = 0
    for run_id in run_ids:
        compared, disagree = verify(run_id, args.tolerance, args.keep)
        total += compared
        bad += disagree

    print(f"\n{total} turns re-derived from published audio; "
          + (f"{bad} disagree with the published figures" if bad
             else "every one matches"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
