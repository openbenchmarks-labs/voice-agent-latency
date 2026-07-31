"""Blind-test the analyzer: hide a delay, see if it finds it.

    python tools/blind_test.py           random secret delay, revealed at the end
    python tools/blind_test.py 730       use your own delay (ms), e.g. 730
    python tools/blind_test.py 730 --noise --barge ...   make it harder

Builds a fake call recording with the chosen gap between "our" speech ending and
the "agent" starting, runs the real measurement code on it, then reveals the
truth and the error. Nothing is shared between construction and measurement
except the WAV file itself -- the same situation the analyzer faces with a real
call.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

import numpy as np
import soundfile as sf

from analyzer.fixtures.generate import Fixture, build
from analyzer.measure import measure_recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delay_ms", nargs="?", type=float, default=None,
                        help="the true delay in ms (default: random 150-2500, kept secret)")
    parser.add_argument("--noise", action="store_true",
                        help="add comfort noise on the agent channel")
    parser.add_argument("--damage", action="store_true",
                        help="delete 40 ms from the middle of our question in transit")
    parser.add_argument("--barge", action="store_true",
                        help="make the delay negative: agent interrupts us")
    parser.add_argument("--keep", metavar="DIR",
                        help="also save the WAV here so you can listen to it")
    args = parser.parse_args()

    secret = args.delay_ms is None
    delay = args.delay_ms if not secret else round(random.uniform(150.0, 2500.0), 1)
    if args.barge:
        delay = -abs(delay)

    fx = Fixture(
        name="blind",
        description="blind test",
        gap_ms=delay,
        comfort_noise_dbfs=-45.0 if args.noise else None,
        drop_ms=40.0 if args.damage else None,
        stimulus="stimulus_hesitation" if args.barge else "stimulus_hours",
    )

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        truth = build(fx, out)

        audio, rate = sf.read(out / truth["wav"], dtype="int16", always_2d=True)
        reference, _ = sf.read(out / truth["reference_wav"], dtype="int16")
        near = np.ascontiguousarray(audio[:, 0])
        far = np.ascontiguousarray(audio[:, 1])

        if args.keep:
            keep = Path(args.keep)
            keep.mkdir(parents=True, exist_ok=True)
            sf.write(keep / "blind_test.wav", audio, rate, subtype="PCM_16")
            print(f"saved {keep / 'blind_test.wav'}  (left = us, right = agent)\n")

        result = measure_recording(near, far, reference, rate)

    if result.discard_reason is not None:
        print(f"the analyzer refused to measure this call: {result.discard_reason}")
        print(f"(the true delay was {delay:.1f} ms)")
        return 1

    measured = result.ttfab_onset_ms
    print(f"analyzer measured : {measured:8.1f} ms")
    if secret:
        print(f"secret true delay : {delay:8.1f} ms   (chosen randomly, never shown to the analyzer)")
    else:
        print(f"true delay        : {delay:8.1f} ms")
    print(f"error             : {measured - delay:+8.1f} ms")

    extras = []
    if result.drift_ms:
        extras.append(f"detected {result.drift_ms:+.0f} ms of in-flight damage")
    if result.noise_floor_dbfs is not None:
        extras.append(f"measured the agent-side noise floor at {result.noise_floor_dbfs:.0f} dB")
    if extras:
        print("also: " + "; ".join(extras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
