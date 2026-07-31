"""One-off: generate the speech clips the analyzer fixtures are assembled from.

Run on macOS only, and only when the source clips need regenerating. The output
WAVs are committed to the repo so that tests are deterministic everywhere --
notably on the Linux VPS, where `say` does not exist.

Why real speech rather than synthesised tones: Silero VAD is trained on speech,
and a harmonic-plus-noise approximation is not a fair test of it. Correlation is
also *harder* on speech than on noise, because speech is quasi-periodic and
produces sidelobes at the pitch period. Testing on real speech keeps Gate A
honest.

These are fixtures for testing the analyzer. The real benchmark stimuli are
human-recorded -- TTS endings are unnaturally clean and trip
endpointing differently, which is exactly why they cannot be used for the
measurement itself.

    python tools/make_fixture_sources.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
VOICE = "Alex"  # pinned so regeneration is reproducible
OUT_DIR = Path(__file__).resolve().parent.parent / "analyzer" / "fixtures" / "sources"

CLIPS = {
    "stimulus_hours": "What are your Saturday opening hours?",
    "stimulus_hesitation": "My account number is, uh, four five nine two.",
    "greeting": "Hi, thanks for calling. How can I help you today?",
    "response": "We are open nine in the morning until five in the afternoon on Saturdays.",
    "filler": "Mm hmm.",
    # Turns 2-4 of the multi-turn script. Turn 1 reuses
    # stimulus_hours so single-turn and multi-turn runs share a comparable turn 1.
    #
    # Deliberately short and context-dependent: "And what about Sunday?" is
    # meaningless without turn 1, so the agent has to carry conversation state --
    # which is exactly the effect the turn-index curve is measuring. All four stay
    # inside the agent's brief (opening hours only), otherwise we would be timing
    # refusals instead of answers.
    "turn_sunday": "And what about Sunday?",
    "turn_holidays": "Are you open on public holidays?",
    "turn_weekdays": "What time do you close on weekdays?",
}

# Trim threshold, relative to each clip's own peak. Low enough to keep breathy
# onsets and final fricatives, high enough to cut the encoder's lead-in silence.
TRIM_REL = 0.01


def _say_to_wav(text: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "s.aiff"
        subprocess.run(
            ["say", "-v", VOICE, "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1",
             str(aiff), str(dest)],
            check=True,
            capture_output=True,
        )


def _trim_to_speech(x: np.ndarray) -> np.ndarray:
    """Cut leading and trailing near-silence.

    The result's LAST SAMPLE is the last speech sample. The fixtures and the real
    stimulus prep both rely on that, because t1 is defined as the end of speech
    and the 200 ms of trailing digital silence is appended afterwards on purpose.
    """
    env = np.abs(x.astype(np.float64))
    thresh = env.max() * TRIM_REL
    loud = np.nonzero(env > thresh)[0]
    if loud.size == 0:
        raise RuntimeError("clip is silent")
    return x[loud[0] : loud[-1] + 1]


def main() -> int:
    if sys.platform != "darwin":
        print("macOS only -- the committed sources in", OUT_DIR, "are the artifact")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for name, text in CLIPS.items():
            raw = Path(td) / f"{name}.wav"
            _say_to_wav(text, raw)
            audio, sr = sf.read(raw, dtype="int16")
            assert sr == SR, sr
            trimmed = _trim_to_speech(audio)
            dest = OUT_DIR / f"{name}.wav"
            sf.write(dest, trimmed, SR, subtype="PCM_16")
            print(f"{dest.name:28s} {len(trimmed):7d} samples  {len(trimmed)/SR:6.3f}s")

    print("\nCommit these. Tests assemble fixtures from them without running `say`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
