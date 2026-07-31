"""Assemble ground-truth fixtures from the committed speech clips.

Each fixture is a stereo recording shaped like a real Plivo dual-channel
recording -- near channel is what we played, far channel is what the vendor sent
-- plus a JSON sidecar stating the truth we constructed.

See the module docstring in __init__.py for the t1/t2 sample-index convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..codec import mulaw_roundtrip
from ..resample import ms_to_samples, resample_int16

SOURCE_RATE = 16000
RECORDING_RATE = 8000  # the rate every reported number is measured at
UNCOMPANDED_RATE = 16000

SOURCES = Path(__file__).resolve().parent / "sources"

# Appended to our stimulus so playback cannot clip the final consonant. It is
# true digital silence, and mu-law preserves that exactly, so it does not move
# t1 -- but it does mean anyone measuring from the END OF FILE rather than the
# end of speech is off by this much. Documented in metadata for that reason.
TRAILING_SILENCE_MS = 200

LEAD_IN_MS = 300  # far-channel noise floor is estimated from this window


@dataclass
class Fixture:
    """A fixture and the truth used to build it."""

    name: str
    description: str
    gap_ms: float
    stimulus: str = "stimulus_hours"
    response: str = "response"
    greeting: str | None = "greeting"
    rate: int = RECORDING_RATE
    companded: bool = True
    comfort_noise_dbfs: float | None = None
    drop_ms: float | None = None
    drop_at_frac: float = 0.5
    filler_before_response: bool = False
    filler_gap_ms: float = 300.0
    # Real acknowledgement tokens run 200-400 ms. The source clip is longer than
    # that, so it is truncated -- otherwise it is long enough to qualify as
    # content and the onset/content distinction cannot be tested.
    filler_max_ms: float = 250.0
    idle_filler: bool = False
    # Silence we leave after the greeting before speaking. Live VAD would use a
    # short hangover; the idle-filler fixture needs a long one so the vendor's
    # idle prompt has somewhere to sit.
    hangover_ms: float = 600.0
    idle_offset_ms: float = 1200.0
    expect_discard: str | None = None
    truth: dict = field(default_factory=dict)


FIXTURES: list[Fixture] = [
    Fixture(
        name="clean_500ms",
        description="Baseline. 500 ms between our speech ending and the vendor "
                    "starting. This is the Gate A fixture.",
        gap_ms=500.0,
    ),
    Fixture(
        name="clean_300ms",
        description="Shorter gap, to catch a bias that only shows up as a "
                    "proportional error.",
        gap_ms=300.0,
    ),
    Fixture(
        name="clean_1200ms",
        description="Longer gap. With 300/500/1200 present, a multiplicative "
                    "error cannot hide behind a single calibration.",
        gap_ms=1200.0,
    ),
    Fixture(
        name="comfort_noise",
        description="Far channel carries continuous low-level noise. This is the "
                    "case that makes an energy-threshold detector fire instantly "
                    "and report a vendor as ~200 ms faster than it is.",
        gap_ms=500.0,
        comfort_noise_dbfs=-45.0,
    ),
    Fixture(
        name="uncompanded_16k",
        description="Uncompanded 16 kHz linear audio, no mu-law. "
                    "Correlation should be at least as accurate here.",
        gap_ms=500.0,
        rate=UNCOMPANDED_RATE,
        companded=False,
    ),
    Fixture(
        name="mid_stimulus_gap",
        description="40 ms excised from the middle of our stimulus, simulating "
                    "packet loss. Whole-file correlation would push t1 out by the "
                    "dropped amount; tail-window correlation should not. The drift "
                    "check exists to notice this.",
        gap_ms=500.0,
        drop_ms=40.0,
    ),
    Fixture(
        name="barge_in",
        description="Vendor starts talking 250 ms BEFORE we stop. TTFAB is "
                    "legitimately negative -- a result, not an error.",
        gap_ms=-250.0,
        stimulus="stimulus_hesitation",
    ),
    Fixture(
        name="filler_first",
        description="Vendor emits 'mm hmm', pauses, then answers. Separates "
                    "TTFAB-onset from TTFAB-content.",
        gap_ms=400.0,
        filler_before_response=True,
    ),
    Fixture(
        name="idle_filler",
        description="Vendor speaks between its greeting ending and our stimulus "
                    "starting -- the 'are you still there?' prompt. Must be "
                    "discarded, not recorded as the response.",
        gap_ms=500.0,
        idle_filler=True,
        # A real idle prompt fires after seconds of silence, so the fixture waits
        # before speaking and drops the intruder into the middle of that window.
        hangover_ms=3000.0,
        idle_offset_ms=1200.0,
        expect_discard="idle_filler",
    ),
]


def _load(name: str) -> np.ndarray:
    audio, sr = sf.read(SOURCES / f"{name}.wav", dtype="int16")
    if sr != SOURCE_RATE:
        raise RuntimeError(f"{name}.wav is {sr} Hz, expected {SOURCE_RATE}")
    if audio.ndim != 1:
        raise RuntimeError(f"{name}.wav is not mono")
    return audio


def _comfort_noise(n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    amp = (10.0 ** (dbfs / 20.0)) * 32767.0
    return np.rint(rng.normal(0.0, amp / 3.0, n)).astype(np.int64)


def _place(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    """Add `src` into `dst` at sample `at`, clipping to the buffer."""
    if at >= len(dst):
        return
    start = max(0, at)
    src_start = start - at
    n = min(len(src) - src_start, len(dst) - start)
    if n <= 0:
        return
    dst[start : start + n] += src[src_start : src_start + n]


def build(fx: Fixture, out_dir: Path) -> dict:
    """Build one fixture. Returns its truth dict, also written as JSON."""
    rng = np.random.default_rng(abs(hash(fx.name)) % (2**32))

    stimulus = _load(fx.stimulus)
    response = _load(fx.response)
    greeting = _load(fx.greeting) if fx.greeting else np.zeros(0, np.int16)
    filler = _load("filler") if fx.filler_before_response else np.zeros(0, np.int16)
    if len(filler):
        filler = filler[: ms_to_samples(fx.filler_max_ms, SOURCE_RATE)]

    # The reference is what we INTENDED to play. When the fixture simulates packet
    # loss, the recording gets the degraded copy while the reference keeps the
    # original -- otherwise the two match perfectly and the drift case tests
    # nothing.
    intended = stimulus
    if fx.drop_ms:
        # Excise from the middle, leaving the tail intact -- that is the whole
        # point: tail-window correlation should be unaffected.
        n_drop = ms_to_samples(fx.drop_ms, SOURCE_RATE)
        cut = int(len(stimulus) * fx.drop_at_frac)
        stimulus = np.concatenate([stimulus[:cut], stimulus[cut + n_drop :]])

    rate = fx.rate
    if rate != SOURCE_RATE:
        stimulus = resample_int16(stimulus, SOURCE_RATE, rate)
        intended = resample_int16(intended, SOURCE_RATE, rate)
        response = resample_int16(response, SOURCE_RATE, rate)
        if len(greeting):
            greeting = resample_int16(greeting, SOURCE_RATE, rate)
        if len(filler):
            filler = resample_int16(filler, SOURCE_RATE, rate)

    lead = ms_to_samples(LEAD_IN_MS, rate)
    trailing = ms_to_samples(TRAILING_SILENCE_MS, rate)
    gap = ms_to_samples(fx.gap_ms, rate)

    # Our stimulus starts after the greeting has finished plus a hangover, which
    # is what live VAD would do.
    greeting_start = lead
    greeting_end = greeting_start + len(greeting)
    hangover = ms_to_samples(fx.hangover_ms, rate)
    stimulus_start = greeting_end + hangover

    # t1: one past the last speech sample. t2: the first speech sample of the
    # vendor's reply. TTFAB == t2 - t1 by construction.
    t1 = stimulus_start + len(stimulus)
    t2 = t1 + gap

    if fx.filler_before_response:
        filler_start = t2
        response_start = filler_start + len(filler) + ms_to_samples(fx.filler_gap_ms, rate)
    else:
        filler_start = None
        response_start = t2

    total = max(t1 + trailing, response_start + len(response)) + ms_to_samples(500.0, rate)

    near = np.zeros(total, dtype=np.int64)
    far = np.zeros(total, dtype=np.int64)

    _place(near, stimulus.astype(np.int64), stimulus_start)
    if len(greeting):
        _place(far, greeting.astype(np.int64), greeting_start)

    idle_start = None
    if fx.idle_filler:
        # Sits strictly between greeting end and stimulus start, which is the
        # window the discard rule watches.
        idle = _load("filler")[: ms_to_samples(fx.filler_max_ms, SOURCE_RATE)]
        if rate != SOURCE_RATE:
            idle = resample_int16(idle, SOURCE_RATE, rate)
        idle_start = greeting_end + ms_to_samples(fx.idle_offset_ms, rate)
        assert idle_start + len(idle) < stimulus_start, (
            "the idle prompt must finish before our stimulus starts, or it is not "
            "in the window the discard rule watches"
        )
        _place(far, idle.astype(np.int64), idle_start)

    if filler_start is not None:
        _place(far, filler.astype(np.int64), filler_start)
    _place(far, response.astype(np.int64), response_start)

    if fx.comfort_noise_dbfs is not None:
        far += _comfort_noise(total, fx.comfort_noise_dbfs, rng)

    near = np.clip(near, -32768, 32767).astype(np.int16)
    far = np.clip(far, -32768, 32767).astype(np.int16)

    if fx.companded:
        near = mulaw_roundtrip(near)
        far = mulaw_roundtrip(far)

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{fx.name}.wav"
    sf.write(wav_path, np.stack([near, far], axis=1), rate, subtype="PCM_16")

    # The reference the analyzer correlates against is the audio we intended to
    # play, companded the same way the recording was.
    ref = mulaw_roundtrip(intended) if fx.companded else intended
    ref_path = out_dir / f"{fx.name}.reference.wav"
    sf.write(ref_path, ref, rate, subtype="PCM_16")

    truth = {
        "name": fx.name,
        "description": fx.description,
        "rate": rate,
        "companded": fx.companded,
        "channels": {"near": 0, "far": 1},
        "stimulus_trailing_silence_ms": TRAILING_SILENCE_MS,
        "greeting_onset": greeting_start if len(greeting) else None,
        "greeting_end": greeting_end if len(greeting) else None,
        "stimulus_start": stimulus_start,
        "t1": t1,
        "t2": t2,
        "response_start": response_start,
        "filler_start": filler_start,
        "idle_filler_start": idle_start,
        "ttfab_samples": t2 - t1,
        "ttfab_ms": (t2 - t1) * 1000.0 / rate,
        "ttfab_content_ms": (response_start - t1) * 1000.0 / rate,
        "expect_discard": fx.expect_discard,
        "reference_wav": ref_path.name,
        "wav": wav_path.name,
    }
    (out_dir / f"{fx.name}.truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def build_all(out_dir: Path) -> dict[str, dict]:
    return {fx.name: build(fx, out_dir) for fx in FIXTURES}
