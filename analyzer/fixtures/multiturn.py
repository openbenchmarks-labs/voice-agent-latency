"""Multi-turn ground truth: one recording, four constructed gaps.

This is Gate A for the turn-index measurement. Single-turn Gate A proved the
analyzer can recover one known interval; this proves it can recover four in a row
without confusing them -- crucially including the case where the gaps DIFFER per
turn, so a bug that reports the same value four times, or that matches turn 3's
reference against turn 2's audio, cannot pass.

Built from the committed speech clips exactly like generate.py, and using the
same sample-index convention:

    t1[i]  one PAST the last speech sample of our turn i (exclusive end)
    t2[i]  the first speech sample of the vendor's reply to turn i (inclusive)
    TTFAB[i] == t2[i] - t1[i], exact by construction
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..codec import mulaw_roundtrip
from ..resample import ms_to_samples, resample_int16
from .generate import (
    LEAD_IN_MS,
    RECORDING_RATE,
    SOURCE_RATE,
    TRAILING_SILENCE_MS,
    _load,
    _place,
)

# The caller's script: turn 1 reuses stimulus_hours so this fixture exercises the
# same clip the live bench sends.
TURN_CLIPS = ("stimulus_hours", "turn_sunday", "turn_holidays", "turn_weekdays")

# Deliberately all different, and deliberately rising -- a fake "context cost"
# curve. If the analyzer reported one turn's value for all four, or paired the
# wrong reference with the wrong reply, these would not come back distinct.
DEFAULT_GAPS_MS = (500.0, 700.0, 900.0, 1100.0)

# How long the vendor's reply runs, and how long we wait after it before speaking
# again. The pause is what the live caller's REPLY_HANGOVER has to sit inside.
REPLY_MS = 1600.0
INTER_TURN_PAUSE_MS = 1400.0


@dataclass
class MultiTurnFixture:
    name: str = "multiturn_4"
    gaps_ms: tuple[float, ...] = DEFAULT_GAPS_MS
    rate: int = RECORDING_RATE
    companded: bool = True
    # Simulates our own audio losing samples in flight on ONE turn, so the
    # per-turn drift check is exercised without wrecking the other turns.
    drop_ms_on_turn: tuple[int, float] | None = None
    # Vendor speaks during our silence between two turns (the idle-prompt hazard,
    # now possible BETWEEN turns rather than only before turn 1). Placed
    # `idle_after_reply_ms` after the previous reply ends -- which must exceed the
    # utterance-merge gap (700 ms) or the prompt is grouped INTO that reply and is
    # correctly not a separate utterance at all.
    idle_prompt_before_turn: int | None = None
    idle_after_reply_ms: float = 1200.0
    inter_turn_pause_ms: float = INTER_TURN_PAUSE_MS
    # We start this turn while the previous reply is still going -- the main risk
    # the turn loop introduces. Negative pause for that one gap.
    barge_on_turn: int | None = None
    truth: dict = field(default_factory=dict)


def build_multiturn(fx: MultiTurnFixture, out_dir: Path) -> dict:
    """Assemble the fixture. Returns the truth dict, also written as JSON."""
    rate = fx.rate

    greeting = _load("greeting")
    reply = _load("response")
    filler = _load("filler")
    turns = [_load(clip) for clip in TURN_CLIPS]

    if rate != SOURCE_RATE:
        greeting = resample_int16(greeting, SOURCE_RATE, rate)
        reply = resample_int16(reply, SOURCE_RATE, rate)
        filler = resample_int16(filler, SOURCE_RATE, rate)
        turns = [resample_int16(t, SOURCE_RATE, rate) for t in turns]

    reply = reply[: ms_to_samples(REPLY_MS, rate)]
    filler = filler[: ms_to_samples(250.0, rate)]

    lead = ms_to_samples(LEAD_IN_MS, rate)
    trailing = ms_to_samples(TRAILING_SILENCE_MS, rate)
    hangover = ms_to_samples(900.0, rate)
    pause = ms_to_samples(fx.inter_turn_pause_ms, rate)

    # --- lay out the conversation on a single timeline ---------------------- #
    # References are what we INTENDED to send; `sent` is what lands on the tape.
    # They differ only on the turn that simulates packet loss, which is the whole
    # point of the drift check.
    references: list[np.ndarray] = []
    plan: list[dict] = []

    cursor = lead + len(greeting) + hangover
    greeting_span = (lead, lead + len(greeting))

    for index, intended in enumerate(turns, start=1):
        sent = intended
        if fx.drop_ms_on_turn and fx.drop_ms_on_turn[0] == index:
            n_drop = ms_to_samples(fx.drop_ms_on_turn[1], rate)
            cut = len(intended) // 2
            sent = np.concatenate([intended[:cut], intended[cut + n_drop:]])

        idle_start = None
        if fx.idle_prompt_before_turn == index:
            # Far enough after the previous reply that utterance grouping keeps it
            # separate (see idle_after_reply_ms).
            idle_start = (cursor - pause) + ms_to_samples(fx.idle_after_reply_ms, rate)

        stimulus_start = cursor
        t1 = stimulus_start + len(sent)
        gap = ms_to_samples(fx.gaps_ms[index - 1], rate)
        t2 = t1 + gap

        plan.append({
            "index": index,
            "clip": TURN_CLIPS[index - 1],
            "stimulus_start": stimulus_start,
            "t1": t1,
            "t2": t2,
            "sent": sent,
            "idle_start": idle_start,
            "gap_ms": fx.gaps_ms[index - 1],
        })
        references.append(intended)
        # Overlap the NEXT turn into this reply when asked: start it 600 ms before
        # the reply finishes.
        if fx.barge_on_turn == index + 1:
            cursor = t2 + len(reply) - ms_to_samples(600.0, rate)
        else:
            cursor = t2 + len(reply) + pause

    total = cursor + ms_to_samples(500.0, rate)
    near = np.zeros(total, dtype=np.int64)
    far = np.zeros(total, dtype=np.int64)

    _place(far, greeting.astype(np.int64), greeting_span[0])
    for step in plan:
        _place(near, step["sent"].astype(np.int64), step["stimulus_start"])
        _place(far, reply.astype(np.int64), step["t2"])
        if step["idle_start"] is not None:
            _place(far, filler.astype(np.int64), step["idle_start"])

    near = np.clip(near, -32768, 32767).astype(np.int16)
    far = np.clip(far, -32768, 32767).astype(np.int16)
    if fx.companded:
        near = mulaw_roundtrip(near)
        far = mulaw_roundtrip(far)

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{fx.name}.wav"
    sf.write(wav_path, np.stack([near, far], axis=1), rate, subtype="PCM_16")

    reference_names = []
    for index, intended in enumerate(references, start=1):
        ref = mulaw_roundtrip(intended) if fx.companded else intended
        name = f"{fx.name}.ref_t{index}.wav"
        sf.write(out_dir / name, ref, rate, subtype="PCM_16")
        reference_names.append(name)

    truth = {
        "name": fx.name,
        "rate": rate,
        "companded": fx.companded,
        "channels": {"near": 0, "far": 1},
        "stimulus_trailing_silence_ms": TRAILING_SILENCE_MS,
        "greeting_onset": greeting_span[0],
        "greeting_end": greeting_span[1],
        "wav": wav_path.name,
        "turns": [
            {
                "index": step["index"],
                "clip": step["clip"],
                "reference_wav": reference_names[step["index"] - 1],
                "stimulus_start": step["stimulus_start"],
                "t1": step["t1"],
                "t2": step["t2"],
                "ttfab_ms": step["gap_ms"],
                "idle_prompt_start": step["idle_start"],
            }
            for step in plan
        ],
    }
    (out_dir / f"{fx.name}.truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


MULTITURN_FIXTURES = [
    MultiTurnFixture(name="multiturn_4"),
    MultiTurnFixture(name="multiturn_4_drift", drop_ms_on_turn=(3, 40.0)),
    MultiTurnFixture(name="multiturn_4_idle", idle_prompt_before_turn=3,
                     inter_turn_pause_ms=3000.0, idle_after_reply_ms=1200.0),
    MultiTurnFixture(name="multiturn_4_uncompanded", companded=False),
    MultiTurnFixture(name="multiturn_4_barge", barge_on_turn=3),
]


def build_all_multiturn(out_dir: Path) -> dict[str, dict]:
    return {fx.name: build_multiturn(fx, out_dir) for fx in MULTITURN_FIXTURES}
