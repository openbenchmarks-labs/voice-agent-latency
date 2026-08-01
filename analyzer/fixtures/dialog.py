"""Ground truth for scripted-dialog mode: known gaps, and no reference audio.

Reference-mode Gate A proves the analyzer can recover a known interval when it
knows exactly what our audio looked like. This proves it can recover the same
intervals when it does NOT -- when the only way to find the end of our own
speech is to detect it, which is the situation every live Plivo dialog creates.

Same sample-index convention as the other fixture families:

    t1[i]  one PAST the last speech sample of our turn i (exclusive end)
    t2[i]  the first speech sample of the vendor's reply to turn i (inclusive)
    TTFAB[i] == t2[i] - t1[i], exact by construction

No `ref_t*.wav` files are written: their absence is the point. What is written
instead is a metadata.json alongside the recording, so the fixture is measured
through `measure_call` -- the real entry point, including its mode gate and
channel handling -- rather than through an inner function no live call uses.

The hazard fixtures exist because a detector-based t1 has failure modes a
matched filter never had. Each one is a way the near channel can lie about
where our turn ended: a pause long enough to look like the end, trailing
silence that is not speech, the vendor talking across our tail, our own audio
bleeding into their channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..codec import mulaw_roundtrip
from ..resample import ms_to_samples, resample_int16, samples_to_ms
from .generate import LEAD_IN_MS, RECORDING_RATE, SOURCE_RATE, _load, _place

# Our lines. In a live call these are Plivo TTS; here they are the same committed
# clips the other fixtures use, which is enough -- what is under test is the
# detector, not the voice.
TURN_CLIPS = ("stimulus_hours", "turn_sunday", "turn_holidays", "turn_weekdays")

# Deliberately all different and rising: a bug that reports one turn's value for
# every turn, or that pairs turn 3's speech with turn 2's reply, cannot pass.
DEFAULT_GAPS_MS = (500.0, 700.0, 900.0, 1100.0)

REPLY_MS = 1600.0
INTER_TURN_PAUSE_MS = 1400.0
GREETING_HANGOVER_MS = 900.0

#: Gate tolerance for this family. t1 is a detected speech offset here, not a
#: sample-exact correlation peak, so reference mode's tolerance had no reason to
#: carry over -- these are the MEASURED errors plus headroom, not aspirations:
#:
#:     worst |t1 error|      3.0 ms
#:     worst |TTFAB error|   4.0 ms   (t1 and t2 errors are independent)
#:
#: Recomputed by attribution_table() on every Gate A run. If real errors creep
#: up to these bounds, the bound is not what should move.
TTFAB_TOLERANCE_MS = 6.0
T1_TOLERANCE_MS = 5.0


@dataclass
class DialogFixture:
    name: str = "dlg_clean"
    gaps_ms: tuple[float, ...] = DEFAULT_GAPS_MS
    rate: int = RECORDING_RATE
    companded: bool = True
    inter_turn_pause_ms: float = INTER_TURN_PAUSE_MS

    # --- hazards ---------------------------------------------------------- #
    #: Digital silence appended to one of our turns. t1 is still the last SPEECH
    #: sample, so a refinement that drifts into the pad fails this.
    trailing_silence_on_turn: tuple[int, float] | None = None
    #: A pause spliced INSIDE one of our turns. Below the merge gap it must be
    #: reunited into one turn; above it, the turn splits and the call must be
    #: refused rather than measured against the wrong pairing.
    pause_inside_turn: tuple[int, float] | None = None
    #: The vendor's reply starts before our turn ends (they talked over us).
    overlap_reply_on_turn: int | None = None
    #: Our own audio bleeding into the vendor's channel at this level, in dB.
    #: If the far-channel detector fires on the bleed, every TTFAB reads early.
    far_bleed_db: float | None = None
    #: The vendor never answers this turn.
    no_reply_on_turn: int | None = None
    #: The vendor speaks again between two turns (idle re-prompt).
    idle_prompt_before_turn: int | None = None
    idle_after_reply_ms: float = 1200.0
    #: A pause spliced INSIDE the greeting, before we have said anything. Past the
    #: merge gap the greeting arrives as two utterances, which is indistinguishable
    #: by COUNT from an idle re-prompt -- so this is the fixture that pins which
    #: way dialog mode resolves that ambiguity.
    split_greeting_pause_ms: float | None = None
    #: The vendor re-prompts before our FIRST turn, in the window our own TTS
    #: latency creates. The genuine article, as opposed to a split greeting.
    idle_prompt_before_first_turn: bool = False
    idle_before_first_turn_ms: float = 2000.0
    #: Silence between the greeting ending and our first turn. Widened by the
    #: fixtures that need room to place something inside that window.
    greeting_hangover_ms: float = GREETING_HANGOVER_MS
    #: Flag prefixes the result must carry. A fixture whose whole point is "kept,
    #: but recorded" passes trivially if only the verdict is checked.
    expect_flags: tuple[str, ...] = ()

    truth: dict = field(default_factory=dict)


def build_dialog(fx: DialogFixture, out_dir: Path) -> dict:
    """Assemble one fixture as a call directory. Returns the truth dict."""
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
    pause = ms_to_samples(fx.inter_turn_pause_ms, rate)

    # The greeting, which is not always one utterance on the tape. `extent` spans
    # from its first speech sample to its last, pause included.
    greeting_pieces: list[tuple[int, np.ndarray]] = []
    if fx.split_greeting_pause_ms is not None:
        cut = len(greeting) // 2
        gap_n = ms_to_samples(fx.split_greeting_pause_ms, rate)
        greeting_pieces.append((lead, greeting[:cut]))
        greeting_pieces.append((lead + cut + gap_n, greeting[cut:]))
        extent = len(greeting) + gap_n
    else:
        greeting_pieces.append((lead, greeting))
        extent = len(greeting)

    greeting_span = (lead, lead + extent)
    cursor = lead + extent + ms_to_samples(fx.greeting_hangover_ms, rate)

    # A re-prompt in the pre-stimulus window. Placed relative to the greeting's
    # END so it reads as a reaction to our silence, which is what makes it the
    # genuine article rather than a second greeting phrase.
    pre_idle_start = None
    if fx.idle_prompt_before_first_turn:
        pre_idle_start = greeting_span[1] + ms_to_samples(
            fx.idle_before_first_turn_ms, rate)
        assert pre_idle_start + ms_to_samples(250.0, rate) < cursor, (
            "the pre-stimulus idle prompt must finish before our first turn, or "
            "it is not in the window the rule watches")

    plan: list[dict] = []
    for index, spoken in enumerate(turns, start=1):
        # Our turn as it lands on the tape. `pieces` is what is actually placed;
        # `speech_end` is the last SPEECH sample regardless of padding.
        pieces: list[tuple[int, np.ndarray]] = []
        offset = 0
        if fx.pause_inside_turn and fx.pause_inside_turn[0] == index:
            cut = len(spoken) // 2
            gap_n = ms_to_samples(fx.pause_inside_turn[1], rate)
            pieces.append((0, spoken[:cut]))
            pieces.append((cut + gap_n, spoken[cut:]))
            offset = cut + gap_n + len(spoken[cut:])
        else:
            pieces.append((0, spoken))
            offset = len(spoken)

        speech_end = cursor + offset
        if fx.trailing_silence_on_turn and fx.trailing_silence_on_turn[0] == index:
            # Padding moves the file's end, never the speech's end.
            offset += ms_to_samples(fx.trailing_silence_on_turn[1], rate)

        gap = ms_to_samples(fx.gaps_ms[index - 1], rate)
        t2 = speech_end + gap
        if fx.overlap_reply_on_turn == index:
            # They start talking before we stop: t2 lands inside our turn, so
            # TTFAB is negative by construction and t1 is not a clean anchor.
            t2 = speech_end - ms_to_samples(200.0, rate)

        idle_start = None
        if fx.idle_prompt_before_turn == index:
            idle_start = (cursor - pause) + ms_to_samples(fx.idle_after_reply_ms, rate)

        answered = fx.no_reply_on_turn != index
        plan.append({
            "index": index,
            "clip": TURN_CLIPS[index - 1],
            "start": cursor,
            "pieces": [(cursor + at, chunk) for at, chunk in pieces],
            "t1": speech_end,
            "t2": t2 if answered else None,
            "gap_ms": fx.gaps_ms[index - 1],
            "idle_start": idle_start,
            "answered": answered,
        })

        reply_end = (t2 + len(reply)) if answered else (speech_end + gap)
        cursor = reply_end + pause

    total = cursor + ms_to_samples(500.0, rate)
    near = np.zeros(total, dtype=np.int64)
    far = np.zeros(total, dtype=np.int64)

    for at, chunk in greeting_pieces:
        _place(far, chunk.astype(np.int64), at)
    if pre_idle_start is not None:
        _place(far, filler.astype(np.int64), pre_idle_start)
    for step in plan:
        for at, chunk in step["pieces"]:
            _place(near, chunk.astype(np.int64), at)
        if step["t2"] is not None:
            _place(far, reply.astype(np.int64), step["t2"])
        if step["idle_start"] is not None:
            _place(far, filler.astype(np.int64), step["idle_start"])

    if fx.far_bleed_db is not None:
        # Echo of our own speech on their channel, as a real recording would
        # carry it. The far detector must not mistake it for the vendor.
        scale = 10.0 ** (fx.far_bleed_db / 20.0)
        far = far + (near.astype(np.float64) * scale).astype(np.int64)

    near = np.clip(near, -32768, 32767).astype(np.int16)
    far = np.clip(far, -32768, 32767).astype(np.int16)
    if fx.companded:
        near = mulaw_roundtrip(near)
        far = mulaw_roundtrip(far)

    out_dir.mkdir(parents=True, exist_ok=True)
    call_dir = out_dir / fx.name
    call_dir.mkdir(parents=True, exist_ok=True)
    sf.write(call_dir / "recording.wav", np.stack([near, far], axis=1), rate,
             subtype="PCM_16")

    expected_split = bool(fx.pause_inside_turn
                          and fx.pause_inside_turn[1] >= 700.0)
    truth = {
        "name": fx.name,
        "rate": rate,
        "companded": fx.companded,
        "channels": {"near": 0, "far": 1},
        "greeting_onset": greeting_span[0],
        "greeting_end": greeting_span[1],
        "call_dir": str(call_dir),
        # What the CALL should come back as, when a hazard invalidates it whole.
        "expect_call_discard": "turn_count_mismatch" if expected_split else None,
        "expect_flags": list(fx.expect_flags),
        "turns": [
            {
                "index": step["index"],
                "clip": step["clip"],
                "start": step["start"],
                "t1": step["t1"],
                "t2": step["t2"],
                "ttfab_ms": (samples_to_ms(step["t2"] - step["t1"], rate)
                             if step["t2"] is not None else None),
                "expect_discard": _expected_turn_discard(fx, step["index"]),
            }
            for step in plan
        ],
    }

    # The metadata a live call would have written, so the fixture goes through
    # measure_call exactly as a real run does -- mode gate, channel map and all.
    (call_dir / "metadata.json").write_text(json.dumps({
        "call_id": fx.name,
        "run_id": "gate-a-dialog",
        "kind": "fixture",
        "mode": "scripted_dialog",
        "vendor": "fixture",
        "channel_map": {"near": 0, "far": 1, "source": "fixture"},
        "turns_played": len(plan),
        "turns": [{"index": s["index"], "kind": "case", "spoken": True,
                   "text": s["clip"]} for s in plan],
    }, indent=2) + "\n")
    (call_dir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def _expected_turn_discard(fx: DialogFixture, index: int) -> str | None:
    if fx.overlap_reply_on_turn == index:
        return "double_talk"
    if fx.no_reply_on_turn == index:
        return "no_response"
    return None


DIALOG_FIXTURES = [
    DialogFixture(name="dlg_clean"),
    # Padding after the last word must not move t1: a refinement that walks into
    # digital silence would report every turn late by the pad.
    DialogFixture(name="dlg_trailing_silence",
                  trailing_silence_on_turn=(2, 400.0)),
    # A 350 ms pause exceeds MIN_SILENCE_MS (200 ms) and splits the turn into two
    # segments; grouping must reunite them, or t1 lands mid-sentence.
    DialogFixture(name="dlg_midturn_pause", pause_inside_turn=(2, 350.0)),
    # A 1200 ms pause is past the merge gap: the call now has more utterances
    # than turns and must be refused, not silently re-paired.
    DialogFixture(name="dlg_split_turn", pause_inside_turn=(2, 1200.0)),
    DialogFixture(name="dlg_double_talk", overlap_reply_on_turn=3),
    DialogFixture(name="dlg_far_bleed", far_bleed_db=-28.0),
    DialogFixture(name="dlg_no_reply", no_reply_on_turn=3),
    DialogFixture(name="dlg_idle_between", idle_prompt_before_turn=3,
                  inter_turn_pause_ms=3000.0),
    DialogFixture(name="dlg_uncompanded", companded=False),

    DialogFixture(name="dlg_split_greeting", split_greeting_pause_ms=1200.0,
                  expect_flags=("vendor_spoke_before_first_turn=",
                                "greeting_pause_ms=")),

    DialogFixture(name="dlg_idle_before_first_turn",
                  idle_prompt_before_first_turn=True,
                  greeting_hangover_ms=3500.0,
                  expect_flags=("vendor_spoke_before_first_turn=",)),
]


def build_all_dialog(out_dir: Path) -> dict[str, dict]:
    return {fx.name: build_dialog(fx, out_dir) for fx in DIALOG_FIXTURES}


def attribution_table(out_dir: Path) -> list[dict]:
    """Measured t1 error per turn on the clean fixture.

    The number that justifies TTFAB_TOLERANCE_MS, recomputed rather than
    remembered: _refine_offset's attribution choice is an empirical claim, and
    this is the evidence for it.
    """
    from ..measure import measure_call

    truth = build_dialog(DialogFixture(name="dlg_clean"), out_dir)
    result = measure_call(Path(truth["call_dir"]))
    rows = []
    for turn, expected in zip(result.turns, truth["turns"]):
        if turn.t1_ms is None:
            continue
        rows.append({
            "turn": turn.index,
            "t1_error_ms": turn.t1_ms - samples_to_ms(expected["t1"], truth["rate"]),
            "ttfab_error_ms": (None if turn.ttfab_onset_ms is None
                               else turn.ttfab_onset_ms - expected["ttfab_ms"]),
        })
    return rows
