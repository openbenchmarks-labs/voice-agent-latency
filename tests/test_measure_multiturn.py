"""Gate A for multi-turn: four constructed gaps in one recording, recovered.

Single-turn Gate A proved the analyzer can recover one known interval. This
proves it can recover four in sequence without confusing them -- and because the
four gaps DIFFER (500/700/900/1100 ms), a bug that reports one turn's value for
all four, or pairs turn 3's reference with turn 2's reply, cannot pass.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from analyzer.fixtures.multiturn import MULTITURN_FIXTURES, build_all_multiturn
from analyzer.measure import measure_call, measure_recording, measure_turns

TOLERANCE_MS = 5.0


@pytest.fixture(scope="module")
def mt(tmp_path_factory):
    out = tmp_path_factory.mktemp("multiturn")
    truths = build_all_multiturn(out)
    return out, truths


def _load(out, truth):
    audio, rate = sf.read(out / truth["wav"], dtype="int16", always_2d=True)
    near = np.ascontiguousarray(audio[:, truth["channels"]["near"]])
    far = np.ascontiguousarray(audio[:, truth["channels"]["far"]])
    refs = [sf.read(out / t["reference_wav"], dtype="int16")[0]
            for t in truth["turns"]]
    return near, far, refs, rate


def _measure(mt, name):
    out, truths = mt
    truth = truths[name]
    near, far, refs, rate = _load(out, truth)
    return measure_turns(near, far, refs, rate), truth, rate


def test_gate_a_multiturn_all_fixtures(mt):
    """The gate: every usable turn within tolerance of its constructed gap."""
    out, truths = mt
    failures = []
    worst = 0.0

    for name, truth in truths.items():
        result, _, rate = _measure(mt, name)
        if result.discard_reason is not None:
            failures.append(f"{name}: whole call discarded ({result.discard_reason})")
            continue
        assert len(result.turns) == len(truth["turns"]), name

        for turn, want in zip(result.turns, truth["turns"]):
            if not turn.usable:
                continue
            error = turn.ttfab_onset_ms - want["ttfab_ms"]
            worst = max(worst, abs(error))
            if abs(error) > TOLERANCE_MS:
                failures.append(f"{name} turn {turn.index}: error {error:+.2f} ms")

    assert not failures, "multi-turn Gate A failed:\n  " + "\n  ".join(failures)
    assert worst <= TOLERANCE_MS


def test_four_distinct_gaps_come_back_distinct(mt):
    """Guards against the failure that would look like success.

    If the analyzer reused one turn's interval, or matched references to the wrong
    replies, the recovered values would not track the constructed ones.
    """
    result, truth, _ = _measure(mt, "multiturn_4")
    got = [t.ttfab_onset_ms for t in result.turns]
    want = [t["ttfab_ms"] for t in truth["turns"]]

    assert want == [500.0, 700.0, 900.0, 1100.0]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=TOLERANCE_MS)
    # Strictly increasing, like the truth.
    assert got == sorted(got)
    assert len(set(round(g) for g in got)) == 4


def test_t1_is_sample_exact_on_every_turn(mt):
    result, truth, rate = _measure(mt, "multiturn_4")
    for turn, want in zip(result.turns, truth["turns"]):
        expected_ms = want["t1"] * 1000.0 / rate
        assert turn.t1_ms == pytest.approx(expected_ms, abs=0.2), f"turn {turn.index}"


def test_turns_are_located_in_order(mt):
    result, _, _ = _measure(mt, "multiturn_4")
    t1s = [t.t1_ms for t in result.turns]
    assert t1s == sorted(t1s)
    assert all(t.discard_reason != "out_of_order" for t in result.turns)


def test_drift_is_isolated_to_the_damaged_turn(mt):
    """40 ms excised from turn 3's audio must not contaminate turns 1, 2, 4."""
    result, _, _ = _measure(mt, "multiturn_4_drift")
    drift = {t.index: t.drift_ms for t in result.turns}
    assert abs(abs(drift[3]) - 40.0) < 3.0, drift
    for index in (1, 2, 4):
        assert abs(drift[index]) < 3.0, f"turn {index} drift {drift[index]}"
    # Still measurable: 40 ms is under the 50 ms discard threshold.
    assert all(t.usable for t in result.turns)


def test_interrupting_a_reply_is_caught_and_localised(mt):
    """The main risk the turn loop introduces.

    Turn 3 starts 600 ms before turn 2's reply finishes. That turn is discarded as
    `barged_reply`; the turns either side stay usable, because per-turn discards
    are the whole point.
    """
    result, _, _ = _measure(mt, "multiturn_4_barge")
    by_index = {t.index: t for t in result.turns}

    assert by_index[3].discard_reason == "barged_reply"
    assert any("overlapped_reply_of_turn_2" in f for f in by_index[3].flags)
    for index in (1, 2, 4):
        assert by_index[index].usable, f"turn {index} should survive"


def test_idle_prompt_between_turns_is_flagged_not_discarded(mt):
    """An extra vendor utterance during our silence does not corrupt the interval.

    TTFAB ends at the FIRST onset after our speech, so a later re-prompt cannot
    move it. What it does change is the conversation history the next turn answers
    against -- which is worth recording, not discarding.
    """
    result, truth, _ = _measure(mt, "multiturn_4_idle")
    by_index = {t.index: t for t in result.turns}

    assert any("vendor_spoke_again_before_next_turn" in f
               for f in by_index[2].flags), by_index[2].flags
    assert by_index[2].usable
    # And the measurement is still right.
    want = {t["index"]: t["ttfab_ms"] for t in truth["turns"]}
    assert by_index[2].ttfab_onset_ms == pytest.approx(want[2], abs=TOLERANCE_MS)


def test_uncompanded_path_matches_companded(mt):
    companded, truth_c, _ = _measure(mt, "multiturn_4")
    linear, truth_l, _ = _measure(mt, "multiturn_4_uncompanded")
    for a, b in zip(companded.turns, linear.turns):
        assert a.ttfab_onset_ms == pytest.approx(b.ttfab_onset_ms, abs=2.0)


def test_greeting_and_ttfg_come_from_turn_one_only(mt):
    """Greeting logic stays keyed to the first turn, as in the single-turn case."""
    result, truth, rate = _measure(mt, "multiturn_4")
    assert result.greeting_onset_ms == pytest.approx(
        truth["greeting_onset"] * 1000.0 / rate, abs=20.0)
    assert result.ttfg_ms is not None


def test_top_level_fields_mirror_turn_one(mt):
    """Every existing consumer reads the top level; it must stay populated."""
    result, _, _ = _measure(mt, "multiturn_4")
    first = result.turns[0]
    assert result.ttfab_onset_ms == first.ttfab_onset_ms
    assert result.t1_ms == first.t1_ms
    assert result.t2_ms == first.t2_ms
    assert result.psr == first.psr
    assert result.drift_ms == first.drift_ms


def test_usable_turns_helper(mt):
    result, _, _ = _measure(mt, "multiturn_4_barge")
    assert len(result.usable_turns) == 3
    assert all(t.usable for t in result.usable_turns)


def test_whole_call_failure_yields_no_usable_turns(mt):
    out, truths = mt
    truth = truths["multiturn_4"]
    near, far, refs, rate = _load(out, truth)
    result = measure_turns(near, np.zeros(0, dtype=np.int16), refs, rate)
    assert result.discard_reason == "audio_missing"
    assert result.usable_turns == []


def test_result_is_json_serialisable_with_turns(mt):
    from dataclasses import asdict

    result, _, _ = _measure(mt, "multiturn_4")
    payload = json.loads(json.dumps(asdict(result)))
    assert len(payload["turns"]) == 4
    assert payload["turns"][2]["index"] == 3


# --------------------------------------------------------------- compatibility


def test_single_turn_still_goes_through_the_same_path(fixture_dir, truths):
    """measure_recording is now the N=1 case; it must behave exactly as before."""
    from tests.conftest import load_fixture

    truth = truths["clean_500ms"]
    near, far, ref, rate = load_fixture(fixture_dir, truth)
    result = measure_recording(near, far, ref, rate)

    assert len(result.turns) == 1
    assert result.ttfab_onset_ms == pytest.approx(truth["ttfab_ms"], abs=TOLERANCE_MS)
    assert result.turns[0].ttfab_onset_ms == result.ttfab_onset_ms


def test_single_turn_discards_still_land_on_the_call(fixture_dir, truths):
    """Pre-multi-turn consumers check `result.discard_reason`; for a one-turn call
    the turn's failure must still surface there."""
    from tests.conftest import load_fixture

    truth = truths["clean_500ms"]
    _, far, ref, rate = load_fixture(fixture_dir, truth)
    rng = np.random.default_rng(3)
    near = rng.integers(-2000, 2000, size=len(far), dtype=np.int16)

    result = measure_recording(near, far, ref, rate)
    assert result.discard_reason == "unlocatable"
    assert result.turns[0].discard_reason == "unlocatable"
    assert any("psr=" in f for f in result.flags)


def test_missing_turn_reference_is_discarded_not_crashed(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "call_id": "c", "turns": [{"index": 1, "reference_wav": "nope.wav"}],
    }))
    sf.write(tmp_path / "recording.wav",
             np.zeros((8000, 2), dtype=np.int16), 8000, subtype="PCM_16")
    sf.write(tmp_path / "our_audio.wav",
             np.zeros(800, dtype=np.int16), 8000, subtype="PCM_16")

    result = measure_call(tmp_path)
    assert result.discard_reason == "audio_missing"
    assert any("missing_reference" in f for f in result.flags)


def test_fixture_set_covers_the_hazards():
    """Documents what the multi-turn gate is actually testing."""
    names = {f.name for f in MULTITURN_FIXTURES}
    assert names == {
        "multiturn_4",               # baseline, four distinct gaps
        "multiturn_4_drift",         # audio lost in flight on one turn
        "multiturn_4_idle",          # vendor re-prompts between turns
        "multiturn_4_uncompanded",   # 16-bit linear path
        "multiturn_4_barge",         # we interrupt a reply
    }
