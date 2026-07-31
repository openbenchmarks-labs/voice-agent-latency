"""Scripted-dialog mode: t1 without a reference, and the ways it can go wrong.

Reference mode could always answer "where did our speech end?" exactly, because
it knew what our speech looked like. Here it has to find out, so most of these
tests are about the failure modes that creates -- and about refusing to report a
number when the answer is not trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyzer.fixtures.dialog import (
    T1_TOLERANCE_MS,
    TTFAB_TOLERANCE_MS,
    DialogFixture,
    build_all_dialog,
    build_dialog,
)
from analyzer.measure import ANALYZER_VERSION, measure_call
from analyzer.resample import samples_to_ms


@pytest.fixture(scope="module")
def dialog_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("dialog_fixtures")
    return out, build_all_dialog(out)


def measured(truths, name):
    return measure_call(Path(truths[name]["call_dir"])), truths[name]


def t1_error_ms(turn, row, rate):
    return turn.t1_ms - samples_to_ms(row["t1"], rate)


# --------------------------------------------------------------------------- #
# Gate A
# --------------------------------------------------------------------------- #


def test_every_fixture_matches_its_constructed_truth(dialog_dir):
    _, truths = dialog_dir
    for name, truth in truths.items():
        result = measure_call(Path(truth["call_dir"]))
        assert result.discard_reason == truth["expect_call_discard"], name
        if result.discard_reason:
            continue
        for turn, row in zip(result.turns, truth["turns"]):
            assert turn.discard_reason == row["expect_discard"], f"{name} t{turn.index}"
            if turn.discard_reason:
                continue
            assert abs(turn.ttfab_onset_ms - row["ttfab_ms"]) <= TTFAB_TOLERANCE_MS
            assert abs(t1_error_ms(turn, row, truth["rate"])) <= T1_TOLERANCE_MS


def test_four_different_gaps_come_back_four_different_values(dialog_dir):
    """A bug that reports one turn's value for every turn, or pairs turn 3's
    speech with turn 2's reply, cannot survive rising gaps."""
    _, truths = dialog_dir
    result, truth = measured(truths, "dlg_clean")
    values = [t.ttfab_onset_ms for t in result.turns]
    assert values == sorted(values)
    assert len(set(values)) == 4


def test_turns_are_reported_in_order_with_advancing_t1(dialog_dir):
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_clean")
    assert [t.index for t in result.turns] == [1, 2, 3, 4]
    t1s = [t.t1_ms for t in result.turns]
    assert t1s == sorted(t1s)


# --------------------------------------------------------------------------- #
# What replaces the correlation diagnostics
# --------------------------------------------------------------------------- #


def test_correlation_diagnostics_are_null_rather_than_invented(dialog_dir):
    """psr and drift describe a matched filter. There is no matched filter on
    this path, and a plausible-looking number would be worse than nothing."""
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_clean")
    for turn in result.turns:
        assert turn.psr is None and turn.psr_global is None
        assert turn.drift_ms is None
    assert result.psr is None and result.drift_ms is None


def test_the_version_records_that_t1_changed_meaning(dialog_dir):
    """Results from the two paths must never be pooled: t1 means a different
    thing and carries a different error class."""
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_clean")
    # Pinned to the MAJOR: 2.x is the detected-t1 era. Minors move when a rule
    # changes a verdict (2.1.0 relaxed idle_filler on clipped greetings), and
    # pinning the full string would make every such fix a test edit.
    assert result.analyzer_version == ANALYZER_VERSION
    assert ANALYZER_VERSION.startswith("2.")


def test_our_turn_start_is_still_reported_for_the_timeline(dialog_dir):
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_clean")
    for turn in result.turns:
        assert turn.stimulus_start_ms is not None
        assert turn.stimulus_start_ms < turn.t1_ms


# --------------------------------------------------------------------------- #
# Hazards a detected t1 introduces
# --------------------------------------------------------------------------- #


def test_trailing_silence_does_not_drag_t1_into_the_padding(dialog_dir):
    """t1 is the last speech sample, not the end of the file. A refinement that
    walks into digital silence would report every padded turn late."""
    _, truths = dialog_dir
    result, truth = measured(truths, "dlg_trailing_silence")
    padded = result.turns[1]
    assert abs(t1_error_ms(padded, truth["turns"][1], truth["rate"])) <= T1_TOLERANCE_MS


def test_a_pause_inside_one_turn_is_reunited_not_split(dialog_dir):
    """TTS pauses at punctuation. A 350 ms gap exceeds the silence threshold and
    arrives as two segments; treating it as two turns would put t1 mid-sentence
    and shift every later pairing by one."""
    _, truths = dialog_dir
    result, truth = measured(truths, "dlg_midturn_pause")
    assert result.discard_reason is None
    assert len(result.turns) == 4
    assert abs(t1_error_ms(result.turns[1], truth["turns"][1],
                           truth["rate"])) <= T1_TOLERANCE_MS


def test_a_turn_that_really_splits_refuses_the_call(dialog_dir):
    """Past the merge gap the utterance count no longer matches what we spoke,
    so turn i is not necessarily our i-th line. Better to lose the call than to
    publish a confident mis-pairing."""
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_split_turn")
    assert result.discard_reason == "turn_count_mismatch"
    assert any("near_utterances=5" in f for f in result.flags)


def test_talking_over_each_other_discards_that_turn_only(dialog_dir):
    """If the vendor was still speaking as we finished, "the end of the caller's
    speech" is not a clean anchor -- but turns 1, 2 and 4 are untouched."""
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_double_talk")
    assert result.turns[2].discard_reason == "double_talk"
    assert [t.discard_reason for t in result.turns] == [None, None, "double_talk", None]


def test_our_own_echo_on_their_channel_is_not_mistaken_for_a_reply(dialog_dir):
    """The most dangerous artifact on this path: leaked copies of our own
    speech start BEFORE the vendor answers, so believing them yields a wildly
    early -- often negative -- TTFAB that still looks like a number."""
    _, truths = dialog_dir
    result, truth = measured(truths, "dlg_far_bleed")
    assert any(f.startswith("far_channel_bleed=") for f in result.flags)
    for turn, row in zip(result.turns, truth["turns"]):
        assert turn.ttfab_onset_ms > 0
        assert abs(turn.ttfab_onset_ms - row["ttfab_ms"]) <= TTFAB_TOLERANCE_MS


def test_a_turn_with_no_reply_is_no_response_and_costs_only_itself(dialog_dir):
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_no_reply")
    assert result.turns[2].discard_reason == "no_response"
    assert result.turns[2].ttfab_onset_ms is None
    assert all(t.discard_reason is None for t in result.turns if t.index != 3)


def test_an_idle_prompt_between_turns_is_flagged_not_discarded(dialog_dir):
    """The measured interval still ends at the FIRST reply onset, so the number
    is right -- but the conversation carries an extra vendor utterance and later
    turns answer a slightly different history."""
    _, truths = dialog_dir
    result, _ = measured(truths, "dlg_idle_between")
    assert result.discard_reason is None
    assert any("vendor_spoke_again_before_next_turn" in f
               for turn in result.turns for f in turn.flags)


def test_companding_does_not_move_the_numbers(dialog_dir):
    """The PSTN companded path and a clean one must agree, or the measurement is
    reading the codec rather than the speech."""
    _, truths = dialog_dir
    companded, _ = measured(truths, "dlg_clean")
    clean, _ = measured(truths, "dlg_uncompanded")
    for a, b in zip(companded.turns, clean.turns):
        assert abs(a.ttfab_onset_ms - b.ttfab_onset_ms) <= 2.0


# --------------------------------------------------------------------------- #
# The channel map
# --------------------------------------------------------------------------- #


def test_an_inverted_channel_map_is_caught_rather_than_measured(tmp_path):
    """Plivo does not document which stereo channel is which party, so the map
    is pinned by a probe. If that assumption is ever wrong the numbers would
    measure the vendor's speech-end against our own reply -- plausible nonsense.
    We know how many times WE spoke, so the count catches it."""
    truth = build_dialog(DialogFixture(name="dlg_inverted"), tmp_path)
    call_dir = Path(truth["call_dir"])
    meta = json.loads((call_dir / "metadata.json").read_text())
    meta["channel_map"] = {"near": 1, "far": 0, "source": "wrong on purpose"}
    (call_dir / "metadata.json").write_text(json.dumps(meta))

    result = measure_call(call_dir)
    assert result.discard_reason == "channel_map_suspect"


def test_a_greeting_clipped_by_the_recorder_does_not_discard_the_call(tmp_path):
    """A vendor fast enough to greet before the recorder armed had EVERY call
    thrown away (bench-vapi-20260730-141015): the tape opens mid-greeting, the
    greeting therefore arrives already split, and two fragments read as
    "greeting + idle prompt". That run was a flawless four-turn conversation
    with three clean per-turn measurements in it.

    The count is only meaningful when the greeting's start is on the tape.
    """
    import numpy as np
    import soundfile as sf

    from analyzer.measure import CLIPPED_START_RMS, CLIPPED_START_WINDOW_MS
    from analyzer.resample import ms_to_samples

    truth = build_dialog(DialogFixture(name="dlg_clipped_greeting"), tmp_path)
    call_dir = Path(truth["call_dir"])
    audio, rate = sf.read(call_dir / "recording.wav", dtype="int16", always_2d=True)

    # Cut the tape so it opens INSIDE the greeting's speech, the way a recorder
    # that armed late does. Pick the loudest window in the greeting so the cut
    # genuinely lands mid-utterance rather than in an intra-word pause.
    far = audio[:, 1].astype(np.float64)
    window = ms_to_samples(CLIPPED_START_WINDOW_MS, rate)
    starts = range(truth["greeting_onset"],
                   max(truth["greeting_onset"] + 1, truth["greeting_end"] - window),
                   window)
    cut = max(starts, key=lambda s: np.sqrt(np.mean(far[s:s + window] ** 2)))
    assert np.sqrt(np.mean(far[cut:cut + window] ** 2)) > CLIPPED_START_RMS, \
        "the cut must land in speech, or it is not the condition under test"
    sf.write(call_dir / "recording.wav", np.ascontiguousarray(audio[cut:]), rate,
             subtype="PCM_16")

    result = measure_call(call_dir)
    assert "recording_started_mid_speech" in result.flags
    assert result.discard_reason != "idle_filler"
    # The check is recorded as skipped, never as passed.
    if any("pre_stimulus_utterances" in f for f in result.flags):
        assert "idle_filler_unassessable_clipped_greeting" in result.flags
    # TTFG stays withheld -- the greeting's start genuinely is not on the tape.
    assert result.ttfg_ms is None


def test_a_silent_call_is_dead_air_not_a_pairing_failure(tmp_path):
    """Five calls in one live run connected and carried no audio at all
    (bench-telnyx-20260730-110251). Reported as turn_count_mismatch they sent
    the reader hunting for an analyzer bug; the call simply had no speech in
    it, and that is a finding about the call."""
    import json

    import numpy as np
    import soundfile as sf

    truth = build_dialog(DialogFixture(name="dlg_silent"), tmp_path)
    call_dir = Path(truth["call_dir"])
    audio, rate = sf.read(call_dir / "recording.wav", dtype="int16", always_2d=True)
    sf.write(call_dir / "recording.wav", np.zeros_like(audio[: rate * 6]), rate,
             subtype="PCM_16")

    result = measure_call(call_dir)
    assert result.discard_reason == "dead_air"
    assert any(f.startswith("duration_s=") for f in result.flags)


def test_a_mono_recording_cannot_be_measured(tmp_path):
    import numpy as np
    import soundfile as sf

    truth = build_dialog(DialogFixture(name="dlg_mono"), tmp_path)
    call_dir = Path(truth["call_dir"])
    audio, rate = sf.read(call_dir / "recording.wav", dtype="int16", always_2d=True)
    sf.write(call_dir / "recording.wav", np.ascontiguousarray(audio[:, 0]), rate,
             subtype="PCM_16")

    result = measure_call(call_dir)
    assert result.discard_reason == "audio_missing"
    assert "recording_not_stereo" in result.flags


# --------------------------------------------------------------------------- #
# Metadata passthrough
# --------------------------------------------------------------------------- #


def test_answer_accuracy_rides_along_without_touching_the_latency(tmp_path):
    """A wrong answer is a different failure from a slow one. It must be
    visible, and it must not silently discard a perfectly good measurement."""
    truth = build_dialog(DialogFixture(name="dlg_verify"), tmp_path)
    call_dir = Path(truth["call_dir"])
    meta = json.loads((call_dir / "metadata.json").read_text())
    meta["turns"][0].update({"case_id": "price-basic", "answer_verified": True})
    meta["turns"][1].update({"case_id": "free-trial", "answer_verified": False})
    (call_dir / "metadata.json").write_text(json.dumps(meta))

    result = measure_call(call_dir)
    assert "answer_verified" in result.turns[0].flags
    assert "case=price-basic" in result.turns[0].flags
    assert "answer_not_verified" in result.turns[1].flags
    assert all(t.discard_reason is None for t in result.turns)


def test_reference_mode_is_untouched_by_the_new_path(dialog_dir):
    """No `mode` key means a run recorded before modes existed, which must keep
    re-analyzing exactly as it did."""
    _, truths = dialog_dir
    call_dir = Path(truths["dlg_clean"]["call_dir"])
    meta = json.loads((call_dir / "metadata.json").read_text())
    del meta["mode"]
    (call_dir / "metadata.json").write_text(json.dumps(meta))
    try:
        result = measure_call(call_dir)
        # Reference mode needs reference audio, which a dialog call never has.
        assert result.discard_reason == "audio_missing"
        assert any(f.startswith("missing_reference") for f in result.flags)
    finally:
        meta["mode"] = "scripted_dialog"
        (call_dir / "metadata.json").write_text(json.dumps(meta))
