"""Offset refinement: finding where speech STOPS.

Only onsets were ever refined, because until scripted-dialog mode the end of
our own speech came from correlation and was sample-exact. Now it is a detector
output and lands directly in every TTFAB, so it gets the same 2 ms energy pass
the onsets get -- and the same scrutiny.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer import onset as O
from analyzer.resample import ms_to_samples, samples_to_ms

RATE = 8000
TOLERANCE_MS = 5.0


def tone(duration_ms: float, rate: int = RATE, amplitude: int = 8000) -> np.ndarray:
    """A voiced-ish burst: a 220 Hz tone is loud, band-limited and deterministic."""
    n = ms_to_samples(duration_ms, rate)
    t = np.arange(n) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def silence(duration_ms: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(ms_to_samples(duration_ms, rate), dtype=np.int16)


def burst_at(lead_ms: float, speech_ms: float, tail_ms: float):
    """One burst on a silent bed. Returns (audio, true_end_sample).

    A tone is enough for the energy pass, which is all _refine_offset uses.
    """
    lead, speech = silence(lead_ms), tone(speech_ms)
    audio = np.concatenate([lead, speech, silence(tail_ms)])
    return audio, len(lead) + len(speech)


def real_speech_at(lead_ms: float, tail_ms: float, clip: str = "response"):
    """The same thing with a committed speech clip, for the tests that go
    through Silero or webrtcvad -- neither of which calls a sine wave speech."""
    from analyzer.fixtures.generate import SOURCE_RATE, _load
    from analyzer.resample import resample_int16

    speech = resample_int16(_load(clip), SOURCE_RATE, RATE)
    lead = silence(lead_ms)
    audio = np.concatenate([lead, speech, silence(tail_ms)])
    return audio, len(lead) + len(speech)


# --------------------------------------------------------------------------- #
# _refine_offset
# --------------------------------------------------------------------------- #


def test_the_refined_offset_lands_on_the_last_speech_sample():
    audio, true_end = burst_at(300.0, 500.0, 400.0)
    floor = O.noise_floor_dbfs(audio, RATE)
    # A coarse estimate on Silero's 32 ms grid, deliberately late.
    coarse = true_end + ms_to_samples(24.0, RATE)

    refined = O._refine_offset(audio, RATE, coarse, floor)
    assert abs(samples_to_ms(refined - true_end, RATE)) <= TOLERANCE_MS


def test_refinement_beats_the_coarse_estimate_it_starts_from():
    """The whole justification for the extra pass: Silero's 32 ms window plus
    its decay put the raw end tens of ms late, and that bias does not cancel."""
    audio, true_end = burst_at(300.0, 500.0, 400.0)
    floor = O.noise_floor_dbfs(audio, RATE)
    coarse = true_end + ms_to_samples(30.0, RATE)

    refined = O._refine_offset(audio, RATE, coarse, floor)
    assert abs(refined - true_end) < abs(coarse - true_end)


def test_a_coarse_estimate_that_is_early_is_also_corrected():
    audio, true_end = burst_at(300.0, 500.0, 400.0)
    floor = O.noise_floor_dbfs(audio, RATE)
    coarse = true_end - ms_to_samples(30.0, RATE)

    refined = O._refine_offset(audio, RATE, coarse, floor)
    assert abs(samples_to_ms(refined - true_end, RATE)) <= TOLERANCE_MS


def test_the_search_cannot_reach_into_the_next_utterance():
    """Without the bound a wide radius would walk forward into the next turn's
    speech and report ITS end as ours -- the pairing would still look sane."""
    lead = silence(200.0)
    first = tone(400.0)
    gap = silence(150.0)
    second = tone(400.0)
    audio = np.concatenate([lead, first, gap, second])
    true_end = len(lead) + len(first)
    next_start = true_end + len(gap)
    floor = O.noise_floor_dbfs(audio, RATE)

    refined = O._refine_offset(audio, RATE, true_end + ms_to_samples(20.0, RATE),
                               floor, earliest=len(lead), latest=next_start)
    assert refined <= next_start
    assert abs(samples_to_ms(refined - true_end, RATE)) <= TOLERANCE_MS


def test_silence_leaves_the_coarse_estimate_alone():
    """A failed refinement must degrade to 32 ms resolution, not to nonsense."""
    audio = silence(500.0)
    coarse = ms_to_samples(250.0, RATE)
    assert O._refine_offset(audio, RATE, coarse, -80.0) == coarse


def test_an_empty_search_window_is_survivable():
    audio, true_end = burst_at(100.0, 100.0, 0.0)
    floor = O.noise_floor_dbfs(audio, RATE)
    assert O._refine_offset(audio, RATE, true_end, floor,
                            earliest=true_end, latest=true_end) == true_end


# --------------------------------------------------------------------------- #
# analyze(refine_offsets=...)
# --------------------------------------------------------------------------- #


def test_segment_ends_are_refined_only_when_asked():
    """Off by default so archived reference-mode runs re-analyze identically."""
    audio, true_end = real_speech_at(300.0, 500.0)

    plain = O.analyze(audio, RATE)
    refined = O.analyze(audio, RATE, refine_offsets=True)

    assert plain.segments and refined.segments
    assert [s.start for s in plain.segments] == [s.start for s in refined.segments]
    error_plain = abs(plain.segments[0].end - true_end)
    error_refined = abs(refined.segments[0].end - true_end)
    assert error_refined <= error_plain
    assert samples_to_ms(error_refined, RATE) <= TOLERANCE_MS


def test_a_refined_end_never_precedes_its_own_onset():
    audio, _ = real_speech_at(300.0, 400.0)
    segments = O.analyze(audio, RATE, refine_offsets=True).segments
    assert segments
    for segment in segments:
        assert segment.end > segment.start


def test_refinement_is_deterministic():
    """Re-analysis must be byte-identical -- it is how a vendor re-derives a
    disputed row."""
    audio, _ = real_speech_at(300.0, 400.0)
    first = O.analyze(audio, RATE, refine_offsets=True)
    second = O.analyze(audio, RATE, refine_offsets=True)
    assert [(s.start, s.end) for s in first.segments] == \
           [(s.start, s.end) for s in second.segments]


# --------------------------------------------------------------------------- #
# The falling-edge cross-check
# --------------------------------------------------------------------------- #


def test_webrtcvad_reports_a_falling_edge_near_the_true_end():
    audio, true_end = real_speech_at(300.0, 500.0)
    offsets = O.webrtcvad_offsets(audio, RATE)
    assert offsets
    nearest = min(offsets, key=lambda o: abs(o - true_end))
    # Coarse by construction: 20 ms frames plus webrtcvad's own hangover. It is
    # a second opinion about WHERE, not a second measurement -- which is why
    # MAX_OFFSET_DISAGREEMENT_MS is set far above the onset check's 40 ms.
    assert samples_to_ms(abs(nearest - true_end), RATE) < 150.0


def test_the_offset_crosscheck_reports_distance_rather_than_correcting():
    audio, true_end = real_speech_at(300.0, 500.0)
    disagreement = O.crosscheck_offset_disagreement_ms(true_end, audio, RATE)
    assert disagreement is not None and disagreement >= 0.0


def test_an_offset_in_the_wrong_place_reads_as_a_large_disagreement():
    """What the rule is actually for: catching t1 landing in a mid-sentence
    pause or on the wrong utterance, which is a hundreds-of-ms error."""
    audio, true_end = real_speech_at(300.0, 900.0)
    honest = O.crosscheck_offset_disagreement_ms(true_end, audio, RATE)
    wrong = O.crosscheck_offset_disagreement_ms(
        true_end + ms_to_samples(600.0, RATE), audio, RATE)
    assert wrong > honest + 300.0


def test_silence_has_no_falling_edge_to_compare_against():
    assert O.crosscheck_offset_disagreement_ms(0, silence(400.0), RATE) is None


@pytest.mark.parametrize("rate", [8000, 16000])
def test_offsets_work_at_both_analysis_rates(rate):
    audio, true_end = burst_at(300.0, 500.0, 400.0, )
    if rate != RATE:
        from analyzer.resample import resample_int16

        audio = resample_int16(audio, RATE, rate)
        true_end = int(true_end * rate / RATE)
    floor = O.noise_floor_dbfs(audio, rate)
    refined = O._refine_offset(audio, rate, true_end + ms_to_samples(20.0, rate), floor)
    assert abs(samples_to_ms(refined - true_end, rate)) <= TOLERANCE_MS
