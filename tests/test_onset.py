"""Onset detection accuracy, and its two known failure modes.

The comfort-noise case and the Silero-context case are the two tests worth having
here. Both are silent failures: they produce plausible numbers rather than
errors, so nothing but a comparison against constructed truth would catch them.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer import onset as O
from analyzer.codec import int16_to_float
from tests.conftest import load_fixture

TOLERANCE_MS = 5.0


def _far(fixture_dir, truth):
    _, far, _, rate = load_fixture(fixture_dir, truth)
    return far, rate


def test_t2_is_within_tolerance_on_every_fixture(fixture_dir, truths):
    errors = {}
    for name, truth in truths.items():
        far, rate = _far(fixture_dir, truth)
        analysis = O.analyze(far, rate)
        seg = analysis.first_after(truth["stimulus_start"])
        assert seg is not None, f"{name}: no speech found after the stimulus"
        errors[name] = (seg.start - truth["t2"]) * 1000.0 / rate

    bad = {k: round(v, 2) for k, v in errors.items() if abs(v) > TOLERANCE_MS}
    assert not bad, f"t2 error beyond {TOLERANCE_MS} ms: {bad}"


def test_greeting_onset_is_within_tolerance(fixture_dir, truths):
    """Looser than t2 on purpose.

    Greeting onset is an internal quantity, not a reported metric:
    it sets the per-vendor playback offset and bounds the idle-filler discard
    window. It measures +2 ms on clean fixtures but +8 ms under -45 dBFS comfort
    noise, because a soft "Hi" genuinely cannot be localised to 5 ms once the
    threshold has to clear real background noise. 15 ms is far tighter than
    anything that depends on it needs.
    """
    for name, truth in truths.items():
        if truth["greeting_onset"] is None:
            continue
        far, rate = _far(fixture_dir, truth)
        analysis = O.analyze(far, rate)
        assert analysis.segments, name
        err = (analysis.segments[0].start - truth["greeting_onset"]) * 1000.0 / rate
        assert abs(err) <= 15.0, f"{name}: greeting onset off by {err:.2f} ms"


def test_comfort_noise_does_not_make_the_vendor_look_faster(fixture_dir, truths):
    """The failure mode that most flatters a vendor.

    A platform that streams continuous low-level noise instead of true silence
    would trip an energy gate on the first frame and appear ~200 ms quicker. Here
    the noise is at -45 dBFS throughout, and t2 must still land on the real onset.
    """
    truth = truths["comfort_noise"]
    far, rate = _far(fixture_dir, truth)

    analysis = O.analyze(far, rate)
    assert analysis.noise_floor_dbfs > -60.0, "fixture should have a measurable floor"

    seg = analysis.first_after(truth["stimulus_start"])
    err = (seg.start - truth["t2"]) * 1000.0 / rate
    assert abs(err) <= TOLERANCE_MS, f"comfort noise shifted t2 by {err:.2f} ms"


def test_silero_needs_its_context_prefix(fixture_dir, truths):
    """Regression test for a silent, catastrophic misuse of the model.

    Silero v5 requires window/8 samples of context prepended to each chunk. The
    onnx graph accepts any input length, so omitting it does not raise -- it just
    returns near-zero probabilities at 16 kHz and badly degraded ones at 8 kHz.
    This pins the contract so nobody "simplifies" the context away.
    """
    far, rate = _far(fixture_dir, truths["clean_500ms"])

    with_ctx, window = O._Silero.probabilities(far, rate)
    assert with_ctx.max() > 0.9, "context path should confidently find speech"

    # Same audio, same window, no context -- deliberately reproducing the bug.
    import onnxruntime as ort

    sess = O._Silero.session()
    x = int16_to_float(far).astype(np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(rate, dtype=np.int64)
    without = []
    for i in range(min(120, len(x) // window)):
        chunk = x[i * window : (i + 1) * window].reshape(1, window)
        out, state = sess.run(None, {"input": chunk, "state": state, "sr": sr})
        without.append(out[0, 0])

    speech_frac_ctx = float((with_ctx[: len(without)] >= O.SPEECH_THRESHOLD).mean())
    speech_frac_no_ctx = float((np.array(without) >= O.SPEECH_THRESHOLD).mean())
    assert speech_frac_ctx > speech_frac_no_ctx, (
        "context must materially improve detection; if these are now equal the "
        "model contract changed and SILERO_CONTEXT should be revisited"
    )


def test_digital_silence_lead_in_does_not_break_refinement(fixture_dir, truths):
    """A true-silence head measures -inf dBFS.

    Unclamped, `floor + margin` is then -inf, every frame clears it, and the
    refinement collapses to the start of its own search window -- putting every
    onset one search radius early. FLOOR_FLOOR_DBFS is what prevents that.
    """
    truth = truths["clean_500ms"]
    far, rate = _far(fixture_dir, truth)

    analysis = O.analyze(far, rate)
    # Digital silence measures far below anything real (the percentile estimator
    # reports the frame-energy floor rather than -inf). Either way it is below
    # FLOOR_FLOOR_DBFS, so the clamp is what keeps refinement sane.
    assert analysis.noise_floor_dbfs < O.FLOOR_FLOOR_DBFS, (
        "fixture head should be true silence"
    )

    seg = analysis.first_after(truth["stimulus_start"])
    err = (seg.start - truth["t2"]) * 1000.0 / rate
    radius = O.REFINE_RADIUS_MS
    assert abs(err) <= TOLERANCE_MS, (
        f"t2 off by {err:.2f} ms; a value near -{radius} ms means the floor clamp "
        "is not being applied"
    )


def test_refinement_beats_the_coarse_estimate(fixture_dir, truths):
    """Stage two must actually earn its place."""
    coarse_err, fine_err = [], []
    for truth in truths.values():
        far, rate = _far(fixture_dir, truth)
        analysis = O.analyze(far, rate)
        seg = analysis.first_after(truth["stimulus_start"])
        idx = analysis.segments.index(seg)
        coarse_err.append(abs(analysis.coarse_starts[idx] - truth["t2"]) * 1000.0 / rate)
        fine_err.append(abs(seg.start - truth["t2"]) * 1000.0 / rate)

    assert np.mean(fine_err) < np.mean(coarse_err)
    assert max(fine_err) <= TOLERANCE_MS


def test_filler_is_reported_separately_from_content(fixture_dir, truths):
    """TTFAB-onset counts the 'mm hmm'. TTFAB-content skips to the real answer."""
    truth = truths["filler_first"]
    far, rate = _far(fixture_dir, truth)
    analysis = O.analyze(far, rate)

    onset = analysis.first_after(truth["stimulus_start"])
    content = analysis.content_after(truth["t1"], min_duration_ms=400.0)

    assert onset is not None and content is not None
    assert content.start > onset.start, "content should land after the filler"

    err = (content.start - truth["response_start"]) * 1000.0 / rate
    assert abs(err) <= 40.0, f"content onset off by {err:.1f} ms"


def test_idle_filler_is_visible_in_the_discard_window(fixture_dir, truths):
    """The 'are you still there?' prompt must be findable between greeting and stimulus.

    Without this the prompt gets recorded as the vendor's response and produces a
    fast, wrong TTFAB.
    """
    truth = truths["idle_filler"]
    far, rate = _far(fixture_dir, truth)
    analysis = O.analyze(far, rate)

    intruder = analysis.first_between(truth["greeting_end"], truth["stimulus_start"])
    assert intruder is not None, "idle filler was not detected"

    clean = truths["clean_500ms"]
    far_c, rate_c = _far(fixture_dir, clean)
    analysis_c = O.analyze(far_c, rate_c)
    assert (
        analysis_c.first_between(clean["greeting_end"], clean["stimulus_start"]) is None
    ), "clean fixture must not trip the idle-filler rule"


def test_crosscheck_agrees_with_stage_one(fixture_dir, truths):
    for name, truth in truths.items():
        far, rate = _far(fixture_dir, truth)
        analysis = O.analyze(far, rate)
        seg = analysis.first_after(truth["stimulus_start"])
        disagreement = O.crosscheck_disagreement_ms(seg.start, far, rate)
        assert disagreement is not None, name
        assert disagreement <= O.DISAGREEMENT_FLAG_MS, f"{name}: {disagreement:.1f} ms"


def test_analysis_is_deterministic(fixture_dir, truths):
    """Exit criterion 4: re-running must reproduce identical numbers."""
    far, rate = _far(fixture_dir, truths["clean_500ms"])
    a = O.analyze(far, rate)
    b = O.analyze(far, rate)
    assert [(s.start, s.end) for s in a.segments] == [(s.start, s.end) for s in b.segments]


def test_negative_ttfab_survives_analysis(fixture_dir, truths):
    """Barge-in is a result, not an error."""
    truth = truths["barge_in"]
    far, rate = _far(fixture_dir, truth)
    analysis = O.analyze(far, rate)

    seg = analysis.first_after(truth["greeting_end"])
    assert seg is not None
    assert seg.start < truth["t1"], "vendor should start before we finish"


@pytest.mark.parametrize("rate", [8000, 16000])
def test_supported_rates_have_both_window_and_context(rate):
    assert rate in O.SILERO_WINDOWS
    assert rate in O.SILERO_CONTEXT
    assert O.SILERO_CONTEXT[rate] == O.SILERO_WINDOWS[rate] // 8


def test_unsupported_rate_is_rejected_loudly():
    with pytest.raises(ValueError, match="Silero supports"):
        O._Silero.probabilities(np.zeros(4000, dtype=np.int16), 44100)


def test_empty_input_is_handled():
    analysis = O.analyze(np.zeros(0, dtype=np.int16), 8000)
    assert analysis.segments == []
