"""t1 location must be exact, and its failure modes must be detectable.

An undetected bias here shifts every reported number by the same amount, and no
amount of self-consistency checking would reveal it -- which is the whole reason
Gate A compares against constructed truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.correlate import (
    DEFAULT_BETA,
    MIN_PSR,
    locate_t1,
    tail_template,
)
from tests.conftest import load_fixture

# t1 must be exact on fixtures. The spec's tolerance is 5 ms; correlation
# measures sample-exact, so holding it to a sample is the stronger assertion and
# will catch a regression the loose bound would let through.
TOLERANCE_MS = 0.2


def _all_names(truths):
    return sorted(truths)


def test_t1_is_exact_on_every_fixture(fixture_dir, truths):
    errors = {}
    for name, truth in truths.items():
        near, _, ref, rate = load_fixture(fixture_dir, truth)
        est = locate_t1(near, ref, rate)
        errors[name] = (est.t1 - truth["t1"]) * 1000.0 / rate

    bad = {k: v for k, v in errors.items() if abs(v) > TOLERANCE_MS}
    assert not bad, f"t1 error beyond {TOLERANCE_MS} ms: {bad}"


def test_companding_does_not_move_t1(fixture_dir, truths):
    # The 8 kHz mu-law fixture and the 16 kHz linear fixture carry the same
    # stimulus; both must land exactly, or the reference-prep step is wrong.
    for name in ("clean_500ms", "uncompanded_16k"):
        truth = truths[name]
        near, _, ref, rate = load_fixture(fixture_dir, truth)
        est = locate_t1(near, ref, rate)
        assert est.t1 == truth["t1"], name


def test_drift_check_catches_excised_audio(fixture_dir, truths):
    """40 ms removed mid-stimulus must surface as ~40 ms of drift.

    This is the case that justifies matching the tail rather than the whole clip:
    the tail estimate stays correct while the whole-reference estimate does not,
    and their disagreement is the signal.
    """
    truth = truths["mid_stimulus_gap"]
    near, _, ref, rate = load_fixture(fixture_dir, truth)
    est = locate_t1(near, ref, rate)

    assert est.t1 == truth["t1"], "tail estimate should be unaffected by a mid-stream gap"
    assert abs(abs(est.drift_ms) - 40.0) < 2.0, f"drift was {est.drift_ms} ms"


def test_clean_fixtures_report_no_drift(fixture_dir, truths):
    for name in ("clean_500ms", "clean_300ms", "clean_1200ms", "comfort_noise"):
        truth = truths[name]
        near, _, ref, rate = load_fixture(fixture_dir, truth)
        est = locate_t1(near, ref, rate)
        assert abs(est.drift_ms) < 2.0, f"{name} drift {est.drift_ms} ms"


def test_confidence_clears_threshold_on_all_fixtures(fixture_dir, truths):
    for name, truth in truths.items():
        near, _, ref, rate = load_fixture(fixture_dir, truth)
        est = locate_t1(near, ref, rate)
        assert est.confident, f"{name} PSR {est.psr} below {MIN_PSR}"


def test_barge_in_has_the_thinnest_confidence_margin(fixture_dir, truths):
    """Documents a known weak spot rather than pretending it isn't there.

    In the barge-in case the vendor's reply overlaps the template window, so
    uncorrelated energy lands exactly where we match. It still clears threshold on
    a fixture, but with far less headroom than the clean cases -- which is why
    MIN_PSR needs re-validating on real recordings in Phase C.
    """
    psrs = {}
    for name, truth in truths.items():
        near, _, ref, rate = load_fixture(fixture_dir, truth)
        psrs[name] = locate_t1(near, ref, rate).psr

    assert psrs["barge_in"] == min(psrs.values())
    assert psrs["barge_in"] < psrs["clean_500ms"]


def test_unlocatable_stimulus_is_reported_not_guessed(fixture_dir, truths):
    # Correlating against audio that does not contain the reference at all must
    # come back unconfident, so the call is discarded rather than assigned a t1
    # picked out of noise.
    truth = truths["clean_500ms"]
    _, _, ref, rate = load_fixture(fixture_dir, truth)

    rng = np.random.default_rng(7)
    noise = rng.integers(-2000, 2000, size=rate * 6, dtype=np.int16)

    est = locate_t1(noise, ref, rate)
    assert not est.confident, f"noise produced a confident match, PSR {est.psr}"


def test_tail_template_ends_at_the_reference_end(fixture_dir, truths):
    truth = truths["clean_500ms"]
    _, _, ref, rate = load_fixture(fixture_dir, truth)
    tpl = tail_template(ref, rate, window_ms=500.0)

    assert len(tpl) == int(round(0.5 * rate))
    # Must be the tail, not the head: t1 is derived as peak + len(template), which
    # is only the end of speech if the template is the final window.
    np.testing.assert_array_equal(tpl, ref[-len(tpl) :])


def test_tail_template_degrades_to_whole_reference_when_too_short():
    short = np.arange(100, dtype=np.int16)
    assert len(tail_template(short, 8000, window_ms=500.0)) == 100


@pytest.mark.parametrize("beta", [0.5, 0.75, 1.0])
def test_t1_is_exact_for_any_whitening_above_zero(fixture_dir, truths, beta):
    # Guards against the constant being retuned into a broken state.
    truth = truths["clean_500ms"]
    near, _, ref, rate = load_fixture(fixture_dir, truth)
    assert locate_t1(near, ref, rate, beta=beta).t1 == truth["t1"]


def test_plain_correlation_cannot_see_drift(fixture_dir, truths):
    """Pins the reason DEFAULT_BETA is not 0.

    Without whitening the drift check silently reports zero on a stimulus that
    lost 40 ms. If someone sets beta to 0 for speed, this fails and explains why.
    """
    truth = truths["mid_stimulus_gap"]
    near, _, ref, rate = load_fixture(fixture_dir, truth)

    whitened = locate_t1(near, ref, rate, beta=DEFAULT_BETA)
    plain = locate_t1(near, ref, rate, beta=0.0)

    assert abs(abs(whitened.drift_ms) - 40.0) < 2.0
    assert abs(plain.drift_ms) < 2.0, "unexpected: plain correlation detected the gap"
