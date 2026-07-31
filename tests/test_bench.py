"""Bench reporting: the figures, and the honesty that ships with them."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness.bench import PUBLISHABLE_N, build_report, honesty_block


@dataclass
class FakeResult:
    """Just the fields build_report reads."""

    ttfab_onset_ms: float | None = None
    ttfab_content_ms: float | None = None
    ttfg_ms: float | None = None
    vendor_response_duration_ms: float | None = None
    discard_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.discard_reason is None


APPLIED = {
    "sha256": "deadbeef" * 8,
    "defaults_used": {"model": "moonshotai/Kimi-K2.6", "stt_model": "deepgram/flux"},
    "unsupported": ["llm_temperature"],
}


def _report(results, vendor="telnyx"):
    return build_report(results, vendor, "bench-test", APPLIED)


def test_percentiles_and_counts():
    results = [FakeResult(ttfab_onset_ms=float(v), ttfg_ms=500.0)
               for v in range(800, 900, 10)]
    report = _report(results)
    assert report["attempts"] == 10
    assert report["usable"] == 10
    assert report["ttfab_onset_n"] == 10
    assert report["ttfab_onset_ms"]["p50"] == pytest.approx(845.0, abs=1)
    assert report["ttfg_from_recording_start_ms"]["p50"] == pytest.approx(500.0)


def test_discards_are_counted_not_hidden():
    """A vendor that fails to answer is worse than one slightly slower, so the
    discard breakdown is a headline figure."""
    results = [
        FakeResult(ttfab_onset_ms=800.0),
        FakeResult(discard_reason="idle_filler"),
        FakeResult(discard_reason="no_response"),
        FakeResult(discard_reason="idle_filler"),
    ]
    report = _report(results)
    assert report["attempts"] == 4
    assert report["usable"] == 1
    assert report["discards"] == {"idle_filler": 2, "no_response": 1}


def test_negative_ttfab_is_reported_as_barge_in():
    results = [FakeResult(ttfab_onset_ms=-250.0), FakeResult(ttfab_onset_ms=900.0)]
    assert _report(results)["negative_ttfab_count"] == 1


def test_consistency_measures_the_tail():
    """Consistency is reported separately from the median, because buyers get
    burned by the tail."""
    results = [FakeResult(ttfab_onset_ms=800.0) for _ in range(9)]
    results.append(FakeResult(ttfab_onset_ms=4000.0))
    report = _report(results)
    assert report["consistency_p95_over_p50"] > 1.5
    assert report["beyond_2x_median_fraction"] == pytest.approx(0.1)


def test_buffering_check_detects_full_reply_tts():
    """A vendor whose TTS waits for the whole answer shows TTFAB
    rising with reply length."""
    results = [
        FakeResult(ttfab_onset_ms=500.0 + d, vendor_response_duration_ms=float(d))
        for d in (500, 1000, 1500, 2000, 2500, 3000)
    ]
    report = _report(results)
    assert report["buffering_correlation_r"] > 0.9
    assert report["buffering_n"] == 6


def test_buffering_check_absent_for_streaming_tts():
    results = [
        FakeResult(ttfab_onset_ms=800.0, vendor_response_duration_ms=float(d))
        for d in (500, 1000, 1500, 2000, 2500, 3000)
    ]
    report = _report(results)
    # Constant onsets -> zero variance -> no correlation claimed at all.
    assert "buffering_correlation_r" not in report


def test_buffering_check_needs_enough_pairs():
    results = [FakeResult(ttfab_onset_ms=800.0 + d, vendor_response_duration_ms=float(d))
               for d in (500, 1000)]
    assert "buffering_correlation_r" not in _report(results)


def test_receipt_travels_with_the_report():
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    assert report["vendor_config_sha256"] == APPLIED["sha256"]
    assert report["vendor_unsupported"] == ["llm_temperature"]
    assert report["vendor_defaults_used"]["model"] == "moonshotai/Kimi-K2.6"


def test_report_survives_all_calls_discarded():
    results = [FakeResult(discard_reason="no_response") for _ in range(3)]
    report = _report(results)
    assert report["usable"] == 0
    assert report["ttfab_onset_ms"] == {}
    assert "consistency_p95_over_p50" not in report


# ------------------------------------------------------------------ honesty ---


CALLER_RECEIPT = {"voice": "Polly.Joanna", "sha256": "abc123"}


def test_honesty_block_admits_the_instrument_is_uncharacterised():
    """No characterisation exists for this measurement path. Quoting an overhead
    figure anyway would put a reassuring, wrong number on a public page -- so the
    block states the absence instead, and still refuses to subtract.
    """
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    joined = " ".join(honesty_block(report, CALLER_RECEIPT))

    assert "UNCHARACTERISED" in joined
    assert "none is subtracted" in joined
    # No overhead or noise figure may appear, because none was measured.
    assert "sweep-" not in joined
    assert "126 ms" not in joined
    assert "8.8 ms" not in joined


def test_honesty_block_states_how_t1_is_now_found():
    """A VAD-derived speech offset is a different measurement from a matched
    filter, and the difference is the reader's to judge."""
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    joined = " ".join(honesty_block(report, CALLER_RECEIPT))
    assert "speech detector" in joined
    assert "sample-exact" in joined


def test_honesty_block_separates_endpointing_from_measurement():
    """Plivo's endpointing drives the conversation; both TTFAB endpoints still
    come off the tape. Confusing the two is the error the prototype made."""
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    joined = " ".join(honesty_block(report, CALLER_RECEIPT))
    assert "endpointing" in joined
    assert "discarded turns" in joined


def test_honesty_block_names_the_synthetic_voice():
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    joined = " ".join(honesty_block(report, CALLER_RECEIPT))
    assert "Polly.Joanna" in joined
    assert "human recording" in joined


def test_honesty_block_warns_when_the_channel_map_is_only_assumed():
    """Once the probe pins the channel order the warning must disappear, or it
    becomes noise readers learn to skip."""
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    assert "channel_map_suspect" not in " ".join(honesty_block(report, CALLER_RECEIPT))
    report["channel_map_provisional"] = True
    assert "channel_map_suspect" in " ".join(honesty_block(report, CALLER_RECEIPT))


def test_honesty_block_flags_small_samples():
    report = _report([FakeResult(ttfab_onset_ms=800.0)])
    joined = " ".join(honesty_block(report, CALLER_RECEIPT))
    assert f"n>={PUBLISHABLE_N}" in joined
    assert "sample" in joined


def test_honesty_block_drops_sample_warning_at_scale():
    results = [FakeResult(ttfab_onset_ms=800.0) for _ in range(PUBLISHABLE_N)]
    joined = " ".join(honesty_block(_report(results), CALLER_RECEIPT))
    assert "not a distribution" not in joined


def test_the_instrument_block_reports_absence_rather_than_a_stale_number():
    """A null instrument renders as "unknown"; a borrowed one renders as fact."""
    from harness.bench import MODE, instrument_block

    block = instrument_block(MODE)
    assert block["valid"] is False
    assert block["path_overhead_subtracted"] is False
    assert "sweep_run" not in block and "path_overhead_ms" not in block


def test_print_report_runs_clean(capsys):
    results = [FakeResult(ttfab_onset_ms=800.0 + i * 10, ttfab_content_ms=900.0,
                          ttfg_ms=500.0, vendor_response_duration_ms=1000.0 + i)
               for i in range(8)]
    results.append(FakeResult(discard_reason="idle_filler"))
    report = _report(results)
    from harness.bench import print_report

    print_report(report, honesty_block(report, CALLER_RECEIPT))
    out = capsys.readouterr().out
    assert "BENCH: telnyx" in out
    assert "TTFAB-onset" in out
    assert "idle_filler" in out
    assert "NOT YET PUBLISHABLE" in out
    # TTFG must never be presented as answer-to-greeting.
    assert "not carrier answer" in out
