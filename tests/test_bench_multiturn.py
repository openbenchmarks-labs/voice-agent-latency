"""Per-turn aggregation: pooling over turns, per-index curve, turn-level discards."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness.bench import build_report, honesty_block, print_report
from report.html_report import render_html

APPLIED = {"sha256": "a" * 64, "defaults_used": {"model": "m"}, "unsupported": []}


@dataclass
class FakeTurn:
    index: int = 1
    ttfab_onset_ms: float | None = None
    ttfab_content_ms: float | None = None
    vendor_response_duration_ms: float | None = None
    psr: float | None = 4.0
    drift_ms: float | None = 0.0
    vad_disagreement_ms: float | None = 6.0
    flags: list = field(default_factory=list)
    discard_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.discard_reason is None


@dataclass
class FakeCall:
    turns: list = field(default_factory=list)
    ttfg_ms: float | None = 500.0
    discard_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.discard_reason is None

    # build_report reads these only for single-turn compatibility
    @property
    def ttfab_onset_ms(self):
        return self.turns[0].ttfab_onset_ms if self.turns else None


def _call(*ttfabs, discard_at: dict | None = None, call_discard=None) -> FakeCall:
    discard_at = discard_at or {}
    turns = [
        FakeTurn(index=i, ttfab_onset_ms=v, ttfab_content_ms=v,
                 vendor_response_duration_ms=2000.0 + 10 * i,
                 discard_reason=discard_at.get(i))
        for i, v in enumerate(ttfabs, start=1)
    ]
    return FakeCall(turns=turns, discard_reason=call_discard)


def _report(calls):
    return build_report(calls, "telnyx", "bench-mt", APPLIED)


def test_latency_is_pooled_over_turns_not_calls():
    """A 4-turn call contributes four measurements, not one."""
    calls = [_call(1000, 1100, 1200, 1300) for _ in range(3)]
    report = _report(calls)

    assert report["attempts"] == 3            # calls
    assert report["turn_attempts"] == 12      # turns
    assert report["turns_usable"] == 12
    assert report["ttfab_onset_n"] == 12
    assert report["turns_per_call"] == 4


def test_per_turn_curve_is_indexed_correctly():
    calls = [_call(1000, 1200, 1400, 1600) for _ in range(5)]
    per_turn = _report(calls)["per_turn"]

    assert sorted(per_turn, key=int) == ["1", "2", "3", "4"]
    assert per_turn["1"]["p50"] == pytest.approx(1000)
    assert per_turn["4"]["p50"] == pytest.approx(1600)
    assert all(v["n"] == 5 for v in per_turn.values())


def test_one_bad_turn_does_not_waste_the_others():
    """The reason discards live on the turn."""
    calls = [_call(1000, 1100, 1200, 1300, discard_at={3: "barged_reply"})
             for _ in range(4)]
    report = _report(calls)

    assert report["turn_attempts"] == 16
    assert report["turns_usable"] == 12          # turn 3 lost in every call
    assert report["discards"] == {"barged_reply": 4}
    assert report["per_turn"]["3"]["usable"] == 0
    assert report["per_turn"]["2"]["usable"] == 4
    # And the pooled figure only sees good turns.
    assert report["ttfab_onset_n"] == 12


def test_call_level_discard_removes_all_its_turns():
    """A poisoned conversation invalidates every turn in it, unlike a turn fault."""
    calls = [_call(1000, 1100, 1200, 1300),
             _call(900, 950, 1000, 1050, call_discard="idle_filler")]
    report = _report(calls)

    assert report["usable"] == 1
    assert report["turn_attempts"] == 4     # only the good call's turns counted
    assert report["turns_usable"] == 4
    assert report["discards"] == {"idle_filler": 1}


def test_single_turn_reports_have_no_curve():
    calls = [_call(1000) for _ in range(6)]
    report = _report(calls)
    assert report["turns_per_call"] == 1
    assert list(report["per_turn"]) == ["1"]
    assert report["ttfab_onset_n"] == 6


def test_ttfg_stays_call_scoped():
    """The greeting happens once per call, so it must not be counted per turn."""
    calls = [_call(1000, 1100, 1200, 1300) for _ in range(3)]
    report = _report(calls)
    # Three calls -> three TTFG values, even though there are twelve turns.
    assert report["ttfg_from_recording_start_ms"]["p50"] == pytest.approx(500.0)


def test_buffering_check_uses_turns():
    calls = [_call(1000 + 100 * i, 1100 + 100 * i) for i in range(4)]
    report = _report(calls)
    assert report.get("buffering_n", 0) >= 4


def test_honesty_block_warns_about_thin_per_index_samples():
    calls = [_call(1000, 1100, 1200, 1300) for _ in range(2)]
    report = _report(calls)
    joined = " ".join(honesty_block(report, {"voice": "Polly.Joanna"}))
    assert "turn curve" in joined
    assert "n=2" in joined
    assert "directional" in joined


def test_printed_report_includes_the_curve(capsys):
    calls = [_call(1000, 1200, 1400, 1600) for _ in range(4)]
    report = _report(calls)
    print_report(report, honesty_block(report, {"voice": "Polly.Joanna"}))
    out = capsys.readouterr().out

    assert "TURN CURVE" in out
    assert "turn 1" in out and "turn 4" in out
    assert "16/16 usable" in out or "turns" in out


def test_html_renders_the_turn_curve():
    calls = [_call(1000, 1200, 1400, 1600) for _ in range(4)]
    report = _report(calls)
    results = [{
        "call_id": "call-000", "ttfab_onset_ms": 1000.0, "psr": 4.0,
        "drift_ms": 0.0, "discard_reason": None, "flags": [],
        "t1_ms": 3000.0, "t2_ms": 4000.0, "stimulus_start_ms": 1000.0,
        "vendor_response_duration_ms": 2000.0,
        "turns": [
            {"index": i, "ttfab_onset_ms": 1000.0 + 200 * (i - 1),
             "vendor_response_duration_ms": 2000.0, "psr": 4.0, "drift_ms": 0.0,
             "discard_reason": None, "flags": []}
            for i in (1, 2, 3, 4)
        ],
    }]
    page = render_html(report, results)

    assert 'id="turns"' in page
    assert "turn curve" in page
    assert "does context growth cost latency?" in page
    assert "turn-curve" in page          # the bar chart
    assert "rises" in page               # 1000 -> 1600 is a rise
    # Per-call turn breakdown table.
    assert "TTFAB ms" in page
    assert page.count("<td>turn ") >= 4


def test_html_omits_the_curve_for_single_turn():
    calls = [_call(1000) for _ in range(5)]
    page = render_html(_report(calls), [])
    assert 'id="turns"' not in page


def test_flat_curve_is_described_as_flat():
    calls = [_call(1000, 1000, 1000, 1000) for _ in range(4)]
    page = render_html(_report(calls), [])
    assert "is flat" in page


def test_falling_curve_is_described_as_falling():
    calls = [_call(1600, 1400, 1200, 1000) for _ in range(4)]
    page = render_html(_report(calls), [])
    assert "falls" in page
