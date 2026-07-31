"""HTML report: renders, stays well-formed, and never hides the caveats."""

from __future__ import annotations

import json
import re

from report.html_report import render_html, write_html

REPORT = {
    "run_id": "bench-telnyx-test",
    "vendor": "telnyx",
    "attempts": 10,
    "usable": 8,
    "discards": {"idle_filler": 2},
    "ttfab_onset_ms": {"p50": 1263.4, "p90": 1681.6, "p95": 1767.0, "p99": 1835.3},
    "ttfab_onset_n": 8,
    "ttfab_content_ms": {"p50": 1263.4, "p95": 1767.0},
    "ttfab_content_n": 8,
    "ttfg_from_recording_start_ms": {"p50": 317.0, "p95": 3308.0},
    "negative_ttfab_count": 0,
    "consistency_p95_over_p50": 1.389,
    "beyond_2x_median_fraction": 0.0,
    "buffering_correlation_r": 0.373,
    "buffering_n": 8,
    "vendor_config_sha256": "08183f9a41822420" + "0" * 48,
    "vendor_defaults_used": {
        "model": "moonshotai/Kimi-K2.6",
        "stt_model": "deepgram/flux",
        "voice": "Telnyx.Ultra.f786b574",
        "user_idle_reply_secs": 10,
        "version_id": "20260728T225813827600",
        "tools": [],
        "endpointing": {"eot_threshold": 0.8, "eot_timeout_ms": 5000,
                        "start_speaking_wait_seconds": 0.1},
    },
    "vendor_unsupported": ["llm_temperature"],
    "not_yet_publishable": [
        "Instrument noise is ~100 ms class and Gate C is UNCERTIFIED.",
        "n=8 usable call(s). This is a sample, not a distribution.",
    ],
}

RESULTS = [
    {"call_id": "call-000", "ttfab_onset_ms": 1504.4, "ttfab_content_ms": 1504.4,
     "ttfg_ms": 152.0, "vendor_response_duration_ms": 2400.0, "psr": 3.92,
     "drift_ms": 0.0, "discard_reason": "idle_filler", "flags": [],
     "t1_ms": 16663.0, "t2_ms": 18167.0, "stimulus_start_ms": 14163.0,
     "greeting_onset_ms": 152.0, "greeting_end_ms": 1216.0},
    {"call_id": "call-001", "ttfab_onset_ms": 1608.4, "ttfab_content_ms": 1608.4,
     "ttfg_ms": 584.0, "vendor_response_duration_ms": 2200.0, "psr": 4.16,
     "drift_ms": 0.0, "discard_reason": None, "flags": [],
     "t1_ms": 5223.0, "t2_ms": 6831.0, "stimulus_start_ms": 2723.0,
     "greeting_onset_ms": 584.0, "greeting_end_ms": 2240.0},
    {"call_id": "call-003", "ttfab_onset_ms": 1190.4, "ttfab_content_ms": 1190.4,
     "ttfg_ms": None, "vendor_response_duration_ms": 2100.0, "psr": 4.16,
     "drift_ms": 0.0, "discard_reason": None,
     "flags": ["recording_started_mid_speech"],
     "t1_ms": 4883.0, "t2_ms": 6073.0, "stimulus_start_ms": 2383.0,
     "greeting_onset_ms": 96.0, "greeting_end_ms": 1760.0},
]


def _render() -> str:
    return render_html(REPORT, RESULTS)


def test_page_is_self_contained_and_offline():
    page = _render()
    assert page.startswith("<!doctype html>")
    # Inline CSS only -- must open from file:// with no network.
    assert "<style>" in page
    assert "http://" not in page and "https://" not in page
    assert "<script src" not in page


def test_shares_the_simulator_visual_vocabulary():
    """Same palette, same dashed-hairline idiom, same theme toggle."""
    page = _render()
    for token in ("--accent:#e8c397", "--ok:#84c08a", "--bad:#d97a7a",
                  "1px dashed var(--line2)", "bg-grid", "theme-toggle",
                  "tag-line", "tag-dot", "step-pct", "citation-table"):
        assert token in page, f"missing shared style token: {token}"


def test_theme_toggle_persists_choice():
    page = _render()
    assert "localStorage.setItem('ob-theme'" in page
    assert "localStorage.getItem('ob-theme')==='light'" in page
    assert "body.light{" in page


def test_markup_is_balanced():
    page = _render()
    for tag in ("section", "div", "table", "tbody", "thead"):
        opens = len(re.findall(rf"<{tag}[ >]", page))
        closes = len(re.findall(rf"</{tag}>", page))
        assert opens == closes, f"<{tag}>: {opens} open vs {closes} close"


def test_headline_numbers_are_present():
    page = _render()
    assert "1,263" in page          # p50, thousands-separated
    assert "8/10" in page           # usable / attempts
    assert "n=8" in page


def test_discards_are_shown_with_their_meaning():
    page = _render()
    assert "idle_filler" in page
    # Not just the code -- a reader must learn what it means.
    assert "would otherwise" in page
    assert "2 of 10 calls were refused" in page


def test_no_discards_renders_an_affirmative_panel():
    clean = dict(REPORT, discards={}, usable=10)
    page = render_html(clean, RESULTS)
    assert "All 10 calls produced a usable measurement" in page


def test_a_clipped_recording_never_fabricates_a_figure():
    """TTFG is no longer a tile on the page, but a report built from a run whose
    recording opened mid-greeting must still render rather than raise -- an empty
    aggregate is the normal case, not an error."""
    no_ttfg = dict(REPORT, ttfg_from_recording_start_ms={})
    page = render_html(no_ttfg, RESULTS)

    assert "TTFAB" in page
    # The tile that used to occupy this case is gone, so the page must not
    # reintroduce it -- and must not print a placeholder in its place.
    assert "withheld" not in page
    assert "opened mid-greeting" not in page


def test_unpinnable_axes_are_surfaced():
    page = _render()
    assert "llm_temperature" in page
    assert "cannot be pinned" in page


def test_buffering_verdict_flips_with_correlation():
    streaming = _render()
    assert "stream as tokens arrive" in streaming

    buffered = render_html(dict(REPORT, buffering_correlation_r=0.91), RESULTS)
    assert "wait for the whole answer" in buffered


def test_per_call_cells_mark_pass_and_fail():
    page = _render()
    assert 'class="cell pass"' in page
    assert 'class="cell fail"' in page
    assert "call-000" in page and "call-003" in page
    # Analyzer flags travel to the page.
    assert "recording_started_mid_speech" in page


def test_barge_in_is_reported_when_present():
    page = render_html(dict(REPORT, negative_ttfab_count=2), RESULTS)
    assert "barge-in" in page
    assert "spoke before we finished" in page


def test_distribution_strip_places_one_tick_per_call():
    page = _render()
    measured = [r for r in RESULTS
                if r["ttfab_onset_ms"] is not None and not r["discard_reason"]]
    dropped = [r for r in RESULTS
               if r["ttfab_onset_ms"] is not None and r["discard_reason"]]
    assert len(re.findall(r'class="tick"', page)) == len(measured)
    assert len(re.findall(r'class="tick dropped"', page)) == len(dropped)
    assert "dist-median" in page


def test_empty_run_still_renders():
    empty = {"run_id": "r", "vendor": "telnyx", "attempts": 0, "usable": 0,
             "discards": {}, "ttfab_onset_ms": {}, "ttfab_onset_n": 0,
             "ttfab_content_ms": {}, "ttfg_from_recording_start_ms": {}}
    page = render_html(empty, [])
    assert page.startswith("<!doctype html>")
    assert "telnyx" in page


def test_html_is_escaped():
    nasty = dict(REPORT, vendor='<script>alert(1)</script>')
    page = render_html(nasty, RESULTS)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_write_html_picks_up_the_applied_config(tmp_path):
    (tmp_path / "applied_config.json").write_text(json.dumps({
        "sha256": "abc", "defaults_used": {"model": "some/model"},
        "unsupported": ["llm_temperature"],
    }))
    out = write_html(dict(REPORT, vendor_defaults_used=None), RESULTS, tmp_path)
    assert out.name == "report.html"
    assert "some/model" in out.read_text()
