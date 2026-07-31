"""Gate A, the discard rules, and reproducibility.

Gate A lives here as well as in the CLI on purpose: it is the only check that
compares the analyzer against a truth we constructed, so it should fail a normal
`pytest` run and not just a manual invocation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from analyzer import onset as O
from analyzer.measure import (
    ANALYZER_VERSION,
    GREETING_MERGE_GAP_MS,
    analyze_run,
    measure_call,
    measure_recording,
)
from tests.conftest import load_fixture

GATE_A_TOLERANCE_MS = 5.0


def _measure(fixture_dir, truth):
    near, far, ref, rate = load_fixture(fixture_dir, truth)
    return measure_recording(near, far, ref, rate), rate


def test_gate_a(fixture_dir, truths):
    """Every fixture either measures within tolerance or discards for the right reason."""
    failures = []
    worst = 0.0

    for name, truth in sorted(truths.items()):
        result, _ = _measure(fixture_dir, truth)
        wanted = truth["expect_discard"]

        if result.discard_reason is not None:
            if result.discard_reason != wanted:
                failures.append(
                    f"{name}: discarded as {result.discard_reason}, "
                    f"expected {wanted or 'a measurement'}"
                )
            continue

        if wanted is not None:
            failures.append(f"{name}: measured but should have discarded as {wanted}")
            continue

        assert result.ttfab_onset_ms is not None, name
        error = result.ttfab_onset_ms - truth["ttfab_ms"]
        worst = max(worst, abs(error))
        if abs(error) > GATE_A_TOLERANCE_MS:
            failures.append(f"{name}: error {error:+.2f} ms")

    assert not failures, "Gate A failed:\n  " + "\n  ".join(failures)
    assert worst <= GATE_A_TOLERANCE_MS


def test_clipped_recording_is_flagged_and_ttfg_withheld(fixture_dir, truths):
    """A tape that opens mid-greeting cannot report TTFG.

    Recording starts when our record_start takes effect, which is after the
    vendor began speaking -- 5/10 calls in the first Telnyx bench run. The
    resulting `greeting_onset` is wherever the clip began, which manufactured an
    8 s TTFG p95 out of nothing. TTFAB is unaffected (its endpoints are seconds
    later), so the call stays usable; only TTFG is withheld.
    """
    truth = truths["clean_500ms"]
    near, far, ref, rate = load_fixture(fixture_dir, truth)

    # Chop off the leading silence so the far channel opens in mid-speech. Cut at
    # a point verified loud rather than a guessed offset -- greetings contain
    # ~200 ms intra-sentence pauses, and landing in one produces a quiet head that
    # (correctly) does not look clipped.
    from analyzer.measure import CLIPPED_START_RMS, CLIPPED_START_WINDOW_MS

    window = int(CLIPPED_START_WINDOW_MS * rate / 1000.0)
    onset = None
    for candidate in range(truth["greeting_onset"], truth["greeting_end"] - window, 80):
        head = far[candidate:candidate + window].astype(np.float64)
        if np.sqrt(np.mean(head * head)) > CLIPPED_START_RMS * 3:
            onset = candidate
            break
    assert onset is not None, "fixture greeting has no loud stretch to cut into"

    clipped_far = np.ascontiguousarray(far[onset:])
    clipped_near = np.ascontiguousarray(near[onset:])

    result = measure_recording(clipped_near, clipped_far, ref, rate)
    assert "recording_started_mid_speech" in result.flags
    assert result.ttfg_ms is None, "TTFG must not be reported from a clipped greeting"
    # The measurement itself survives.
    assert result.discard_reason is None
    assert result.ttfab_onset_ms == pytest.approx(truth["ttfab_ms"], abs=GATE_A_TOLERANCE_MS)


def test_clean_recording_still_reports_ttfg(fixture_dir, truths):
    result, _ = _measure(fixture_dir, truths["clean_500ms"])
    assert "recording_started_mid_speech" not in result.flags
    assert result.ttfg_ms is not None


def test_a_discarded_call_keeps_its_measured_value(fixture_dir, truths):
    """Diagnostics need to show what the wrong answer would have been.

    Also guards the gate itself: branching on `ttfab is None` instead of on
    `discard_reason` would score a correct discard as a passing measurement, which
    is how the idle-filler rule initially appeared to work when it did not.
    """
    result, _ = _measure(fixture_dir, truths["idle_filler"])
    assert result.discard_reason == "idle_filler"
    assert result.ttfab_onset_ms is not None
    assert not result.usable


def test_two_part_greeting_is_not_mistaken_for_an_idle_prompt(fixture_dir, truths):
    """Regression test.

    Greetings arrive as several segments -- intra-sentence pauses run ~180 ms,
    above MIN_SILENCE_MS, and background noise lengthens them further. Treating
    `segments[0]` as the whole greeting put its own later fragments inside the
    idle-prompt window, so any vendor with a two-part greeting was discarded.
    """
    for name in ("comfort_noise", "uncompanded_16k"):
        truth = truths[name]
        near, far, ref, rate = load_fixture(fixture_dir, truth)

        analysis = O.analyze(far, rate)
        before = analysis.utterance_groups(
            GREETING_MERGE_GAP_MS, before=truth["stimulus_start"]
        )
        raw = [s for s in analysis.segments if s.start < truth["stimulus_start"]]
        assert len(raw) > 1, f"{name} should have a fragmented greeting to be a real test"
        assert len(before) == 1, f"{name} greeting should group into one utterance"

        result = measure_recording(near, far, ref, rate)
        assert result.discard_reason is None, f"{name} falsely discarded"


def test_idle_prompt_is_still_caught_after_grouping(fixture_dir, truths):
    """Grouping must not absorb a genuine idle prompt into the greeting."""
    truth = truths["idle_filler"]
    near, far, ref, rate = load_fixture(fixture_dir, truth)

    analysis = O.analyze(far, rate)
    groups = analysis.utterance_groups(
        GREETING_MERGE_GAP_MS, before=truth["stimulus_start"]
    )
    assert len(groups) == 2, "greeting and idle prompt should stay distinct"
    assert measure_recording(near, far, ref, rate).discard_reason == "idle_filler"


def test_unlocatable_stimulus_is_discarded(fixture_dir, truths):
    truth = truths["clean_500ms"]
    _, far, ref, rate = load_fixture(fixture_dir, truth)
    rng = np.random.default_rng(3)
    near = rng.integers(-2000, 2000, size=len(far), dtype=np.int16)

    result = measure_recording(near, far, ref, rate)
    assert result.discard_reason == "unlocatable"
    assert any("psr=" in f for f in result.flags)


def test_missing_audio_is_discarded_not_crashed(tmp_path):
    call_dir = tmp_path / "call-1"
    call_dir.mkdir()
    (call_dir / "metadata.json").write_text(json.dumps({"call_id": "call-1"}))

    result = measure_call(call_dir)
    assert result.discard_reason == "audio_missing"
    assert result.call_id == "call-1"
    assert result.analyzer_version == ANALYZER_VERSION


def test_mono_recording_is_rejected(tmp_path, fixture_dir, truths):
    truth = truths["clean_500ms"]
    near, far, ref, rate = load_fixture(fixture_dir, truth)

    call_dir = tmp_path / "call-mono"
    call_dir.mkdir()
    (call_dir / "metadata.json").write_text(json.dumps({"call_id": "call-mono"}))
    sf.write(call_dir / "recording.wav", far, rate, subtype="PCM_16")
    sf.write(call_dir / "our_audio.wav", ref, rate, subtype="PCM_16")

    result = measure_call(call_dir)
    assert result.discard_reason == "audio_missing"
    assert "recording_not_stereo" in result.flags


def _make_run(tmp_path: Path, fixture_dir: Path, truths: dict) -> Path:
    """Lay fixtures out as a run directory."""
    run = tmp_path / "run-test"
    run.mkdir()
    for i, (name, truth) in enumerate(sorted(truths.items())):
        near, far, ref, rate = load_fixture(fixture_dir, truth)
        call = run / f"call-{i:03d}-{name}"
        call.mkdir()
        (call / "metadata.json").write_text(json.dumps({
            "call_id": call.name,
            "run_id": "run-test",
            "vendor": "farend",
            "kind": "validation",
            "stimulus_id": "hours_q",
            "channel_map": {"near": 0, "far": 1},
        }))
        sf.write(call / "recording.wav", np.stack([near, far], axis=1), rate,
                 subtype="PCM_16")
        sf.write(call / "our_audio.wav", ref, rate, subtype="PCM_16")
    return run


def test_analyze_run_reads_every_call(tmp_path, fixture_dir, truths):
    run = _make_run(tmp_path, fixture_dir, truths)
    results = analyze_run(run)

    assert len(results) == len(truths)
    assert all(r.analyzer_version == ANALYZER_VERSION for r in results)
    assert all((run / r.call_id / "result.json").exists() for r in results)


def test_re_analysis_is_byte_identical(tmp_path, fixture_dir, truths):
    """Exit criterion 4.

    Re-running over saved audio must reproduce the same files exactly. Anything
    order-dependent or thread-dependent shows up here.
    """
    run = _make_run(tmp_path, fixture_dir, truths)

    analyze_run(run)
    first = {p.name: p.read_bytes() for p in run.rglob("result.json")}

    analyze_run(run)
    second = {p.name: p.read_bytes() for p in run.rglob("result.json")}

    assert first.keys() == second.keys()
    differing = [k for k in first if first[k] != second[k]]
    assert not differing, f"re-analysis differed for: {differing}"


def test_results_are_ordered_stably(tmp_path, fixture_dir, truths):
    run = _make_run(tmp_path, fixture_dir, truths)
    a = [r.call_id for r in analyze_run(run, write=False)]
    b = [r.call_id for r in analyze_run(run, write=False)]
    assert a == b == sorted(a)


def test_result_is_json_serialisable(fixture_dir, truths):
    result, _ = _measure(fixture_dir, truths["clean_500ms"])
    round_tripped = json.loads(json.dumps(asdict(result), sort_keys=True))
    assert round_tripped["ttfab_onset_ms"] == pytest.approx(result.ttfab_onset_ms)


def test_response_duration_is_recorded(fixture_dir, truths):
    """Needed to detect vendors whose TTS waits for the whole answer."""
    result, _ = _measure(fixture_dir, truths["clean_500ms"])
    assert result.vendor_response_duration_ms is not None
    assert result.vendor_response_duration_ms > 100.0


def test_content_and_onset_differ_when_filler_leads(fixture_dir, truths):
    result, _ = _measure(fixture_dir, truths["filler_first"])
    assert result.ttfab_onset_ms is not None
    assert result.ttfab_content_ms is not None
    assert result.ttfab_content_ms > result.ttfab_onset_ms


def test_negative_ttfab_is_reported_not_discarded(fixture_dir, truths):
    result, _ = _measure(fixture_dir, truths["barge_in"])
    assert result.discard_reason is None
    assert result.ttfab_onset_ms < 0
