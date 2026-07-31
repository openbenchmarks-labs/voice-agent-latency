"""What the report is allowed to claim about the instrument.

These tests exist because the failure they guard against is silent: a report
that quotes an overhead or noise figure it never measured reads as rigorous and
is wrong. No characterisation exists for this measurement path, and a
characterisation is only valid for the host, carrier and audio path it was
measured on -- so there is nothing to borrow, by design.
"""

from __future__ import annotations

from harness import instrument
from harness.bench import MODE, honesty_block, instrument_block

CALLER_RECEIPT = {"voice": "Polly.Joanna", "sha256": "abc123"}
BARE_REPORT = {"usable": 3, "per_turn": {}, "discards": {}}


# --------------------------------------------------------------------------- #
# The instrument the run actually had (none)
# --------------------------------------------------------------------------- #


def test_no_characterisation_exists_for_the_scripted_dialog_path():
    assert instrument.for_mode(MODE) is None
    # No mode has one. There is no characterisation to borrow, by construction.
    assert set(instrument.CURRENT_FOR_MODE.values()) == {None}


def test_the_instrument_block_says_unknown_instead_of_borrowing_numbers():
    """Absent must render as absent. The backfill leaves instrument_id null
    when there is no sweep_run, and the page shows "instrument unknown" -- a
    borrowed figure would instead be published as a property of these calls."""
    block = instrument_block(MODE)
    assert block["valid"] is False
    assert block["note"]
    for borrowed in ("sweep_run", "host", "noise_sd_ms", "slope",
                     "path_overhead_ms"):
        assert borrowed not in block


def test_overhead_is_never_subtracted_in_any_mode():
    """The no-subtraction rule holds whether or not the overhead is known: a
    correction would turn a stated overhead into an invisible one."""
    assert instrument_block(MODE)["path_overhead_subtracted"] is False
    assert instrument_block("some_future_mode")["path_overhead_subtracted"] is False


def test_an_uncharacterised_run_publishes_the_absence_not_a_number():
    """Every field a characterisation would fill comes back null, and the note
    says why -- so a reader cannot mistake silence for a clean instrument."""
    block = instrument_block(MODE)
    assert block["valid"] is False
    assert "unmeasured" in block["note"]
    assert block["path_overhead_subtracted"] is False
    # The fields a characterisation would fill are absent, not zero -- a zero
    # would read as "no overhead", which is a claim we have not earned.
    for field in ("sweep_run", "host", "noise_sd_ms", "slope", "path_overhead_ms"):
        assert field not in block


# --------------------------------------------------------------------------- #
# What the prose may and may not say
# --------------------------------------------------------------------------- #


def test_the_honesty_block_states_the_overhead_is_unmeasured_here():
    text = " ".join(honesty_block(BARE_REPORT, CALLER_RECEIPT))
    assert "UNCHARACTERISED" in text
    assert "unmeasured" in text
    # The rule itself, not a citation marker: the prose has to say the overhead
    # is left in, because that is what makes every figure an upper bound.
    assert "none is subtracted" in text


def test_no_overhead_or_noise_figure_is_quoted_at_all():
    """A characterisation would license an overhead figure. None exists, so the
    prose must not contain one -- the failure mode is a plausible number with no
    measurement behind it."""
    text = " ".join(honesty_block(BARE_REPORT, CALLER_RECEIPT))
    assert "126 ms" not in text
    assert "8.8 ms" not in text
    assert "sweep-" not in text


def test_the_self_reported_latency_finding_is_published_with_its_provenance():
    """Worth publishing, and worth naming the run and n it came from rather than
    asserting it as a general truth about the industry."""
    text = " ".join(honesty_block(BARE_REPORT, CALLER_RECEIPT))
    gap = instrument.SELF_REPORTED_LATENCY_GAP
    assert f"{gap.floor_ms:.0f} ms" in text
    assert gap.bench_run in text
    assert "at the platform's edge" in text
