"""Matching bench calls to vendor billing rows, and pricing a minute.

We dial INTO the vendor, so the call is inbound from its side and our Plivo
uuid means nothing to it. Nothing links the two records except the caller's
number and the wall clock, which makes every pairing an inference -- and a
wrong inference prices a call at its neighbour's cost, silently, with a
plausible-looking number. These tests pin the conservatism that prevents it.
"""

from __future__ import annotations

import json

import pytest

from harness import costs as C
from vendors.base import CallCost, epoch_to_iso

BASE = 1_785_458_100.0  # arbitrary fixed epoch; no wall-clock reads in tests
OURS = "+15550001234"          # fictional; never a real account number


def cost(at: float, *, cost_usd=0.05, duration=45.0, billed=None,
         caller=OURS, call_id="v1", notes=()) -> CallCost:
    return CallCost(
        vendor_call_id=call_id,
        cost=cost_usd,
        currency="USD",
        duration_s=duration,
        billed_s=billed,
        started_at=epoch_to_iso(at),
        caller=caller,
        source="test",
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_a_call_matches_the_billing_row_that_starts_just_after_it():
    """Our clock starts at `placed`, before ringing; theirs starts on connect."""
    matched, unmatched = C.match({"call-000": BASE},
                                 [cost(BASE + 2.0)], OURS)

    assert [m.call_id for m in matched] == ["call-000"]
    assert matched[0].skew_s == 2.0
    assert unmatched == []


def test_a_row_from_a_different_caller_is_never_claimed():
    """The account under test takes other traffic. Time alone would match it."""
    matched, unmatched = C.match({"call-000": BASE},
                                 [cost(BASE + 2.0, caller="+15550001111")], OURS)

    assert matched == []
    assert unmatched == ["call-000"]


def test_a_row_outside_the_window_is_not_claimed():
    matched, _ = C.match({"call-000": BASE},
                         [cost(BASE + C.MATCH_AFTER_S + 5)], OURS)
    assert matched == []


def test_two_calls_can_never_share_one_billing_row():
    """The failure this whole module exists to prevent: one row counted twice
    doubles a vendor's reported spend and prices a call it never paid for."""
    row = cost(BASE + 2.0, call_id="only-one")
    matched, unmatched = C.match({"call-000": BASE, "call-001": BASE + 1.0},
                                 [row], OURS)

    assert len(matched) == 1
    assert len(unmatched) == 1
    assert {m.cost.vendor_call_id for m in matched} == {"only-one"}


def test_the_closest_pairing_wins_regardless_of_iteration_order():
    """Two near-simultaneous candidates must not be split by dict order: the
    globally closest pair is taken first, then both sides are spent."""
    ours = {"call-000": BASE, "call-001": BASE + 20.0}
    theirs = [cost(BASE + 21.0, call_id="late"), cost(BASE + 1.0, call_id="early")]

    matched, unmatched = C.match(ours, theirs, OURS)

    by_call = {m.call_id: m.cost.vendor_call_id for m in matched}
    assert by_call == {"call-000": "early", "call-001": "late"}
    assert unmatched == []


def test_an_unpriced_call_is_named_not_silently_dropped():
    """Unmatched means either billing lag or a matching failure, and those want
    different fixes -- a count would hide which."""
    matched, unmatched = C.match({"call-000": BASE, "call-001": BASE + 100.0},
                                 [cost(BASE + 1.0)], OURS)

    assert [m.call_id for m in matched] == ["call-000"]
    assert unmatched == ["call-001"]


def test_a_vendor_that_hides_the_caller_still_matches_on_time():
    """Not every platform discloses the inbound number. Refusing to match at
    all would report that vendor as costless, which is worse than pairing on a
    30-second window we already know is narrower than the gap between calls."""
    matched, _ = C.match({"call-000": BASE}, [cost(BASE + 2.0, caller=None)], OURS)
    assert len(matched) == 1


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def test_cost_per_minute_pools_totals_rather_than_averaging_rates():
    """A mean of per-call rates would weight a 6-second call the same as a
    60-second one. Pooled: 0.10 over 66 s = 0.0909/min. Averaged rates would
    give (0.50 + 0.05)/2 = 0.275/min -- three times too high."""
    matched = [
        C.MatchedCost("call-000", cost(BASE, cost_usd=0.05, duration=6.0), 0.0),
        C.MatchedCost("call-001", cost(BASE, cost_usd=0.05, duration=60.0), 0.0),
    ]

    summary = C.summarise(matched)

    assert summary["cost_per_minute"] == pytest.approx(0.0909, abs=1e-4)


def test_a_billing_minimum_produces_two_different_per_minute_figures():
    """Telnyx bills 60 s for a 42 s call. Cost per minute of conversation and
    cost per billed minute are both true and they are not the same number;
    collapsing them would either understate the invoice or overstate the rate."""
    matched = [C.MatchedCost(
        "call-000", cost(BASE, cost_usd=0.05, duration=42.0, billed=60.0), 0.0)]

    summary = C.summarise(matched)

    assert summary["cost_per_minute"] == pytest.approx(0.0714, abs=1e-4)
    assert summary["cost_per_billed_minute"] == pytest.approx(0.05, abs=1e-4)


def test_billed_seconds_fall_back_to_duration_when_not_reported():
    """Most platforms report no separate billed figure; they bill what they
    measured. The two rates must then agree rather than one going null."""
    matched = [C.MatchedCost("call-000", cost(BASE, cost_usd=0.06, duration=60.0), 0.0)]

    summary = C.summarise(matched)

    assert summary["cost_per_minute"] == summary["cost_per_billed_minute"]


def test_an_unpriced_row_is_counted_as_matched_but_not_as_money():
    """A matched call whose cost the vendor has not posted yet must not enter
    the total as zero -- that would drag the reported rate down."""
    matched = [
        C.MatchedCost("call-000", cost(BASE, cost_usd=0.06, duration=60.0), 0.0),
        C.MatchedCost("call-001", cost(BASE, cost_usd=None, duration=60.0), 0.0),
    ]

    summary = C.summarise(matched)

    assert summary["calls_matched"] == 2
    assert summary["calls_priced"] == 1
    assert summary["cost_per_minute"] == pytest.approx(0.06, abs=1e-6)


def test_mixed_currencies_refuse_to_collapse_into_one():
    """Summing dollars and euros into a single figure is the one arithmetic
    error a cost benchmark must never make silently."""
    euro = CallCost("e1", 0.05, "EUR", 60.0, None, epoch_to_iso(BASE), OURS)
    matched = [
        C.MatchedCost("call-000", cost(BASE, cost_usd=0.06, duration=60.0), 0.0),
        C.MatchedCost("call-001", euro, 0.0),
    ]

    summary = C.summarise(matched)

    assert summary["currency"] is None
    assert summary["currencies"] == ["EUR", "USD"]


def test_notes_survive_into_the_summary():
    """Caveats travel WITH the number: a free-tier price or a unit conversion
    is not an aside, it is what the figure means."""
    matched = [C.MatchedCost(
        "call-000", cost(BASE, notes=("account tier is 'free'",)), 0.0)]

    assert C.summarise(matched)["notes"] == ["account tier is 'free'"]


def test_no_priced_calls_reports_none_rather_than_zero():
    """Zero cost per minute would read as 'free', not as 'unknown'."""
    summary = C.summarise([])

    assert summary["cost_per_minute"] is None
    assert summary["total_cost"] is None


# --------------------------------------------------------------------------- #
# Reading our own side of the pairing
# --------------------------------------------------------------------------- #


def test_our_start_comes_from_the_placed_event(tmp_path):
    call_dir = tmp_path / "call-000"
    call_dir.mkdir()
    (call_dir / "events.jsonl").write_text(
        json.dumps({"event": "dialing", "wall": BASE - 1}) + "\n"
        + json.dumps({"event": "placed", "wall": BASE}) + "\n"
        + json.dumps({"event": "answered", "wall": BASE + 3}) + "\n")

    assert C.our_call_started(call_dir) == BASE


def test_a_call_with_no_event_log_has_no_start(tmp_path):
    call_dir = tmp_path / "call-000"
    call_dir.mkdir()

    assert C.our_call_started(call_dir) is None
