"""Which run represents a vendor on the board.

Exporting is additive to the run registry but REPLACING for the vendor entry,
and the vendor entry is what the board renders. Getting the choice wrong is
silent: the numbers are real and the label is real, they just belong to
different runs.
"""

from __future__ import annotations

from tools.export_snapshot import promotes

OLDER = {"run_id": "bench-telnyx-20260730-173213",
         "started_at": "2026-07-30T17:32:13+00:00"}
NEWER = {"run_id": "bench-telnyx-20260730-173501",
         "started_at": "2026-07-30T17:35:01+00:00"}


def test_the_first_run_for_a_vendor_is_always_the_entry():
    assert promotes(NEWER, None) is True


def test_a_newer_run_replaces_the_entry():
    assert promotes(NEWER, OLDER) is True


def test_re_exporting_an_older_run_does_not_demote_the_vendor():
    """The 2026-07-31 case: restoring a 1-call smoke test into the snapshot
    pointed the telnyx row at 4 turns instead of the 36 it had measured."""
    assert promotes(OLDER, NEWER) is False


def test_re_exporting_the_same_run_refreshes_it():
    """Re-analyzing a run and exporting it again must update the entry, not be
    rejected for failing to be strictly newer than itself."""
    assert promotes(NEWER, NEWER) is True


def test_a_run_with_no_timestamp_never_displaces_one_that_has_one():
    """run_started_at parses the id; a run whose id does not carry a stamp
    would otherwise compare as '' and win or lose by accident."""
    undated = {"run_id": "bench-telnyx-probe", "started_at": None}
    assert promotes(undated, NEWER) is False
    assert promotes(NEWER, undated) is True
