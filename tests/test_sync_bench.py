"""The synchronised multi-vendor mode: what must be true of its concurrency.

The whole value of this mode is that vendor A's set 7 and vendor B's set 7
happened at the same moment. If the calls are not actually simultaneous, or if
one vendor's teardown reaches another vendor's live call, the mode produces
numbers that look comparable and are not -- which is worse than the sequential
bench it replaces, because the confound is now invisible.

These tests drive the real Registry and the real per-call state machine with a
fake carrier. No network, no vendor accounts, no cost.
"""

from __future__ import annotations

import json
import threading
import time

from harness.bench import Registry
from harness.dialog import DialogSession


class FakeCarrier:
    """Records what it was asked to do, and never touches a network.

    `hangup_all` is present and counted on purpose: reaching for it on this path
    is the one mistake that silently corrupts a synchronised run, so a test has
    to be able to see it happen.
    """

    name = "fake"

    def __init__(self):
        self.placed: list[tuple[str, str, str]] = []
        self.hung_up: list[str] = []
        self.hangup_all_calls = 0
        self._lock = threading.Lock()

    def place_call(self, to_number, *, answer_path="/webhooks/answer",
                   status_path="/webhooks/hangup"):
        with self._lock:
            self.placed.append((to_number, answer_path, status_path))
        return f"sid-{len(self.placed)}"

    def hangup(self, call_control_id):
        with self._lock:
            self.hung_up.append(call_control_id)

    def hangup_all(self):
        with self._lock:
            self.hangup_all_calls += 1


def _session(tmp_path, call_id="call-000"):
    out = tmp_path / call_id
    out.mkdir(parents=True, exist_ok=True)
    return DialogSession(call_id=call_id, out_dir=out, script=None)


# --------------------------------------------------------------------------- #
# The registry has to hold several calls at once
# --------------------------------------------------------------------------- #


def test_several_calls_are_addressable_at_once(tmp_path):
    """Sequential mode had one slot. With five vendors live, a single slot means
    four of them get a 409 on their first webhook and die."""
    registry = Registry()
    calls = [_session(tmp_path, f"call-{i:03d}") for i in range(5)]
    for c in calls:
        registry.add(c)

    for c in calls:
        assert registry.get(c.token) is c


def test_a_token_routes_to_its_own_call_and_nothing_else(tmp_path):
    """The token is the routing key AND the auth. A webhook for one call must
    never be able to drive another -- that would put words in the wrong mouth."""
    registry = Registry()
    a, b = _session(tmp_path, "call-000"), _session(tmp_path, "call-001")
    registry.add(a)
    registry.add(b)

    assert registry.get(a.token) is a
    assert registry.get(b.token) is b
    assert registry.get("not-a-real-token") is None


def test_current_refuses_to_guess_when_several_calls_are_live(tmp_path):
    """The untokenised fallback routes resolve via `current`. With more than one
    call in flight there is no right answer, and guessing would drive the wrong
    conversation -- so it returns None and the route 409s, which is recoverable."""
    registry = Registry()
    a, b = _session(tmp_path, "call-000"), _session(tmp_path, "call-001")

    registry.add(a)
    assert registry.current is a          # unambiguous
    registry.add(b)
    assert registry.current is None       # ambiguous -> refuse

    registry.drop(b)
    assert registry.current is a          # unambiguous again


def test_dropping_a_finished_call_restores_the_fallback(tmp_path):
    registry = Registry()
    calls = [_session(tmp_path, f"call-{i:03d}") for i in range(3)]
    for c in calls:
        registry.add(c)
    for c in calls[1:]:
        registry.drop(c)

    assert registry.current is calls[0]


# --------------------------------------------------------------------------- #
# Simultaneity
# --------------------------------------------------------------------------- #


def test_the_barrier_makes_the_dials_simultaneous_not_the_setup():
    """Without a barrier the first vendor is dialled while the last is still
    being resolved, and the spread between placements is the setup cost. The
    barrier moves the synchronisation point to the dial itself."""
    n = 5
    gate = threading.Barrier(n)
    placed_at: list[float] = []
    lock = threading.Lock()

    def worker(i):
        # Uneven setup, deliberately: this is what the barrier has to absorb.
        time.sleep(0.02 * i)
        gate.wait(timeout=5.0)
        with lock:
            placed_at.append(time.perf_counter())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(placed_at) == n
    # Setup was staggered across 80 ms; the dials must land far tighter than that.
    assert max(placed_at) - min(placed_at) < 0.02


def test_timing_records_both_iso_and_epoch(tmp_path):
    """ISO for a human comparing vendors, epoch for arithmetic. Deriving one from
    the other at read time is where timezone bugs come from."""
    call = _session(tmp_path)
    call.placed_at = 1_785_458_100.0
    call.answered_at = 1_785_458_103.5
    call.ended_at = 1_785_458_148.0

    t = call.timing()

    assert t["placed_at"].startswith("2026-")
    assert t["placed_at"].endswith("+00:00")          # always UTC
    assert t["placed_at_epoch"] == 1_785_458_100.0
    assert t["wall_duration_s"] == 48.0
    assert t["answer_delay_s"] == 3.5


def test_timing_is_all_none_before_the_call_starts(tmp_path):
    """An unplaced call must not report a duration of zero -- zero would read as
    an instant call rather than an absent one."""
    t = _session(tmp_path).timing()

    assert t["placed_at"] is None and t["ended_at"] is None
    assert "wall_duration_s" not in t


# --------------------------------------------------------------------------- #
# The failure that would silently corrupt a synchronised run
# --------------------------------------------------------------------------- #


def test_the_shared_driver_never_calls_hangup_all():
    """`hangup_all()` ends every call on the account. On this path the other
    vendors' calls are live, so it would truncate their conversations mid-turn
    and the damage would look like those platforms being unreliable.

    Asserted against the source because the alternative is discovering it on a
    100-call run across five paid accounts.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent
              / "harness" / "onecall.py").read_text()

    assert "hangup_all" not in source.replace(
        "# Per-call hangup, never hangup_all(): in synchronised mode the other", "")


def test_webhook_paths_carry_the_call_token(tmp_path):
    """With several calls live, an untokenised answer URL is unattributable --
    the carrier's POST could belong to any of them."""
    call = _session(tmp_path)
    answer = f"/webhooks/answer/{call.token}"
    status = f"/webhooks/hangup/{call.token}"

    assert call.token in answer and call.token in status
    assert len(call.token) >= 16          # unguessable, since the URL is the auth


# --------------------------------------------------------------------------- #
# Bounding a stalled call without truncating a slow one
# --------------------------------------------------------------------------- #


def test_the_idle_threshold_clears_the_longest_legitimate_silence():
    """A call is allowed to be silent for a whole listen. The greeting listen is
    greeting_timeout long and a turn listen is execution_timeout long, so the
    idle threshold has to sit above both or slow-but-valid calls get abandoned
    mid-conversation and their turns counted as failures."""
    import yaml
    from harness.onecall import IDLE_ABORT_S
    from harness.config import _PKG_ROOT

    cfg = yaml.safe_load((_PKG_ROOT / "config" / "dialog.yaml").read_text())
    longest_legitimate_silence = max(cfg["greeting_timeout"],
                                     cfg["execution_timeout"])

    assert IDLE_ABORT_S > longest_legitimate_silence * 1.5


def test_the_total_deadline_still_clears_the_worst_legitimate_call():
    """Cutting CALL_DEADLINE_S is the tempting way to bound a stall, and it is
    the wrong one: a call that uses every listen in full is legitimate and must
    finish. This is the arithmetic that says how little room there is."""
    import yaml
    from harness.dialog import CALL_DEADLINE_S
    from harness.config import _PKG_ROOT

    cfg = yaml.safe_load((_PKG_ROOT / "config" / "dialog.yaml").read_text())
    turns = 4
    our_line_s = 5
    worst_legitimate = (cfg["greeting_timeout"]
                        + turns * (cfg["execution_timeout"] + our_line_s))

    assert CALL_DEADLINE_S > worst_legitimate


def test_an_event_refreshes_the_idle_clock(tmp_path):
    """Idle is measured from the last event, not from the call's start -- a call
    making steady progress must never trip the abort."""
    call = _session(tmp_path)
    before = call.last_event_at
    time.sleep(0.01)
    call.event("turn_prompt_served", turn=1)

    assert call.last_event_at > before


# --------------------------------------------------------------------------- #
# Repairing sets inside an existing run
# --------------------------------------------------------------------------- #


def _attachable(tmp_path, monkeypatch, *, caller_sha, applied_sha,
                stored_caller, stored_applied):
    """A Vendor wired for attach() without touching a vendor account."""
    from harness import sync_bench as S
    from harness.sync_bench import Vendor

    run_dir = tmp_path / "runs" / "bench-fake-20260731-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "caller_config.json").write_text(
        json.dumps({"sha256": stored_caller, "rotation_length": 50}))
    (run_dir / "applied_config.json").write_text(
        json.dumps({"sha256": stored_applied}))
    monkeypatch.setattr(S.settings, "runs_dir", tmp_path / "runs")

    v = object.__new__(Vendor)
    v.slug = "fake"
    v.caller_receipt = {"sha256": caller_sha, "rotation_length": 50}
    v.applied = {"sha256": applied_sha}
    return v


def test_repair_refuses_when_the_question_rotation_changed(tmp_path, monkeypatch):
    """Set 18 is only set 18 for as long as the rotation is unchanged. Adding a
    question set renumbers everything after it, so a repair against a newer
    dialog.yaml would drop DIFFERENT questions into the slot and pool them with
    the originals -- a corruption no later reader could detect."""
    v = _attachable(tmp_path, monkeypatch, caller_sha="new-plan",
                    applied_sha="same", stored_caller="old-plan",
                    stored_applied="same")

    refusals = v.attach("bench-fake-20260731-000000")

    assert len(refusals) == 1
    assert "rotation changed" in refusals[0]


def test_repair_refuses_when_the_vendor_config_changed(tmp_path, monkeypatch):
    """Every call in a run carries the config receipt it was measured under. A
    repaired call under a different one is a different instrument sharing a
    directory with the old one."""
    v = _attachable(tmp_path, monkeypatch, caller_sha="same",
                    applied_sha="new-config", stored_caller="same",
                    stored_applied="old-config")

    refusals = v.attach("bench-fake-20260731-000000")

    assert len(refusals) == 1
    assert "same instrument" in refusals[0]


def test_repair_accepts_matching_receipts(tmp_path, monkeypatch):
    v = _attachable(tmp_path, monkeypatch, caller_sha="same", applied_sha="cfg",
                    stored_caller="same", stored_applied="cfg")

    assert v.attach("bench-fake-20260731-000000") == []


def test_superseded_calls_leave_the_run_directory(tmp_path, monkeypatch):
    """analyze_run measures EVERY subdirectory of a run. A call left behind as
    `call-018.old` would be measured as an extra call and join the percentiles,
    so the replaced call has to leave the run entirely -- while still existing,
    because it is the evidence for whatever went wrong."""
    from harness import sync_bench as S

    runs = tmp_path / "runs"
    run_dir = runs / "bench-fake-20260731-000000"
    (run_dir / "call-018").mkdir(parents=True)
    (run_dir / "call-018" / "recording.wav").write_bytes(b"audio")
    monkeypatch.setattr(S.settings, "runs_dir", runs)

    moved = S.supersede(run_dir, "call-018", "20260731-235959")

    assert not (run_dir / "call-018").exists()
    assert [p.name for p in run_dir.iterdir() if p.is_dir()] == []
    assert moved.is_dir() and (moved / "recording.wav").read_bytes() == b"audio"
    assert run_dir not in moved.parents


def test_superseding_a_call_that_is_not_there_is_not_an_error(tmp_path, monkeypatch):
    """A never-answered call has a directory; a call that was never reached at
    all does not. Repairing both in one command must not need two code paths."""
    from harness import sync_bench as S

    run_dir = tmp_path / "runs" / "bench-fake-20260731-000000"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(S.settings, "runs_dir", tmp_path / "runs")

    assert S.supersede(run_dir, "call-018", "20260731-235959") is None
