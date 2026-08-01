"""Place, drive and measure ONE call. Shared by both bench modes.

Pulled out of harness/bench.py so the sequential loop and the synchronised
multi-vendor loop cannot drift apart. A difference between them would be
invisible in the output and would show up as a vendor-vs-vendor difference,
which is exactly the thing this bench is supposed to isolate.

Nothing here times anything that becomes a reported number. `placed_at` and
`ended_at` are wall-clock bookkeeping -- they say when a call happened, never how
long the agent took. TTFAB comes from the analyzer reading the recording.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from analyzer.measure import measure_call

from .dialog import (
    ANSWER_TIMEOUT_S,
    CALL_DEADLINE_S,
    DialogSession,
    channel_map,
    fetch_recording,
)

#: Grace after the dialog ends, waiting for the carrier's hangup webhook.
HANGUP_GRACE_S = 10.0

#: Give up on a call that has gone SILENT for this long, rather than sitting out
#: the full CALL_DEADLINE_S.
#:
#: The distinction matters. A slow call is still emitting events -- our line
#: served, the vendor's reply transcribed -- every few seconds. A stalled call
#: emits nothing, which is what happens when the carrier stops POSTing the
#: dialog action (its GetInput timeout does not always call back). Waiting the
#: full deadline for one is 180s of dead time; cutting the deadline instead
#: would truncate the other, because the legitimate ceiling is ~163s
#: (greeting_timeout + 4 x execution_timeout + our speech).
#:
#: 75s is comfortably above the longest legitimate silence: a 20s greeting
#: listen or a 30s turn listen, plus margin.
IDLE_ABORT_S = 75.0


def place_and_measure(*, call_id: str, run_dir: Path, script, vendor: str,
                      run_id: str, target: str, carrier, registry,
                      applied: dict, caller_receipt: dict, mode: str,
                      verify_skipped: bool = False, extra: dict | None = None):
    """Run one call end to end. Returns (result, session).

    `result` is None only if the call was never answered -- there is no audio to
    measure and no honest number to report, so the caller records the attempt and
    moves on rather than inventing one.
    """
    out_dir = run_dir / call_id
    out_dir.mkdir(parents=True, exist_ok=True)
    call = DialogSession(call_id=call_id, out_dir=out_dir, script=script)
    # Registered BEFORE dialling: the answer webhook can arrive while
    # place_call is still returning, and an unregistered token is a 409 that
    # silently kills the call.
    registry.add(call)
    try:
        # Token-scoped webhook paths -- with several calls live at once, the
        # carrier's POSTs are only attributable by the token in the URL.
        call.call_sid = carrier.place_call(
            target,
            answer_path=f"/webhooks/answer/{call.token}",
            status_path=f"/webhooks/hangup/{call.token}",
        )
        call.placed_at = call.placed_at or time.time()
        call.event("placed", call_sid=call.call_sid, vendor=vendor)

        # Answered, then finished, are two different waits. A call nobody picks
        # up has nothing to converse about, so it must not sit out the whole
        # conversation deadline.
        if not call.answered.wait(timeout=ANSWER_TIMEOUT_S):
            call.event("never_answered")
            # Per-call hangup, never hangup_all(): in synchronised mode the other
            # vendors' calls are live on the same account and would be killed.
            if call.call_control_id:
                carrier.hangup(call.call_control_id)
            call.ended_at = call.ended_at or time.time()
            return None, call

        # The dialog ends itself: the last turn's action webhook returns
        # <Hangup/> and sets dialog_done. Two backstops behind that -- the total
        # deadline, and idle silence, whichever comes first.
        deadline = time.time() + CALL_DEADLINE_S
        while not call.dialog_done.wait(timeout=2.0):
            now = time.time()
            if now >= deadline:
                call.event("call_deadline_reached")
                break
            idle = now - call.last_event_at
            if idle >= IDLE_ABORT_S:
                call.event("abandoned_idle", idle_s=round(idle, 1))
                break
        if call.call_control_id and not call.hangup_seen.is_set():
            carrier.hangup(call.call_control_id)
        call.hangup_seen.wait(timeout=HANGUP_GRACE_S)
        call.ended_at = call.ended_at or time.time()

        recording = fetch_recording(call, carrier)
        metadata = {
            "call_id": call_id, "run_id": run_id,
            "kind": "measure", "mode": mode, "vendor": vendor,
            "carrier": carrier.name,
            "vendor_config_sha256": applied["sha256"],
            # THIS call's script, not the run's plan -- calls differ.
            "caller_config_sha256": script.receipt()["sha256"],
            "caller_plan_sha256": caller_receipt["sha256"],
            "call_sid": call.call_sid,
            "call_control_id": call.call_control_id,
            "verify_skipped": verify_skipped,
            "cases": [t.case_id for t in script.turns if t.case_id],
            "turns_requested": script.n_turns,
            "turns_played": call.turns_spoken,
            "greeting_transcript": call.greeting_transcript,
            "greeting_timed_out": call.greeting_timed_out,
            "turns": call.turn_metadata(),
            "channel_map": channel_map(),
            # When the call happened, in wall clock. Bookkeeping, never a
            # measurement -- but it is what proves the synchronised mode really
            # did dial every vendor at the same moment.
            "timing": call.timing(),
        }
        # Caller-supplied fields (the synchronised mode stamps set_index and its
        # session id here). Applied last so nothing can quietly shadow a receipt.
        for k, v in (extra or {}).items():
            metadata.setdefault(k, v)
        if recording is None:
            metadata["recording_missing"] = True
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        result = measure_call(out_dir)
        (out_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
        return result, call
    finally:
        # Dropped either way. A finished call left in the registry would keep
        # `Registry.current` ambiguous for the untokenised fallback routes.
        registry.drop(call)
