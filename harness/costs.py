"""Pair bench calls with the vendor's own billing records, and price a minute.

We dial INTO the vendor, so we never learn its call id -- the call is inbound
from its side and our Plivo uuid means nothing to it. The only things both ends
share are the caller's number and the wall clock. That makes matching an
inference, and an inference that can be wrong: attribute the wrong record and a
call is priced at its neighbour's cost, silently and plausibly.

So the matching here is deliberately conservative:

  * the caller must match ours (digits only -- '+1555…' and '555…' are one
    number), when the vendor discloses it at all
  * the vendor's start must fall in a window around ours, wide enough for
    ring-and-answer and narrower than the gap between consecutive bench calls
  * each vendor record is claimed at most ONCE, nearest first, so two bench
    calls can never both point at the same billing row
  * anything left ambiguous stays unmatched, and unmatched is reported

Cost per minute is then total cost over total duration -- pooled, not an
average of per-call rates, for the same reason the latency figures pool turns:
a mean of ratios weights a 3-second call the same as a 60-second one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vendors.base import CallCost, digits

#: How far after our dial the vendor's own start may fall. Our clock starts at
#: `placed` (before ringing); theirs starts when the call connects, a second or
#: two later. Bench calls are ~50 s apart, so this cannot reach the next one.
MATCH_AFTER_S = 30.0

#: And how far before. Small but not zero: clock skew between our host and a
#: vendor's billing timestamps is real and one-sided in neither direction.
MATCH_BEFORE_S = 10.0

MANIFEST_NAME = "costs.json"


@dataclass
class MatchedCost:
    call_id: str
    cost: CallCost
    skew_s: float          # vendor start minus ours; provenance for the match


def our_call_started(call_dir: Path) -> float | None:
    """Wall-clock seconds when we placed this call, from its event log."""
    events = call_dir / "events.jsonl"
    if not events.is_file():
        return None
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "placed" and event.get("wall"):
            return float(event["wall"])
    return None


def match(ours: dict[str, float], theirs: list[CallCost],
          caller: str | None) -> tuple[list[MatchedCost], list[str]]:
    """Pair our calls with vendor billing rows. Returns (matched, unmatched).

    `ours` maps call_id -> wall-clock start. Greedy by smallest skew across all
    candidate pairs, so a near-simultaneous pair cannot be split by iteration
    order -- the closest pairing is taken first and both sides are then spent.
    """
    wanted = digits(caller) if caller else None
    candidates = []
    for call_id, started in sorted(ours.items()):
        for index, cost in enumerate(theirs):
            if wanted and cost.caller and digits(cost.caller) != wanted:
                continue
            their_start = _epoch(cost)
            if their_start is None:
                continue
            skew = their_start - started
            if -MATCH_BEFORE_S <= skew <= MATCH_AFTER_S:
                candidates.append((abs(skew), skew, call_id, index))

    candidates.sort()
    used_calls: set[str] = set()
    used_rows: set[int] = set()
    matched: list[MatchedCost] = []
    for _, skew, call_id, index in candidates:
        if call_id in used_calls or index in used_rows:
            continue
        used_calls.add(call_id)
        used_rows.add(index)
        matched.append(MatchedCost(call_id, theirs[index], round(skew, 3)))

    matched.sort(key=lambda m: m.call_id)
    unmatched = sorted(set(ours) - used_calls)
    return matched, unmatched


def _epoch(cost: CallCost) -> float | None:
    from vendors.base import iso_to_epoch
    return iso_to_epoch(cost.started_at)


def summarise(matched: list[MatchedCost]) -> dict:
    """Cost per minute, pooled over every matched call.

    Two denominators, because two honest questions: `duration` is what the
    conversation actually took, `billed` is what the vendor charged for. They
    differ wherever a platform applies a minimum, and reporting only the first
    would understate the invoice while reporting only the second would overstate
    the running cost of a longer call.
    """
    priced = [m for m in matched if m.cost.cost is not None]
    total_cost = sum(float(m.cost.cost) for m in priced)
    total_duration = sum(float(m.cost.duration_s or 0) for m in priced)
    total_billed = sum(float(m.cost.billed_s or m.cost.duration_s or 0)
                       for m in priced)

    currencies = sorted({m.cost.currency for m in priced if m.cost.currency})
    notes = sorted({note for m in priced for note in m.cost.notes})

    return {
        "calls_priced": len(priced),
        "calls_matched": len(matched),
        "currency": currencies[0] if len(currencies) == 1 else None,
        "currencies": currencies,
        "total_cost": round(total_cost, 6) if priced else None,
        "total_duration_s": round(total_duration, 3) if priced else None,
        "total_billed_s": round(total_billed, 3) if priced else None,
        "cost_per_minute": (round(total_cost / total_duration * 60.0, 6)
                            if total_duration else None),
        "cost_per_billed_minute": (round(total_cost / total_billed * 60.0, 6)
                                   if total_billed else None),
        "cost_per_call_mean": (round(total_cost / len(priced), 6)
                               if priced else None),
        "notes": notes,
    }


def manifest(run_id: str, vendor: str, matched: list[MatchedCost],
             unmatched: list[str], caller: str | None) -> dict:
    return {
        "run_id": run_id,
        "vendor": vendor,
        "caller": caller,
        "summary": summarise(matched),
        # Named, not counted: an unmatched call is either billing lag (come
        # back later) or a matching failure, and those want different fixes.
        "unmatched_calls": unmatched,
        "calls": {m.call_id: {**m.cost.as_dict(), "match_skew_s": m.skew_s}
                  for m in matched},
    }


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text()) or {}


def write_manifest(run_dir: Path, payload: dict) -> Path:
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def collect(vendor_name: str, adapter, run_dir: Path, caller: str | None,
            pad_s: float = 300.0) -> dict:
    """Fetch, match and summarise one run's costs. Returns the manifest."""
    call_dirs = sorted(run_dir.glob("call-*"))
    ours = {}
    for call_dir in call_dirs:
        started = our_call_started(call_dir)
        if started is not None:
            ours[call_dir.name] = started
    if not ours:
        return manifest(run_dir.name, vendor_name, [], [], caller)

    window_start = min(ours.values()) - pad_s
    window_end = max(ours.values()) + pad_s
    theirs = adapter.call_costs(window_start, window_end)
    matched, unmatched = match(ours, theirs, caller)
    return manifest(run_dir.name, vendor_name, matched, unmatched, caller)
