"""What the instrument contributes to every number, measured rather than assumed.

These are not tuning knobs. They are recorded measurements with provenance, so the
report can state its own overhead instead of hand-waving about it -- and so that
nothing silently subtracts them: path overhead is reported, never removed.

IMPORTANT -- there is currently NO characterisation. Earlier figures were
measured against a known-delay rig on a measurement path this bench no longer
uses, and a characterisation is only valid for the host, the carrier and the
audio path it was measured on. Rather than carry numbers that would look
reassuring and be wrong, `for_mode` returns None and the report states the
absence.

`Characterisation` below is therefore a shape, not a value: it documents what a
future characterisation must publish. Producing one needs a far end that answers
after a known delay and is reachable over the PSTN -- tracked, not built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Characterisation:
    """One measurement of the instrument against a known-delay far end."""

    host: str
    sweep_run: str
    n_calls: int
    noise_sd_ms: float
    slope: float
    slope_gate_passed: bool
    path_overhead_ms: float
    note: str


#: Which characterisation applies to which measurement path. A mode absent from
#: this map (or mapped to None) has no instrument: no overhead figure, no noise
#: figure, no slope. Callers must handle None; there is no value to fall back to,
#: which is deliberate.
CURRENT_FOR_MODE: dict[str, "Characterisation | None"] = {
    "scripted_dialog": None,      # Plivo Speak/GetInput -- uncharacterised
}


def for_mode(mode: str) -> "Characterisation | None":
    return CURRENT_FOR_MODE.get(mode)


@dataclass(frozen=True)
class TwoTapFinding:
    """What a platform's own view of a call reads, against ours of the same call.

    Same call, two vantage points. A platform measures at its own edge; the
    caller is roughly half a second further away, and that distance is the
    finding.
    """

    bench_run: str
    n_turns: int
    floor_ms: float
    floor_sd_ms: float
    note: str


# The reason this matters: measured from the same VM, against a far end whose true
# response time we know, our reading is 126 ms high. Against Telnyx's assistant,
# measured against the assistant's OWN recording, it is 547 ms high.
#
# The assistant sits inside Telnyx, so its audio reaches the recording point over
# a SHORTER path than our VM's does -- its overhead should therefore be smaller
# than 126 ms, not larger. Distance cannot explain the difference, and it points
# the wrong way to try.
#
# Independent support: Telnyx's own published first-token latencies are 610-760 ms
# (telnyx.com/resources/reducing-latency-stateful-actors). A 440 ms median that is
# supposed to contain endpointing + first token + first audio does not fit that.
#
#: The same question asked of a REPORTED number rather than a recording, on the
#: current path: our TTFAB against the latency the platform publishes for the
#: identical calls. Matched per call by caller id and start time.
SELF_REPORTED_LATENCY_GAP = TwoTapFinding(
    bench_run="bench-retell-20260730-181628",
    n_turns=10,          # matched CALLS here; the vendor reports per-call p50
    floor_ms=489.5,
    floor_sd_ms=78.4,
    note="platform's own reported latency reads lower than audio measured at "
         "the caller; consistent with the gap between its own recording tap "
         "and ours, so it reflects vantage point rather than a stamping bug",
)
