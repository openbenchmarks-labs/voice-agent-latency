"""The vendor interface. Small on purpose: a vendor is "an agent spec to check,
a number to dial, and a config receipt."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol


@dataclass(frozen=True)
class AgentSpec:
    """What WE require of the agent under test. Vendor-neutral.

    None means "accept the vendor's default and record what it chose" -- the
    closed-division philosophy: defaults are the product being measured.
    """

    system_prompt: str
    greeting: str
    model: str | None = None
    stt: str | None = None
    tts: str | None = None
    tools: tuple = ()  # V1: empty -- vendor-shipped defaults (e.g. hangup) are
    #                    recorded in AppliedConfig, not counted as ours


@dataclass(frozen=True)
class DialTarget:
    kind: Literal["pstn", "sip"]
    value: str  # E.164 or sip: URI


@dataclass(frozen=True)
class CallCost:
    """What one call cost, as the vendor's own billing API reports it.

    Never derived from a published price list: a rate card is what a platform
    advertises, this is what it charged. The two disagree in ways that matter
    (minimums, bundled telephony, promotional tiers), and the disagreement is
    exactly what a cost benchmark should surface rather than smooth over.

    `duration_s` is the vendor's own measure of the call, `billed_s` what it
    actually charged for. Telnyx bills a 60-second minimum, so a 42-second call
    bills 60 -- keeping both is what makes a cost-per-minute figure explicable
    instead of merely surprising.
    """

    vendor_call_id: str
    cost: float | None
    currency: str
    duration_s: float | None
    billed_s: float | None
    started_at: str | None      # ISO-8601 UTC
    caller: str | None          # the number WE dialled from, as they saw it
    breakdown: dict = field(default_factory=dict)
    source: str = ""            # endpoint + field this came from
    notes: tuple[str, ...] = ()  # caveats that travel WITH the number

    def as_dict(self) -> dict:
        return {
            "vendor_call_id": self.vendor_call_id,
            "cost": self.cost,
            "currency": self.currency,
            "duration_s": self.duration_s,
            "billed_s": self.billed_s,
            "started_at": self.started_at,
            "caller": self.caller,
            "breakdown": self.breakdown,
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass
class AppliedConfig:
    """The receipt: what the vendor is ACTUALLY running, at bench time."""

    vendor: str
    normalized: dict          # what we asked for (the AgentSpec, as dict)
    raw: dict                 # the vendor's live echo, verbatim
    defaults_used: dict       # knobs we left to the vendor + what it chose
    unsupported: list[str]    # knobs this vendor does not expose
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            canonical = json.dumps(self.raw, sort_keys=True, default=str)
            self.sha256 = hashlib.sha256(canonical.encode()).hexdigest()

    def as_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "normalized": self.normalized,
            "defaults_used": self.defaults_used,
            "unsupported": self.unsupported,
            "sha256": self.sha256,
            "raw": self.raw,
        }


def iso_to_epoch(text: str | None) -> float | None:
    """Unix seconds from an ISO-8601 timestamp, or None if unparseable.

    Vendors spell UTC differently ('Z', '+00:00', bare) and a naive datetime
    compared against a unix clock is off by the local offset -- silently, and
    by exactly enough to match a bench call to its neighbour. So anything
    without a zone is read as UTC, which is what all five APIs document.
    """
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def epoch_to_iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def stack_summary(defaults_used: dict) -> dict:
    """A vendor-neutral view of a receipt, for display only.

    Receipts are deliberately NOT uniform: each adapter names what its platform
    actually exposes, because flattening them would erase real differences (a
    Bland "tier" is not an LLM identity; Telnyx's eot_threshold and Vapi's
    waitSeconds are not the same knob). The cost is that a display hardcoding one
    vendor's key names prints "—" for every other vendor -- reporting a value the
    receipt does have as unknown, which is the one thing a receipt must never do.

    So this resolves only the aliases that genuinely mean the same thing, marks a
    capability tier as a tier rather than passing it off as a model name, and
    passes the endpointing block through as whatever that platform calls it.
    """
    tier = defaults_used.get("model_tier")
    versions = [
        defaults_used.get("version_id"),
        *(f"{part}={defaults_used[key]}"
          for part, key in (("agent", "agent_version"), ("llm", "llm_version"))
          if defaults_used.get(key) is not None),
    ]
    version = next((v for v in versions if v), None) or defaults_used.get("created_at")

    idle = next(
        (f"{label} {defaults_used[key]}"
         for label, key in (("idle reply", "user_idle_reply_secs"),
                            ("reminder", "reminder_trigger_ms"),
                            ("silence message", "silence_end_message"))
         if defaults_used.get(key)),
        None,
    )

    return {
        "model": defaults_used.get("model") or (f"{tier} (tier)" if tier else None),
        "stt": defaults_used.get("stt_model"),
        "voice": defaults_used.get("voice"),
        # Only the knobs this platform actually reports, under its own names.
        "endpointing": {
            key: value
            for key, value in (defaults_used.get("endpointing") or {}).items()
            if value is not None
        },
        "idle": idle,
        "version": version,
        "tools": [t for t in (defaults_used.get("tools") or []) if t],
    }


def digits(number: str | None) -> str:
    """A phone number reduced to its digits, for comparison.

    `+15551234567`, `15551234567` and `(555) 123-4567` name the same line; a
    formatting difference must never read as a different agent. Lives here
    because every adapter that checks a configured number against an account's
    numbers needs the same comparison.
    """
    return "".join(c for c in str(number or "") if c.isdigit())


class VendorAdapter(Protocol):
    name: str

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between the spec and the vendor's LIVE config.

        Empty list = ready to bench. Non-empty = the bench refuses to run and
        prints these as paste-ready fixes for the operator. READ-ONLY by contract:
        this checks, it never repairs.
        """
        ...

    def dial_target(self) -> DialTarget:
        """The number that reaches the agent -- VERIFIED, not assumed.

        An adapter must not return a hand-configured number without confirming
        it routes to the agent the receipt describes. A number pointing
        elsewhere yields a run where every turn is usable and every latency is
        plausible, published against a configuration that had nothing to do
        with it (measured 2026-07-30; see vendors/telnyx.py dial_target).

        Raises with human instructions when nothing is attached, and when the
        account's numbers cannot be listed -- an unverifiable number is refused
        rather than guessed at.
        """
        ...

    def applied_config(self) -> AppliedConfig:
        """The live-config receipt. Read-only snapshot, hashed."""
        ...

    def call_costs(self, since: float, until: float) -> list[CallCost]:
        """What this vendor billed for calls in a wall-clock window.

        Window bounds are unix seconds. Returns EVERY call the account saw in
        that window, not only ours -- matching to bench calls happens in
        harness/costs.py, which has our own timestamps to match against and can
        refuse an ambiguous pairing. An adapter that guessed here would have
        less information to guess with.

        Billing lags on some platforms, so an empty list is a legitimate
        "not yet", not an error; tools/backfill_costs.py exists to come back
        later. Raises VendorNotReady when the account cannot be read at all.
        """
        ...
