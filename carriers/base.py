"""The carrier interface, sized for the scripted-dialog bench.

A carrier is "place a call, authenticate a webhook, hang up, and find the
finished recording". The conversation itself is carrier XML served by
harness/dialog.py, and every measurement comes from the stereo call recording
-- there is no media streaming, so there is no media vocabulary here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fastapi import Request


class Carrier(Protocol):
    name: str

    def place_call(self, to_number: str, *, answer_path: str = "/webhooks/answer",
                   status_path: str = "/webhooks/hangup") -> str:
        """Dial out. Returns the carrier's REQUEST id -- the call id proper
        arrives in the answer webhook (CallUUID for Plivo)."""
        ...

    async def verify_webhook(self, request: "Request") -> dict:
        """Authenticate an inbound webhook; return its params. Raises 403."""
        ...

    def hangup(self, call_uuid: str) -> None:
        """End one live call, best-effort."""
        ...

    def hangup_all(self) -> None:
        """Best-effort: end every live call on this account (probe cleanup)."""
        ...

    def recording_auth_headers(self) -> dict[str, str]:
        """Headers needed to download a recording URL, if any."""
        ...

    def find_wav_recording(self, call_uuid: str) -> str | None:
        """Download URL of the completed stereo WAV for this call, or None if
        not ready yet. One non-blocking probe; the retry loop is the caller's.
        Polled because recording-ready webhooks have a record of silently not
        firing (Telnyx TeXML never fired one; Plivo's callbackUrl is treated
        as a fast path only)."""
        ...


def get_carrier(name: str | None = None) -> Carrier:
    """Resolve the configured carrier. The only place concrete classes appear."""
    from harness.config import settings

    chosen = name or settings.carrier
    if chosen == "plivo":
        from .plivo import PlivoCarrier

        return PlivoCarrier()
    raise ValueError(f"unknown carrier {chosen!r} -- the bench runs on plivo")
