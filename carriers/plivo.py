"""Plivo: the carrier for the scripted-dialog bench.

The only carrier. The bench's caller side is Plivo XML (built in
harness/dialog.py -- <Record>/<GetInput>/<Speak>), so this
adapter is small: originate the call, authenticate webhooks, hang up, and find
the finished stereo recording.
"""

from __future__ import annotations

import logging

from fastapi import Request

from harness.config import settings
from harness.signature import verify_plivo_webhook

log = logging.getLogger(__name__)


class PlivoCarrier:
    name = "plivo"

    def _client(self):
        import plivo

        return plivo.RestClient(settings.plivo_auth_id, settings.plivo_auth_token)

    def place_call(self, to_number: str, *, answer_path: str = "/webhooks/answer",
                   status_path: str = "/webhooks/hangup") -> str:
        settings.require_carrier()
        base = settings.public_base_url.rstrip("/")
        response = self._client().calls.create(
            from_=settings.plivo_from_number,
            to_=to_number,
            answer_url=base + answer_path,
            answer_method="POST",
            hangup_url=base + status_path,
            hangup_method="POST",
        )
        # This is the REQUEST uuid, not the call. The CallUUID -- the id that
        # hangup and recording lookup need -- arrives in the answer webhook's
        # `CallUUID` param and is captured there.
        uuid = getattr(response, "request_uuid", None) or str(response)
        log.info("plivo call placed: %s", uuid)
        return uuid

    async def verify_webhook(self, request: Request) -> dict:
        return await verify_plivo_webhook(request)

    def hangup(self, call_uuid: str) -> None:
        """End one live call. Best-effort: by the time the bench calls this the
        vendor may already have hung up, and a 404 is success, not failure."""
        try:
            self._client().calls.delete(call_uuid)
        except Exception as exc:  # noqa: BLE001 -- cleanup is best-effort
            # "call not found" is the NORMAL outcome: the vendor ends the call
            # after its goodbye, so there is nothing left to hang up. Logged at
            # debug because at info it read as an error on every single call
            # and buried the failures that mattered.
            text = str(exc)
            if "not found" in text.lower() or "404" in text:
                log.debug("hangup %s: already ended", call_uuid)
            else:
                log.info("hangup %s: %s", call_uuid, text)

    def hangup_all(self) -> None:
        try:
            client = self._client()
            for call_id in client.live_calls.list_ids():
                client.calls.delete(call_id)
        except Exception as exc:  # noqa: BLE001 -- cleanup is best-effort
            log.info("hangup sweep: %s", exc)

    def recording_auth_headers(self) -> dict[str, str]:
        # Plivo recording URLs download without auth by default (accounts can
        # opt into basic auth; if this account ever does, return it here).
        return {}

    def find_wav_recording(self, call_uuid: str) -> str | None:
        """One non-blocking probe of List Recordings for this call.

        Returns the WAV download URL once the recording exists, else None.
        The retry loop lives in harness/dialog.py fetch_recording, same
        contract the Telnyx adapter had: this method must never block or raise.
        """
        try:
            response = self._client().recordings.list(call_uuid=call_uuid)
        except Exception as exc:  # noqa: BLE001 -- poll again later
            log.info("recordings list failed: %s", exc)
            return None
        objects = getattr(response, "objects", None) or []
        for rec in objects:
            url = _field(rec, "recording_url")
            fmt = (_field(rec, "recording_format") or "").lower()
            if not url:
                continue
            if fmt and fmt != "wav":
                log.info("recording for %s is %r, not wav -- check the answer "
                         "XML's fileFormat", call_uuid, fmt)
                continue
            return url
        return None


def _field(obj, name: str):
    """SDK resources expose attributes; raw JSON gives dicts. Accept both."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
