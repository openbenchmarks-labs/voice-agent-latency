"""PlivoCarrier against a faked SDK client.

The contract under test is the one harness/dialog.py's fetch_recording poll
depends on: find_wav_recording is one non-blocking probe that returns a URL or
None and never raises -- a carrier hiccup must read as "not ready yet", not as
a crashed call.
"""

from __future__ import annotations

import pytest

from carriers import get_carrier
from carriers.plivo import PlivoCarrier
from harness.config import settings


class FakeRecording:
    def __init__(self, url=None, fmt=None, call_uuid="CALL-1"):
        self.recording_url = url
        self.recording_format = fmt
        self.call_uuid = call_uuid


class FakeRecordings:
    def __init__(self, objects=None, boom=None):
        self._objects = objects or []
        self._boom = boom

    def list(self, **kwargs):
        if self._boom:
            raise self._boom
        self.last_kwargs = kwargs

        class Response:
            objects = self._objects

        return Response()


class FakeCalls:
    def __init__(self, boom=None):
        self.deleted = []
        self._boom = boom

    def create(self, **kwargs):
        self.create_kwargs = kwargs

        class Response:
            request_uuid = "REQ-123"

        return Response()

    def delete(self, call_uuid):
        if self._boom:
            raise self._boom
        self.deleted.append(call_uuid)


class FakeClient:
    def __init__(self, recordings=None, calls=None):
        self.recordings = recordings or FakeRecordings()
        self.calls = calls or FakeCalls()


def carrier_with(client: FakeClient) -> PlivoCarrier:
    carrier = PlivoCarrier()
    carrier._client = lambda: client  # type: ignore[method-assign]
    return carrier


# --------------------------------------------------------------------------- #
# find_wav_recording
# --------------------------------------------------------------------------- #


def test_finds_the_wav_url_for_this_call():
    client = FakeClient(recordings=FakeRecordings(
        objects=[FakeRecording(url="https://media.plivo.com/r1.wav", fmt="wav")]))
    carrier = carrier_with(client)
    assert carrier.find_wav_recording("CALL-1") == "https://media.plivo.com/r1.wav"
    assert client.recordings.last_kwargs == {"call_uuid": "CALL-1"}


def test_not_ready_yet_is_none_not_an_error():
    carrier = carrier_with(FakeClient(recordings=FakeRecordings(objects=[])))
    assert carrier.find_wav_recording("CALL-1") is None


def test_a_non_wav_recording_is_skipped_loudly_not_returned():
    """If the answer XML's fileFormat regresses to mp3, the analyzer must not
    be handed an mp3 -- the poll keeps returning None and the run fails with
    recording_not_found instead of a decode error mid-measurement."""
    carrier = carrier_with(FakeClient(recordings=FakeRecordings(
        objects=[FakeRecording(url="https://media.plivo.com/r1.mp3", fmt="mp3")])))
    assert carrier.find_wav_recording("CALL-1") is None


def test_dict_shaped_recordings_are_accepted_too():
    carrier = carrier_with(FakeClient(recordings=FakeRecordings(
        objects=[{"recording_url": "https://media.plivo.com/r2.wav",
                  "recording_format": "wav"}])))
    assert carrier.find_wav_recording("CALL-1") == "https://media.plivo.com/r2.wav"


def test_a_carrier_hiccup_reads_as_not_ready():
    carrier = carrier_with(FakeClient(
        recordings=FakeRecordings(boom=RuntimeError("api down"))))
    assert carrier.find_wav_recording("CALL-1") is None


# --------------------------------------------------------------------------- #
# hangup / place_call
# --------------------------------------------------------------------------- #


def test_hangup_deletes_the_call_uuid():
    client = FakeClient()
    carrier_with(client).hangup("CALL-9")
    assert client.calls.deleted == ["CALL-9"]


def test_hangup_of_an_already_dead_call_is_fine():
    carrier = carrier_with(FakeClient(calls=FakeCalls(boom=RuntimeError("404"))))
    carrier.hangup("CALL-9")  # must not raise


def test_place_call_passes_our_webhook_urls(monkeypatch):
    monkeypatch.setattr(settings, "plivo_auth_id", "MA_TEST")
    monkeypatch.setattr(settings, "plivo_auth_token", "tok")
    monkeypatch.setattr(settings, "plivo_from_number", "+15550009876")
    monkeypatch.setattr(settings, "public_base_url", "https://bench.example:8443")
    monkeypatch.setattr(settings, "carrier", "plivo")

    client = FakeClient()
    carrier = carrier_with(client)
    request_uuid = carrier.place_call("+15551234567")

    assert request_uuid == "REQ-123"
    kwargs = client.calls.create_kwargs
    assert kwargs["to_"] == "+15551234567"
    assert kwargs["from_"] == "+15550009876"
    assert kwargs["answer_url"] == "https://bench.example:8443/webhooks/answer"
    assert kwargs["hangup_url"] == "https://bench.example:8443/webhooks/hangup"
    assert kwargs["answer_method"] == kwargs["hangup_method"] == "POST"


def test_get_carrier_dispatches_plivo_and_rejects_the_rest(monkeypatch):
    assert isinstance(get_carrier("plivo"), PlivoCarrier)
    with pytest.raises(ValueError):
        get_carrier("telnyx")
    monkeypatch.setattr(settings, "carrier", "plivo")
    assert isinstance(get_carrier(), PlivoCarrier)
