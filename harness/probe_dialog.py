"""Probe P: one Plivo call that answers the scripted-dialog unknowns.

An earlier prototype proved the GetInput/Speak conversation loop against a live
vendor agent, but never exercised the measurement half. That is what this probe
pins down, on one cheap call:

  P1  Does <Record recordSession recordChannelType="stereo"> coexist with a
      GetInput/Speak dialog on the same leg?
  P2  Which stereo channel carries OUR Speak audio? (Undocumented. The answer
      becomes the pinned channel map in harness/dialog.py, citing this run.)
  P3  What does the recording callback POST actually contain, and does it fire
      at all? (Telnyx's equivalent never fired; polling is the fallback.)
  P4  What does the List Recordings API return (fields, latency after hangup),
      and does the recording URL download without auth?
  P5  What rate/container is the WAV? (The analyzer normalises to 8 kHz, but
      the raw facts belong in the provenance note.)
  P6  Do the V3 signatures on answer/action webhooks validate against our
      reconstruction (public_base_url + path, no query string)?

FINDINGS (probe-dialog-20260729-214809, 2026-07-30, outbound to a mobile):

  P1  Yes. `<Record recordSession recordChannelType="stereo">` runs happily
      alongside a GetInput/Speak dialog on the same leg; the whole 20.9 s call
      landed on one tape while three GetInput exchanges completed.

  P2  OUR LEG IS CHANNEL 1, the callee's is channel 0 -- the opposite of the
      intuitive order, and the reason harness/dialog.py pins it rather than
      assuming. Two independent signals agreed:
        ch1  5.89-8.11 s and 18.87-20.91 s -- exactly our two spoken lines
        ch0  2.07-2.42 s ("hello") and 9.85-14.88 s (the callee's answer)
        noise floor: ch1 -240.0 dBFS (digital silence: our speech is
        synthesised into the leg and never meets a microphone),
        ch0 -72.2 dBFS (real line noise)

  P3  The recording callback FIRES (Event=RecordStop), unlike the Telnyx
      equivalent, and arrives just after the hangup webhook. It carries the URL
      under BOTH `RecordUrl` and `RecordFile`, plus `RecordingID`,
      `RecordingDuration`, `RecordingDurationMs`, `RecordingStartMs` and
      `RecordingEndMs`. Polling remains the contract; this only skips the wait.

  P4  The recording URL downloads with NO authentication (HTTP 200 on a bare
      GET, 669 KB). `recording_auth_headers()` returning {} is correct for this
      account.

  P5  8000 Hz, 2 channels, PCM_16 in a WAV container -- already the analyzer's
      working rate, so the resample in fetch_recording is a no-op here.

  P6  V3 signatures validated on every inbound webhook (answer, both dialog
      actions, hangup, recording) against `public_base_url + request.path`.
      The reconstruction in harness/signature.py is right.

  Also worth knowing: the answer webhook's `CallUUID` equals the
  `request_uuid` returned by place_call on this account. Do not rely on that --
  the two are documented as different things and the code still reads CallUUID
  from the webhook.

Run it (the operator dials -- every call costs money):

    python -m harness.probe_dialog --to +15551234567 \
        --host 0.0.0.0 --port 8443 \
        --ssl-certfile ~/certs/fullchain.pem --ssl-keyfile ~/certs/privkey.pem

Requires CARRIER=plivo and the PLIVO_* settings in .env. The
probe is deliberately self-contained (its own XML, its own recording poll):
it documents raw carrier behavior, so it must not depend on the abstractions
whose design its findings decide.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request, Response

from .config import settings
from .serving import add_server_args, describe, server_kwargs, server_problems
from .signature import verify_plivo_webhook

QUESTION_LINE = "What are your Saturday opening hours?"
GOODBYE_LINE = "That's all I needed. Goodbye."

CALL_DEADLINE_S = 90.0
RECORDING_POLL_S = 120.0  # generous: the probe also measures readiness latency


# --------------------------------------------------------------------------- #
# Probe state
# --------------------------------------------------------------------------- #


@dataclass
class ProbeCall:
    out_dir: Path
    call_uuid: str | None = None
    recording_url: str | None = None
    recording_callback_at: float | None = None
    hangup_seen: threading.Event = field(default_factory=threading.Event)
    dialog_done: threading.Event = field(default_factory=threading.Event)
    hangup_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def event(self, name: str, **fields) -> None:
        record = {"event": name, "mono_ns": time.monotonic_ns(),
                  "wall": datetime.now(timezone.utc).isoformat(), **fields}
        with self._lock:
            with (self.out_dir / "events.jsonl").open("a") as fh:
                fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# XML -- inline on purpose (see module docstring)
# --------------------------------------------------------------------------- #


def _record_element(base_url: str) -> str:
    return (
        "<Record "
        'recordSession="true" '
        'recordChannelType="stereo" '
        'fileFormat="wav" '
        'playBeep="false" '
        'maxLength="300" '
        f"callbackUrl={quoteattr(base_url + '/webhooks/recording')} "
        'callbackMethod="POST" '
        'redirect="false"'
        "/>"
    )


def _getinput_element(action_url: str, prompt: str | None) -> str:
    # speechEndTimeout="auto" is what the prototype validated live; the probe
    # keeps it so the dialog behaves like the bench will.
    body = f"<Speak>{escape(prompt)}</Speak>" if prompt else ""
    return (
        "<GetInput "
        f"action={quoteattr(action_url)} "
        'method="POST" '
        'inputType="speech" '
        'language="en-US" '
        'speechEndTimeout="auto" '
        'executionTimeout="30"'
        f">{body}</GetInput>"
    )


def _response(*elements: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
            + "".join(elements) + "</Response>")


# --------------------------------------------------------------------------- #
# The app: answer -> greeting captured -> question -> reply captured -> bye
# --------------------------------------------------------------------------- #


async def _params_and_verdict(request: Request) -> tuple[dict, str]:
    """Parse the webhook and record whether its V3 signature validates (P6).

    The probe never rejects: a signature failure here is a finding, not an
    error, and rejecting would burn the call that produced it.
    """
    try:
        params = await verify_plivo_webhook(request)
        return params, "valid"
    except Exception as exc:  # noqa: BLE001 -- probe observes, never blocks
        form = await request.form()
        params = {key: str(value) for key, value in form.items()}
        return params, f"FAILED: {exc}"


def build_probe_app(call: ProbeCall) -> FastAPI:
    app = FastAPI()
    base = (settings.public_base_url or "").rstrip("/")

    @app.post("/webhooks/answer")
    async def answer(request: Request) -> Response:
        params, verdict = await _params_and_verdict(request)
        call.call_uuid = params.get("CallUUID")
        call.event("answered", signature=verdict, params=params)
        xml = _response(
            _record_element(base),
            # No prompt: Telnyx assistants greet immediately on connect
            # (prototype gotcha) -- listen first.
            _getinput_element(base + "/webhooks/dialog/greeting", None),
            f"<Speak>{escape(GOODBYE_LINE)}</Speak>",
            "<Hangup/>",
        )
        call.event("xml_served", step="answer", xml=xml)
        return Response(content=xml, media_type="application/xml")

    @app.post("/webhooks/dialog/{step}")
    async def dialog(step: str, request: Request) -> Response:
        params, verdict = await _params_and_verdict(request)
        call.event("dialog_action", step=step, signature=verdict, params=params)
        if step == "greeting":
            xml = _response(
                _getinput_element(base + "/webhooks/dialog/reply", QUESTION_LINE),
                f"<Speak>{escape(GOODBYE_LINE)}</Speak>",
                "<Hangup/>",
            )
        else:  # the vendor's answer to our question -- wrap up
            call.dialog_done.set()
            xml = _response(f"<Speak>{escape(GOODBYE_LINE)}</Speak>", "<Hangup/>")
        call.event("xml_served", step=step, xml=xml)
        return Response(content=xml, media_type="application/xml")

    @app.post("/webhooks/recording")
    async def recording(request: Request) -> dict:
        params, verdict = await _params_and_verdict(request)
        call.recording_url = (params.get("RecordUrl") or params.get("RecordFile")
                              or params.get("recording_url") or None)
        call.recording_callback_at = time.monotonic()
        call.event("recording_callback", signature=verdict, params=params)
        return {"ok": True}

    @app.post("/webhooks/hangup")
    async def hangup(request: Request) -> dict:
        params, verdict = await _params_and_verdict(request)
        call.hangup_at = time.monotonic()
        call.event("hangup_webhook", signature=verdict, params=params)
        call.hangup_seen.set()
        call.dialog_done.set()
        return {"ok": True}

    return app


# --------------------------------------------------------------------------- #
# Recording retrieval -- raw API, dumped verbatim (P3/P4)
# --------------------------------------------------------------------------- #


def _api_auth() -> tuple[str, str]:
    return settings.plivo_auth_id, settings.plivo_auth_token


def poll_recordings(call: ProbeCall) -> str | None:
    """Poll the raw List Recordings endpoint and dump what it returns."""
    auth_id, _ = _api_auth()
    url = f"https://api.plivo.com/v1/Account/{auth_id}/Recording/"
    deadline = time.monotonic() + RECORDING_POLL_S
    attempt = 0
    with httpx.Client(timeout=30.0, auth=_api_auth()) as client:
        while time.monotonic() < deadline:
            attempt += 1
            response = client.get(url, params={"call_uuid": call.call_uuid})
            body: object
            try:
                body = response.json()
            except ValueError:
                body = response.text[:2000]
            call.event("recordings_list", attempt=attempt,
                       status=response.status_code, body=body)
            if isinstance(body, dict):
                for obj in body.get("objects", []):
                    rec_url = obj.get("recording_url")
                    if rec_url:
                        return rec_url
            time.sleep(3.0)
    return None


def download_recording(call: ProbeCall, url: str) -> Path | None:
    """Download twice: unauthenticated first (P4), basic auth as fallback."""
    raw_path = call.out_dir / "recording_raw.wav"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        call.event("recording_download", auth="none",
                   status=response.status_code, bytes=len(response.content))
        if response.status_code != 200:
            credentials = base64.b64encode(
                ("%s:%s" % _api_auth()).encode()).decode()
            response = client.get(
                url, headers={"Authorization": f"Basic {credentials}"})
            call.event("recording_download", auth="basic",
                       status=response.status_code, bytes=len(response.content))
        if response.status_code != 200:
            return None
        raw_path.write_bytes(response.content)
    return raw_path


# --------------------------------------------------------------------------- #
# Channel inspection (P2/P5)
# --------------------------------------------------------------------------- #


def print_channel_report(call: ProbeCall, wav: Path) -> None:
    audio, rate = sf.read(wav, dtype="int16", always_2d=True)
    info = sf.info(wav)
    duration = audio.shape[0] / rate
    print(f"\nrecording: rate={rate} Hz  channels={audio.shape[1]}  "
          f"duration={duration:.1f}s  subtype={info.subtype}  format={info.format}")
    call.event("recording_facts", rate=rate, channels=int(audio.shape[1]),
               duration_s=round(duration, 2), subtype=info.subtype)

    print("\nper-channel RMS by second -- we speak exactly twice "
          f"({QUESTION_LINE!r}, then {GOODBYE_LINE!r}); everything else is the vendor")
    header = "  sec  " + "".join(f"     ch{c}" for c in range(audio.shape[1]))
    print(header)
    rows = []
    for sec in range(int(np.ceil(duration))):
        window = audio[sec * rate:(sec + 1) * rate].astype(np.float64)
        rms = [float(np.sqrt(np.mean(window[:, c] ** 2)) if window.size else 0.0)
               for c in range(audio.shape[1])]
        rows.append({"sec": sec, "rms": [round(v, 1) for v in rms]})
        print(f"  {sec:3d}  " + "".join(f"{v:8.1f}" for v in rms))
    call.event("channel_rms", rows=rows)
    print(
        "\nP2: the channel whose bursts line up with the xml_served events for "
        "the question/goodbye is OURS (near). Cross-check by listening to "
        f"{wav}."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="number to dial, E.164")
    parser.add_argument("--port", type=int, default=8000)
    add_server_args(parser)
    args = parser.parse_args()

    problems = server_problems(args)
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        return 2
    if settings.carrier != "plivo":
        print(f"PROBLEM: CARRIER={settings.carrier!r}; this probe is Plivo-only",
              file=sys.stderr)
        return 2
    settings.require_carrier()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = settings.runs_dir / f"probe-dialog-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    call = ProbeCall(out_dir=out_dir)
    print(f"run dir: {out_dir}")
    print(describe(args))

    app = build_probe_app(call)
    server = uvicorn.Server(uvicorn.Config(app, **server_kwargs(args)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.5)

    from carriers import get_carrier

    carrier = get_carrier("plivo")
    request_uuid = carrier.place_call(args.to)
    call.event("placed", request_uuid=request_uuid, to=args.to)
    print(f"placed: request_uuid={request_uuid} -- talking to {args.to}")

    deadline = time.monotonic() + CALL_DEADLINE_S
    while time.monotonic() < deadline:
        if call.hangup_seen.wait(timeout=1.0):
            break
    else:
        print("deadline reached without a hangup webhook")
        call.event("deadline_reached")
    try:
        carrier.hangup_all()
    except Exception:  # noqa: BLE001 -- cleanup only
        pass

    if not call.call_uuid:
        print("no answer webhook arrived -- nothing to retrieve; see events.jsonl")
        return 1

    # P3: did the callback beat the poll?
    if call.recording_url:
        lag = None
        if call.recording_callback_at and call.hangup_at:
            lag = call.recording_callback_at - call.hangup_at
        print(f"recording callback fired (lag after hangup: "
              f"{lag:.1f}s)" if lag is not None else "recording callback fired")
        url = call.recording_url
    else:
        print("no recording callback yet -- polling List Recordings (P3/P4)")
        url = poll_recordings(call)
    if not url:
        print("no recording found; see events.jsonl for the raw API responses")
        return 1

    wav = download_recording(call, url)
    if not wav:
        print("recording download failed; see events.jsonl")
        return 1
    print_channel_report(call, wav)
    print(f"\nevents: {out_dir / 'events.jsonl'}")
    print("Fill in the FINDINGS block in this module's docstring from the above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
