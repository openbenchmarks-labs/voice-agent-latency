"""The caller side of a bench call: a scripted Plivo dialog, and its receipt.

One call is a fixed sequence of caller utterances, each spoken by Plivo TTS
inside a `<GetInput>` that then listens for the vendor's reply:

    answer  ── <Record stereo> + GetInput(no prompt)   listen for the greeting
            ── GetInput("Hi there.")                   turn 1
            ── GetInput("<case question>")             turns 2..n-1
            ── GetInput("That's all I needed...")      turn n
            ── <Hangup/>

Every caller utterance is a measured turn: t1 is where that utterance ends on
the near channel of the recording, t2 is where the vendor's reply begins on the
far channel. Nothing timed here is a measurement -- webhook timestamps include
TTS playout, network, and Plivo's endpointing -- exactly the error that makes
webhook-derived latency unusable. This module drives the conversation; the
analyzer measures it.

The script comes from data/voice-bench/ttfab_scenarios.json -- one source of
truth for the caller's lines, the vendor's system prompt, and the keywords each
answer must contain -- and the dialog knobs from config/dialog.yaml. Both are
hashed into a caller receipt, because with a live caller the stimulus is no
longer a committed WAV: the caller is part of the instrument now.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import yaml

from analyzer.resample import resample_int16

from .answerxml import (
    build_getinput,
    build_hangup,
    build_record_element,
    build_redirect,
    build_response,
)
from .config import data_root, settings

log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = data_root() / "data" / "voice-bench" / "ttfab_scenarios.json"
DIALOG_CONFIG_PATH = _PKG_ROOT / "config" / "dialog.yaml"

ANALYZER_RATE = 8000
RECORDING_POLL_S = 90.0
RECORDING_POLL_INTERVAL_S = 3.0

# Wall-clock backstop for one call: greeting listen + n turns, each bounded by
# execution_timeout, plus room for TTS playout and the hangup round trip.
CALL_DEADLINE_S = 180.0

# How long to wait for the vendor to pick up before giving up on the call.
# Separate from CALL_DEADLINE_S, which bounds a conversation once it starts: a
# call that is never answered has nothing to wait for, and waiting the full
# deadline for it cost three minutes per unanswered call in
# bench-telnyx-20260730-111843.
ANSWER_TIMEOUT_S = 30.0

GREETING_TIMEOUT_S = 6.0
MIN_GREETING_TIMEOUT_S = 4.0

# --------------------------------------------------------------------------- #
# Which stereo channel is ours
# --------------------------------------------------------------------------- #
#
# An earlier design re-derived this per call by correlating our known playback
# against both channels. With a live caller there is no known waveform, so the
# channel map is pinned here instead -- and pinning demands
# provenance, because a silent inversion would swap every t1 and t2 and still
# produce plausible-looking numbers.
#
# Plivo does not document the channel order, and it is NOT the intuitive one:
# measured on probe-dialog-20260729-214809 (2026-07-30), our own leg is
# channel 1 and the callee's is channel 0. Two independent signals agreed on
# that call:
#
#   - utterance layout: ch1 carried exactly our two spoken lines (5.89-8.11 s,
#     18.87-20.91 s) while ch0 carried the callee's "hello" and their answer
#   - noise floor: ch1 sat at -240 dBFS (digital silence -- our speech is
#     synthesised into the leg and never touches a microphone), ch0 at
#     -72.2 dBFS (real line noise)
#
# The analyzer does not trust this constant either way: it counts near-channel
# utterance groups against the number of turns we actually spoke and discards
# the call as channel_map_suspect if the other channel fits better
# (analyzer/measure.py, measure_dialog). That check is what makes a future
# change in Plivo's behaviour loud instead of silent -- an inverted map would
# otherwise measure their speech-end against our reply and still produce
# plausible numbers.
PLIVO_CHANNEL_MAP = {"near": 1, "far": 0}
CHANNEL_MAP_SOURCE = "probe-dialog-20260729-214809 (2026-07-30)"


def channel_map() -> dict:
    return {**PLIVO_CHANNEL_MAP, "source": CHANNEL_MAP_SOURCE}


def channel_map_is_provisional() -> bool:
    return CHANNEL_MAP_SOURCE.startswith("provisional")


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Turn:
    """One caller utterance, and what the reply to it should contain."""

    index: int                      # 1-based; matches the analyzer's turn_index
    kind: str                       # t1_greeting | case | t3_goodbye
    text: str
    case_id: str | None = None
    expect_keywords: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"index": self.index, "kind": self.kind, "text": self.text,
                "case_id": self.case_id,
                "expect_keywords": list(self.expect_keywords)}


@dataclass(frozen=True)
class CallerScript:
    turns: tuple[Turn, ...]
    voice: str
    language: str
    speech_end_timeout: int | str
    execution_timeout: int
    greeting_timeout_s: float
    system_prompt: str              # what the vendor must be running
    scenarios_version: str

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def turn(self, index: int) -> Turn:
        return self.turns[index - 1]

    def receipt(self) -> dict:
        """The caller half of the run's provenance, hashed like a vendor's."""
        body = {
            "turns": [t.as_dict() for t in self.turns],
            "voice": self.voice,
            "language": self.language,
            "speech_end_timeout": self.speech_end_timeout,
            "execution_timeout": self.execution_timeout,
            "greeting_timeout_s": self.greeting_timeout_s,
            "scenarios_version": self.scenarios_version,
            "tts": "plivo-speak",
        }
        canonical = json.dumps(body, sort_keys=True, default=str)
        return {**body, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def load_scenarios(path: Path | None = None) -> dict:
    return json.loads((path or SCENARIOS_PATH).read_text())


def load_dialog_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DIALOG_CONFIG_PATH).read_text()) or {}


def greeting_timeout_for(idle_reply_secs: float | None,
                         ceiling: float = GREETING_TIMEOUT_S) -> float:
    """How long to wait for the vendor's greeting before speaking anyway.

    Deliberately NOT scaled by the vendor's idle-reprompt setting, which is
    what this used to do (60% of it). That heuristic came from an earlier
    design -- where we had to get a word in before the vendor re-prompted --
    and it is the wrong question here.

    Measured 2026-07-30 across bench-telnyx-20260730-{114637,115245}: scaling
    by a 10 s idle setting gave a 6 s window, and the agent's greeting lands at
    9-10 s. We therefore started talking before it had greeted on SIX of SEVEN
    calls. The one call where it greeted first (1.4 s) was flawless; the rest
    degraded or collapsed -- the agent's endpointer never saw a clean turn
    boundary, buffered our questions, and answered them in batches. The earlier
    run of "dead air" calls was the same cause with the old hangup fallback.

    Waiting longer is nearly free: this timeout only fires when the greeting
    has NOT arrived, and a vendor that greets returns the listener the moment
    it finishes. The cost of waiting too long is hearing an idle prompt, which
    the analyzer flags; the cost of waiting too little is the whole call.
    """
    return max(MIN_GREETING_TIMEOUT_S, ceiling)


def build_script(case_ids: list[str] | None = None, *,
                 idle_reply_secs: float | None = None,
                 scenarios_path: Path | None = None,
                 config_path: Path | None = None) -> CallerScript:
    """T1 + the requested cases + T3, as the turn list the dialog will speak.

    T1 and T3 are switchable in config/dialog.yaml, but think before dropping
    T1. It is the turn that absorbs first-response cost -- cold start, first
    inference, connection warmup -- and it is reliably the slowest turn of a
    call. Remove it and that cost does not disappear: it moves onto your first
    case question, which then reads slower than the same question asked second,
    for a reason that has nothing to do with the question.
    """
    scenarios = load_scenarios(scenarios_path)
    config = load_dialog_config(config_path)

    structure = scenarios.get("conversation_structure", {})
    t1_line = structure.get("turn_1", {}).get("caller_says", "Hi there.")
    t3_line = structure.get("turn_3", {}).get("caller_says",
                                              "That's all I needed. Goodbye.")

    by_id = {case["id"]: case for case in scenarios.get("cases", [])}
    wanted = list(case_ids or config.get("default_cases") or [])
    unknown = [cid for cid in wanted if cid not in by_id]
    if unknown:
        raise KeyError(f"unknown case id(s) {unknown} -- "
                       f"{(scenarios_path or SCENARIOS_PATH)} has {sorted(by_id)}")
    if not wanted:
        raise KeyError("no cases requested and config/dialog.yaml has no "
                       "default_cases -- a call with no questions measures nothing")

    turns: list[Turn] = []
    if config.get("include_greeting_turn", True):
        turns.append(Turn(index=1, kind="t1_greeting", text=t1_line))
    for case_id in wanted:
        case = by_id[case_id]
        turns.append(Turn(
            index=len(turns) + 1,
            kind="case",
            text=case["question"],
            case_id=case_id,
            expect_keywords=tuple(case.get("expect_keywords", ())),
        ))
    if config.get("include_goodbye_turn", True):
        turns.append(Turn(index=len(turns) + 1, kind="t3_goodbye", text=t3_line))

    return CallerScript(
        turns=tuple(turns),
        voice=config.get("voice", "Polly.Joanna"),
        language=config.get("language", "en-US"),
        speech_end_timeout=config.get("speech_end_timeout", "auto"),
        execution_timeout=int(config.get("execution_timeout", 30)),
        greeting_timeout_s=greeting_timeout_for(
            idle_reply_secs, ceiling=float(config.get("greeting_timeout",
                                                      GREETING_TIMEOUT_S))),
        system_prompt=scenarios.get("system_prompt", ""),
        scenarios_version=scenarios.get("benchmark", "voice-ttfab"),
    )


def case_rotation(config: dict | None = None,
                  explicit: list[str] | None = None) -> list[list[str]]:
    """The pinned question sets, one per call slot.

    `explicit` (--cases) collapses the rotation to a single set, so every call
    asks the same thing -- which is what you want when debugging one question,
    and not what you want when measuring.
    """
    if explicit:
        return [list(explicit)]
    config = config if config is not None else load_dialog_config()
    rotation = config.get("case_rotation")
    if rotation:
        return [list(entry) for entry in rotation]
    return [list(config.get("default_cases") or [])]


def build_run_scripts(n_calls: int, case_ids: list[str] | None = None, *,
                      idle_reply_secs: float | None = None,
                      scenarios_path: Path | None = None,
                      config_path: Path | None = None) -> list[CallerScript]:
    """One script per call, cycling the rotation.

    Rotating rather than sampling. Both give variation across calls, but a
    rotation also gives the SAME variation to every vendor: call 3 asks the
    same three questions whoever is being measured, and asks them again on a
    re-run. That matters because answers differ in length, and a platform that
    buffers its TTS starts speaking later on longer ones -- so a vendor that
    happened to draw short questions would look faster for a reason that has
    nothing to do with the platform.
    """
    config = load_dialog_config(config_path)
    rotation = case_rotation(config, case_ids)
    return [
        build_script(rotation[i % len(rotation)],
                     idle_reply_secs=idle_reply_secs,
                     scenarios_path=scenarios_path, config_path=config_path)
        for i in range(max(1, n_calls))
    ]


def run_plan_receipt(scripts: list[CallerScript]) -> dict:
    """The caller half of the run's provenance when calls differ.

    A single script's receipt cannot describe a rotating run, so the run-level
    receipt records the plan -- what every call slot asked, and the hash of
    each -- while each call's metadata still carries its own script hash.
    """
    first = scripts[0]
    body = {
        "voice": first.voice,
        "language": first.language,
        "speech_end_timeout": first.speech_end_timeout,
        "execution_timeout": first.execution_timeout,
        "greeting_timeout_s": first.greeting_timeout_s,
        "scenarios_version": first.scenarios_version,
        "tts": "plivo-speak",
        "turn_kinds": [t.kind for t in first.turns],
        "rotation_length": len({s.receipt()["sha256"] for s in scripts}),
        "calls": [
            {
                "call_index": i,
                "cases": [t.case_id for t in s.turns if t.case_id],
                "sha256": s.receipt()["sha256"],
            }
            for i, s in enumerate(scripts)
        ],
    }
    canonical = json.dumps(body, sort_keys=True, default=str)
    return {**body, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def answer_matches(transcript: str, keywords: tuple[str, ...]) -> bool | None:
    """Did the vendor's reply contain every keyword the case requires?

    None when there is nothing to check (the T1/T3 turns, or a case with no
    keywords). Accuracy is reported alongside latency, never mixed into it: a
    wrong-but-fast answer is a different failure from a slow one.
    """
    if not keywords:
        return None
    haystack = transcript.lower()
    return all(re.search(re.escape(word.lower()), haystack) for word in keywords)


# --------------------------------------------------------------------------- #
# Per-call state
# --------------------------------------------------------------------------- #

STATE_DIALING = "DIALING"
STATE_AWAIT_GREETING = "AWAIT_GREETING"
STATE_COMPLETE = "COMPLETE"


def turn_state(index: int) -> str:
    return f"TURN_{index}"


@dataclass
class DialogSession:
    """One call's conversation state. Webhooks mutate it; the bench reads it."""

    call_id: str
    out_dir: Path
    script: CallerScript
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    state: str = STATE_DIALING
    call_sid: str | None = None          # Plivo request_uuid, from place_call
    call_control_id: str | None = None   # Plivo CallUUID, from the answer webhook
    greeting_transcript: str = ""
    greeting_timed_out: bool = False
    turns_spoken: int = 0
    actions_handled: int = 0
    replies: dict[int, dict] = field(default_factory=dict)
    recording_url: str | None = None

    answered: threading.Event = field(default_factory=threading.Event)
    hangup_seen: threading.Event = field(default_factory=threading.Event)
    dialog_done: threading.Event = field(default_factory=threading.Event)

    _served: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ----------------------------------------------------------------- events

    def event(self, name: str, **kw) -> None:
        with self._lock:
            with (self.out_dir / "events.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "mono_ns": time.monotonic_ns(), "wall": time.time(),
                    "event": name, "state": self.state, **kw,
                }) + "\n")

    # -------------------------------------------------------------------- xml

    def _base(self) -> str:
        return (settings.public_base_url or "").rstrip("/")

    def _action_url(self, step: str) -> str:
        return f"{self._base()}/webhooks/dialog/{self.token}/{step}"

    def _listen(self, step: str, prompt: str | None, *,
                execution_timeout: int | None = None) -> str:
        return build_getinput(
            action_url=self._action_url(step),
            prompt_text=prompt,
            voice=self.script.voice,
            language=self.script.language,
            speech_end_timeout=self.script.speech_end_timeout,
            execution_timeout=execution_timeout or self.script.execution_timeout,
        )

    def answer_xml(self, params: dict | None = None) -> str:
        """Served when the vendor answers: arm the tape, then listen.

        The first GetInput speaks nothing. Vendors greet the moment they connect
        (prototype finding), so speaking here would talk over the greeting and
        cost turn 1.
        """
        params = params or {}
        self.call_control_id = params.get("CallUUID") or self.call_control_id
        self.state = STATE_AWAIT_GREETING
        self.event("answered", call_uuid=self.call_control_id,
                   from_=params.get("From"), to=params.get("To"))
        self.answered.set()
        xml = build_response(
            build_record_element(callback_url=self._base() + "/webhooks/recording",
                                 max_seconds=int(CALL_DEADLINE_S)),
            self._listen("greeting", None,
                         execution_timeout=int(self.script.greeting_timeout_s)),
            # A GetInput that times out with NO speech does not POST its action
            # -- it falls through to here. Redirect back into the dialog so a
            # vendor that never greets costs us a greeting, not the call.
            build_redirect(self._action_url("greeting")),
        )
        self._served["answer"] = xml
        self.event("xml_served", step="answer")
        return xml

    # ------------------------------------------------------------- transitions

    def handle_action(self, step: str, params: dict) -> str:
        """A GetInput completed: log what the vendor said, speak the next line.

        `step` is "greeting", or the index of the turn whose reply this POST
        carries. Idempotent per step: Plivo retries and duplicate deliveries
        must not skip a turn, so a repeated step replays its recorded XML.
        """
        transcript = (params.get("Speech") or params.get("speech") or "").strip()
        confidence = params.get("SpeechConfidenceScore")

        # The dialog is driven by webhooks we do not control the delivery of,
        # and every GetInput now redirects back here on timeout. A retry storm
        # or a redirect loop would otherwise keep a call alive burning money
        # for as long as the carrier allowed.
        self.actions_handled += 1
        if self.actions_handled > 3 * self.script.n_turns + 6:
            self.event("action_storm", step=step, count=self.actions_handled)
            self.state = STATE_COMPLETE
            self.dialog_done.set()
            return build_response(build_hangup())

        if step in self._served:
            self.event("action_replayed", step=step, transcript=transcript)
            return self._served[step]

        index = self._turn_index(step)
        if step != "greeting" and index is None:
            # The dialog webhook is publicly reachable, so a step we did not
            # issue must end the call rather than raise -- a 500 here would
            # strand a live call mid-conversation.
            self.event("action_unknown_step", step=step)
            return build_response(build_hangup())

        if step == "greeting":
            xml = self._after_greeting(transcript, confidence)
        else:
            xml = self._after_reply(index, transcript, confidence)

        self._served[step] = xml
        self.event("xml_served", step=step)
        return xml

    def _turn_index(self, step: str) -> int | None:
        """The turn `step` refers to, or None if it names no turn of ours."""
        try:
            index = int(step)
        except (TypeError, ValueError):
            return None
        return index if 1 <= index <= self.script.n_turns else None

    def _after_greeting(self, transcript: str, confidence) -> str:
        self.greeting_transcript = transcript
        if transcript:
            self.event("greeting_heard", transcript=transcript,
                       confidence=confidence)
        else:
            # The listen window expired with nothing on it. Proceed anyway: the
            # tape is the record of what happened, and a vendor that never
            # greeted is a finding, not a reason to abandon the call.
            self.greeting_timed_out = True
            self.event("greeting_timeout")
        return self._speak_turn(1)

    def _after_reply(self, index: int, transcript: str, confidence) -> str:
        turn = self.script.turn(index)
        verified = answer_matches(transcript, turn.expect_keywords)
        self.replies[index] = {
            "transcript": transcript,
            "confidence": confidence,
            "answer_verified": verified,
            "timed_out": not transcript,
        }
        if transcript:
            self.event("vendor_speech", turn=index, kind=turn.kind,
                       case=turn.case_id, transcript=transcript,
                       confidence=confidence, answer_verified=verified)
        else:
            # No reply inside the window. The turn stays in the tape and the
            # analyzer discards it as no_response -- a discard is a result.
            self.event("reply_timeout", turn=index, kind=turn.kind,
                       case=turn.case_id)

        if index < self.script.n_turns:
            return self._speak_turn(index + 1)

        self.state = STATE_COMPLETE
        self.event("dialog_complete", turns_spoken=self.turns_spoken)
        self.dialog_done.set()
        return build_response(build_hangup())

    def _speak_turn(self, index: int) -> str:
        turn = self.script.turn(index)
        self.state = turn_state(index)
        self.turns_spoken = max(self.turns_spoken, index)
        self.event("turn_prompt_served", turn=index, kind=turn.kind,
                   case=turn.case_id, text=turn.text)
        return build_response(
            self._listen(str(index), turn.text),
            # Same fallthrough: a turn the vendor never answers redirects back
            # here rather than ending the call, so the remaining questions still
            # get asked and this turn is measured as no_response.
            build_redirect(self._action_url(str(index))),
        )

    # ------------------------------------------------------------------ status

    def note_hangup(self, params: dict | None = None) -> None:
        self.event("hangup_webhook", **(params or {}))
        self.hangup_seen.set()
        self.dialog_done.set()

    def note_recording_callback(self, params: dict) -> None:
        # Plivo sends the same URL under both RecordUrl and RecordFile
        # (probe-dialog-20260729-214809); accept either so one of them going
        # away is not a silent loss of the fast path.
        self.recording_url = (params.get("RecordUrl") or params.get("RecordFile")
                              or params.get("recording_url") or None)
        self.event("recording_callback", url=self.recording_url,
                   recording_id=params.get("RecordingID"),
                   duration=params.get("RecordingDuration"))

    def turn_metadata(self) -> list[dict]:
        """Per-turn record for metadata.json -- what we said, what came back."""
        rows = []
        for turn in self.script.turns:
            reply = self.replies.get(turn.index, {})
            rows.append({
                **turn.as_dict(),
                "spoken": turn.index <= self.turns_spoken,
                "transcript": reply.get("transcript", ""),
                "answer_verified": reply.get("answer_verified"),
                "reply_timed_out": reply.get("timed_out"),
            })
        return rows


# --------------------------------------------------------------------------- #
# Post-call: get the tape
# --------------------------------------------------------------------------- #


def fetch_recording(call: DialogSession, carrier) -> Path | None:
    """Wait for the stereo WAV, download it, normalise to 8 kHz.

    The recording callback is a fast path only. Its Telnyx equivalent never
    fired at all, so polling is the contract and the callback merely skips the
    wait when it works.
    """
    if not call.call_control_id:
        call.event("recording_skipped", reason="no CallUUID from the answer webhook")
        return None

    url = call.recording_url
    if not url:
        deadline = time.monotonic() + RECORDING_POLL_S
        while time.monotonic() < deadline:
            url = call.recording_url or carrier.find_wav_recording(call.call_control_id)
            if url:
                break
            time.sleep(RECORDING_POLL_INTERVAL_S)
    if not url:
        call.event("recording_not_found")
        return None

    raw_path = call.out_dir / "recording_raw.wav"
    headers = carrier.recording_auth_headers() or {}
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        raw_path.write_bytes(response.content)

    audio, rate = sf.read(raw_path, dtype="int16", always_2d=True)
    call.event("recording_downloaded", rate=rate, samples=int(audio.shape[0]),
               channels=int(audio.shape[1]))
    if audio.shape[1] < 2:
        # Mono means the answer XML lost recordChannelType, or the recording
        # came from the API instead. Either way there is no near/far to
        # separate and the call cannot be measured.
        call.event("recording_not_stereo", channels=int(audio.shape[1]))

    if rate != ANALYZER_RATE:
        audio = np.stack(
            [resample_int16(np.ascontiguousarray(audio[:, c]), rate, ANALYZER_RATE)
             for c in range(audio.shape[1])],
            axis=1,
        )
    out = call.out_dir / "recording.wav"
    sf.write(out, audio, ANALYZER_RATE, subtype="PCM_16")
    return out
