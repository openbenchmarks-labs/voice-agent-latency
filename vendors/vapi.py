"""Vapi assistant as a vendor under test.

Vapi is the orchestrator case, the structural opposite of Telnyx: it owns
turn-taking and the speaking plans but rents STT, LLM and TTS from whoever the
assistant names (Deepgram, OpenAI, ElevenLabs, ...). Both are measured the same
way here -- as shipped, whatever stack the platform chose for a new signup --
which is the point of the closed division.

READ-ONLY by contract. This module issues GETs and nothing else. `verify_agent`
reports mismatches for a human to fix in the dashboard; it never repairs them.

Field mapping (from the live OpenAPI spec, https://api.vapi.ai/api-json,
fetched 2026-07-30):
  GET /assistant/{id}        -> full config
    firstMessage             the greeting (Telnyx: `greeting`)
    firstMessageMode         enum: assistant-speaks-first |
                             assistant-speaks-first-with-model-generated-message |
                             assistant-waits-for-user
    model.messages[role=system].content
                             the system prompt (Telnyx: `instructions`)
    model.provider/.model/.temperature
    transcriber.provider/.model     voice.provider/.voiceId
    startSpeakingPlan / stopSpeakingPlan   the endpointing knobs
  GET /phone-number          -> numbers; `assistantId` is what makes a number
                                reach this assistant on inbound PSTN

Two differences from Telnyx worth their place in the receipt:
  * Vapi DOES expose `model.temperature`, so the reply-determinism request is
    satisfiable here and is not satisfiable on Telnyx. The receipt records the
    asymmetry rather than footnoting it.
  * Vapi's assistant schema has no idle-message plan, so there is no
    vendor-side idle threshold to record -- an `idle_filler` discard on this
    vendor cannot be explained by a configured idle reply.
"""

from __future__ import annotations

import logging

import httpx

from harness.config import settings

from .base import (AgentSpec, AppliedConfig, CallCost, DialTarget, digits,
                    iso_to_epoch)

log = logging.getLogger(__name__)

API = "https://api.vapi.ai"

# The only mode that gives live VAD something to wait for. The
# model-generated variant would make the greeting differ call to call, which
# would make the greeting-match gate meaningless and TTFG unattributable.
REQUIRED_FIRST_MESSAGE_MODE = "assistant-speaks-first"

# Vapi returns startSpeakingPlan/stopSpeakingPlan as null when the assistant
# never overrode them, but the platform still applies these values server-side
# (docs.vapi.ai/customization/voice-pipeline-configuration, read 2026-07-30).
# Recording null would describe the run as "endpointing unknown", which is worse
# than wrong -- these are the knobs that set TTFAB. Recording the documented
# number silently would pass an assumption off as an observation. So the receipt
# carries the value AND where it came from (`endpointing_source`).
DOCUMENTED_ENDPOINTING_DEFAULTS = {
    "wait_seconds": 0.4,
    "on_punctuation_seconds": 0.1,
    "on_no_punctuation_seconds": 1.5,
    "on_number_seconds": 0.5,
    "stop_num_words": 0,
    "stop_voice_seconds": 0.2,
    "stop_backoff_seconds": 1.0,
}


class VendorNotReady(RuntimeError):
    """Raised with human-actionable instructions, never auto-repaired."""


def _slash(provider: str | None, model: str | None) -> str | None:
    """provider + model as one comparable string, the shape Telnyx reports."""
    if not provider and not model:
        return None
    return "/".join(part for part in (provider, model) if part)


def _matches(live: str | None, want: str) -> bool:
    """A pinned value matches either `provider/model` or the bare model name."""
    if live is None:
        return False
    return live == want or live.split("/")[-1] == want.split("/")[-1]


class VapiVendor:
    name = "vapi"

    def __init__(self, block: dict | None = None):
        self.block = block or {}
        self.assistant_id = (
            self.block.get("assistant_id") or settings.vapi_assistant_id
        )
        self._assistant: dict | None = None

    # ------------------------------------------------------------------ fetch

    def _client(self) -> httpx.Client:
        if not settings.vapi_api_key:
            raise VendorNotReady("VAPI_API_KEY is not set in .env")
        return httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
            timeout=15.0,
        )

    @staticmethod
    def _items(payload) -> list[dict]:
        """Vapi list endpoints return a bare array; tolerate {"data": [...]}."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("data") or payload.get("results") or []
        return []

    def assistant(self, refresh: bool = False) -> dict:
        """The live assistant config. Cached per adapter instance."""
        if self._assistant is not None and not refresh:
            return self._assistant

        with self._client() as client:
            if self.assistant_id:
                response = client.get(f"/assistant/{self.assistant_id}")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot read assistant {self.assistant_id}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                payload = response.json()
                data = payload.get("data", payload) if isinstance(payload, dict) else payload
            else:
                response = client.get("/assistant")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot list assistants: {response.status_code} "
                        f"{response.text[:200]}"
                    )
                assistants = self._items(response.json())
                if not assistants:
                    raise VendorNotReady(
                        "no assistants on this Vapi account -- create one in the "
                        "dashboard (Assistants > Create), give it the Northwind "
                        "prompt and greeting from "
                        "data/voice-bench/ttfab_scenarios.json, then set "
                        "VAPI_ASSISTANT_ID in .env"
                    )
                if len(assistants) > 1:
                    names = ", ".join(
                        f"{a.get('name')}={a.get('id')}" for a in assistants
                    )
                    raise VendorNotReady(
                        "multiple assistants found; pin one with VAPI_ASSISTANT_ID "
                        f"in .env or assistant_id in config/vendors.yaml. Found: {names}"
                    )
                data = assistants[0]

        self._assistant = data
        return data

    # -------------------------------------------------------------- accessors

    @staticmethod
    def _system_prompt(assistant: dict) -> tuple[str, int]:
        """The system message content, and how many system messages there were.

        More than one is reported rather than concatenated: which one the
        platform actually sends is not ours to guess, and a receipt that
        silently merges them would misdescribe the run.
        """
        messages = (assistant.get("model") or {}).get("messages") or []
        system = [
            (m.get("content") or "")
            for m in messages
            if (m.get("role") or "").lower() == "system"
        ]
        return (system[0].strip() if system else ""), len(system)

    # ----------------------------------------------------------------- verify

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between what the bench requires and what is live."""
        problems: list[str] = []
        assistant = self.assistant()

        live_greeting = (assistant.get("firstMessage") or "").strip()
        want_greeting = spec.greeting.strip()
        if live_greeting != want_greeting:
            problems.append(
                f"greeting mismatch\n    live: {live_greeting!r}\n    want: {want_greeting!r}"
            )

        live_prompt, system_count = self._system_prompt(assistant)
        want_prompt = spec.system_prompt.strip()
        if live_prompt != want_prompt:
            problems.append(
                f"system prompt mismatch\n    live: {live_prompt!r}\n    want: {want_prompt!r}"
            )
        if system_count > 1:
            problems.append(
                f"assistant has {system_count} system messages in model.messages -- "
                "the bench cannot tell which one is in force; leave exactly one"
            )

        if not live_greeting:
            problems.append(
                "assistant has no firstMessage -- the bench needs the agent to speak "
                "first or live VAD has no greeting to wait for (dashboard > assistant "
                "> First Message)"
            )

        mode = assistant.get("firstMessageMode")
        if mode and mode != REQUIRED_FIRST_MESSAGE_MODE:
            problems.append(
                f"firstMessageMode is {mode!r}, must be "
                f"{REQUIRED_FIRST_MESSAGE_MODE!r} -- "
                "'assistant-waits-for-user' means no greeting to detect, and the "
                "model-generated variant makes the greeting differ every call"
            )

        model = assistant.get("model") or {}
        if spec.model:
            live_model = _slash(model.get("provider"), model.get("model"))
            if not _matches(live_model, spec.model):
                problems.append(
                    f"model mismatch: live {live_model!r}, want {spec.model!r}"
                )
        if spec.stt:
            transcriber = assistant.get("transcriber") or {}
            live_stt = _slash(transcriber.get("provider"), transcriber.get("model"))
            if not _matches(live_stt, spec.stt):
                problems.append(f"stt mismatch: live {live_stt!r}, want {spec.stt!r}")
        if spec.tts:
            voice = assistant.get("voice") or {}
            live_voice = _slash(voice.get("provider"), voice.get("voiceId"))
            if not _matches(live_voice, spec.tts):
                problems.append(f"voice mismatch: live {live_voice!r}, want {spec.tts!r}")

        return problems

    # ------------------------------------------------------------ dial target

    def dial_target(self) -> DialTarget:
        """The number that reaches the assistant -- checked, not assumed.

        A hand-set number is the quietest way to publish a wrong measurement:
        the receipt describes the assistant pinned in vendors.yaml, so if the
        number reaches a DIFFERENT assistant the run completes, every turn is
        usable, the latencies look plausible, and the published configuration
        had nothing to do with them. That is not hypothetical -- it happened on
        Telnyx on 2026-07-30, where VAPI_PHONE_NUMBER's counterpart pointed at
        an echo bot. So an explicit number is verified against the assistant's
        own routing exactly as a discovered one is.
        """
        explicit = self.block.get("number") or settings.vapi_phone_number
        assistant = self.assistant()
        assistant_id = assistant.get("id") or self.assistant_id

        with self._client() as client:
            response = client.get("/phone-number", params={"limit": 100})
        if response.status_code >= 300:
            raise VendorNotReady(
                f"could not list this account's phone numbers (HTTP "
                f"{response.status_code}), so the number we would dial cannot be "
                f"tied to the config we would publish. Refusing rather than "
                f"guessing."
            )
        numbers = self._items(response.json())

        attached = [
            n for n in numbers
            if n.get("assistantId") == assistant_id
            and (n.get("status") or "active") == "active"
        ]
        routed = [n.get("number") for n in attached if n.get("number")]

        if explicit:
            if digits(explicit) not in {digits(n) for n in routed}:
                raise VendorNotReady(
                    f"{explicit} does not route to assistant "
                    f"{assistant.get('name')!r} ({assistant_id}).\n"
                    f"  Numbers that do: {', '.join(routed) or '(none)'}\n"
                    f"  Dialling it would measure a different agent while the "
                    f"published receipt described this one.\n"
                    f"  Either drop VAPI_PHONE_NUMBER (the right number is "
                    f"discovered automatically), or pin the assistant you "
                    f"actually mean in config/vendors.yaml."
                )
            return DialTarget(kind="pstn", value=explicit)

        if not attached:
            raise VendorNotReady(
                f"no phone number routes to assistant {assistant.get('name')!r} "
                f"({assistant_id}).\n"
                "  Attach one in the dashboard: Phone Numbers > pick or create a "
                "number > Inbound Settings > Assistant = this assistant.\n"
                "  (A free US number is available under Phone Numbers > Create.)"
            )
        if len(attached) > 1:
            log.info("assistant has %d numbers; using %s",
                     len(attached), attached[0].get("number"))

        if not routed:
            raise VendorNotReady(
                f"number {attached[0].get('id')} routes to the assistant but has no "
                "E.164 `number` field (BYO/SIP trunk?). The bench dials PSTN, so "
                "it cannot reach this assistant as configured."
            )
        return DialTarget(kind="pstn", value=routed[0])

    # ---------------------------------------------------------------- receipt

    def applied_config(self) -> AppliedConfig:
        """The config receipt that ships with every measurement."""
        assistant = self.assistant()
        model = assistant.get("model") or {}
        transcriber = assistant.get("transcriber") or {}
        voice = assistant.get("voice") or {}
        start_plan = assistant.get("startSpeakingPlan") or {}
        stop_plan = assistant.get("stopSpeakingPlan") or {}
        smart_plan = start_plan.get("smartEndpointingPlan") or {}
        transcription_plan = start_plan.get("transcriptionEndpointingPlan") or {}

        def _effective(key: str, live):
            """The value in force: what the API echoed, else the documented default."""
            return live if live is not None else DOCUMENTED_ENDPOINTING_DEFAULTS[key]

        defaults_used = {
            "model": _slash(model.get("provider"), model.get("model")),
            # Unlike Telnyx, Vapi exposes the LLM temperature -- recorded because
            # its presence here and absence there is exactly the kind of
            # incomparability the receipt exists to surface.
            "model_temperature": model.get("temperature"),
            "voice": _slash(voice.get("provider"), voice.get("voiceId")),
            "voice_speed": voice.get("speed"),
            "background_sound": assistant.get("backgroundSound"),
            "background_speech_denoising": bool(
                assistant.get("backgroundSpeechDenoisingPlan")
            ),
            "stt_model": _slash(transcriber.get("provider"), transcriber.get("model")),
            "stt_language": transcriber.get("language"),
            # The knobs that most directly set TTFAB on this platform: how long
            # the orchestrator waits before deciding the caller's turn ended.
            "endpointing": {
                "wait_seconds": _effective("wait_seconds",
                                           start_plan.get("waitSeconds")),
                "smart_endpointing_enabled": start_plan.get("smartEndpointingEnabled"),
                "smart_endpointing_provider": smart_plan.get("provider"),
                "on_punctuation_seconds": _effective(
                    "on_punctuation_seconds",
                    transcription_plan.get("onPunctuationSeconds")),
                "on_no_punctuation_seconds": _effective(
                    "on_no_punctuation_seconds",
                    transcription_plan.get("onNoPunctuationSeconds")),
                "on_number_seconds": _effective(
                    "on_number_seconds", transcription_plan.get("onNumberSeconds")),
                "custom_endpointing_rules": len(
                    start_plan.get("customEndpointingRules") or []
                ),
                # Barge-in side: not TTFAB itself, but it decides whether our own
                # question can cut the agent off mid-greeting (a `barged_greeting`
                # discard).
                "stop_num_words": _effective("stop_num_words",
                                             stop_plan.get("numWords")),
                "stop_voice_seconds": _effective("stop_voice_seconds",
                                                 stop_plan.get("voiceSeconds")),
                "stop_backoff_seconds": _effective("stop_backoff_seconds",
                                                   stop_plan.get("backoffSeconds")),
            },
            # Where the numbers above came from, per plan: an unset plan means
            # the platform default is in force, not that endpointing is unknown.
            "endpointing_source": {
                "start_speaking_plan": "api" if start_plan else "vapi-documented-default",
                "stop_speaking_plan": "api" if stop_plan else "vapi-documented-default",
            },
            "first_message_mode": assistant.get("firstMessageMode"),
            "first_message_interruptions_enabled": assistant.get(
                "firstMessageInterruptionsEnabled"
            ),
            # No idle-message plan exists on this platform's assistant schema, so
            # an `idle_filler` discard here cannot be attributed to a configured
            # idle reply the way it can on Telnyx.
            "user_idle_reply_secs": None,
            "max_duration_secs": assistant.get("maxDurationSeconds"),
            "voicemail_detection": bool(assistant.get("voicemailDetection")),
            "tools": [
                t.get("type") or t.get("name") for t in (model.get("tools") or [])
            ],
            "version_id": assistant.get("latestVersion") or assistant.get("updatedAt"),
            "assistant_id": assistant.get("id"),
        }

        unsupported = [
            # Recorded for symmetry with the Telnyx receipt: there is no
            # assistant-level idle-reply threshold in the Vapi API, so the bench
            # cannot report the setting that would explain an unprompted filler.
            "idle_reply_threshold",
        ]

        return AppliedConfig(
            vendor=self.name,
            normalized={
                "greeting": (assistant.get("firstMessage") or "").strip(),
                "instructions": self._system_prompt(assistant)[0],
                "model_requested": self.block.get("agent", {}).get("model"),
                "stt_requested": self.block.get("agent", {}).get("stt"),
                "tts_requested": self.block.get("agent", {}).get("tts"),
            },
            raw=assistant,
            defaults_used=defaults_used,
            unsupported=unsupported,
        )

    # ----------------------------------------------------------------- cost
    def call_costs(self, since: float, until: float) -> list[CallCost]:
        """Vapi bills per call and reports it inline on the call object.

        `costBreakdown` splits the total by component (stt/llm/tts/vapi), which
        is the most legible receipt of the five -- worth keeping whole, because
        a platform fee and an LLM bill move for different reasons.
        """
        with self._client() as client:
            response = client.get("/call", params={"limit": 100})
            response.raise_for_status()
            payload = response.json()
        items = payload if isinstance(payload, list) else self._items(payload)

        costs = []
        for call in items:
            started = iso_to_epoch(call.get("startedAt"))
            if started is None or not (since <= started <= until):
                continue
            ended = iso_to_epoch(call.get("endedAt"))
            costs.append(CallCost(
                vendor_call_id=str(call.get("id") or ""),
                cost=call.get("cost"),
                currency="USD",
                duration_s=(ended - started) if ended else None,
                billed_s=None,
                started_at=call.get("startedAt"),
                caller=(call.get("customer") or {}).get("number"),
                breakdown=call.get("costBreakdown") or {},
                source="GET /call .cost",
            ))
        return costs
