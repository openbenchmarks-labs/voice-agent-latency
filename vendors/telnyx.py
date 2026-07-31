"""Telnyx as a vendor under test.

Telnyx is the vertically-integrated case: carrier, STT, LLM, TTS and
orchestration all in-house. That makes it the vendor a fixed-stack methodology
structurally cannot bench (you cannot swap Deepgram into a platform that owns its
own STT) -- and therefore the most interesting first row for a benchmark that
measures products as shipped.

READ-ONLY by contract. This module issues GETs and nothing else. `verify_agent`
reports mismatches for a human to fix in the portal; it never repairs them.

Facts verified live against the account (2026-07-28):
  GET /v2/ai/assistants[/{id}]  -> full config incl. voice/transcription/
                                   interruption/telephony settings
  GET /v2/phone_numbers?filter[connection_id]=<telephony_settings
      .default_texml_app_id>    -> the number(s) that reach the assistant

The assistant gets its OWN TeXML application id, separate from the harness's.
Attaching a number to that app is what makes the assistant answer calls, and it
is a portal action the operator performs.
"""

from __future__ import annotations

import logging

import httpx

from harness.config import settings

from .base import (AgentSpec, AppliedConfig, CallCost, DialTarget, digits,
                    iso_to_epoch)

log = logging.getLogger(__name__)

API = "https://api.telnyx.com"


class VendorNotReady(RuntimeError):
    """Raised with human-actionable instructions, never auto-repaired."""


#: Compare numbers by digits: +1 555..., 1555... and 555... are one number.
#: Shared with every other adapter that checks a configured number.
_digits = digits


class TelnyxVendor:
    name = "telnyx"

    def __init__(self, block: dict | None = None):
        self.block = block or {}
        self.assistant_id = (
            self.block.get("assistant_id") or settings.telnyx_assistant_id
        )
        self._assistant: dict | None = None

    # ------------------------------------------------------------------ fetch

    def _client(self) -> httpx.Client:
        if not settings.telnyx_api_key:
            raise VendorNotReady("TELNYX_API_KEY is not set in .env")
        return httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {settings.telnyx_api_key}"},
            timeout=15.0,
        )

    def assistant(self, refresh: bool = False) -> dict:
        """The live assistant config. Cached per adapter instance."""
        if self._assistant is not None and not refresh:
            return self._assistant

        with self._client() as client:
            if self.assistant_id:
                response = client.get(f"/v2/ai/assistants/{self.assistant_id}")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot read assistant {self.assistant_id}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                data = response.json().get("data", response.json())
            else:
                response = client.get("/v2/ai/assistants")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot list assistants: {response.status_code}"
                    )
                assistants = response.json().get("data", [])
                if not assistants:
                    raise VendorNotReady(
                        "no AI assistants on this Telnyx account -- create one in the "
                        "portal (AI > Assistants), then set TELNYX_ASSISTANT_ID in .env"
                    )
                if len(assistants) > 1:
                    names = ", ".join(f"{a['name']}={a['id']}" for a in assistants)
                    raise VendorNotReady(
                        "multiple assistants found; pin one with TELNYX_ASSISTANT_ID "
                        f"in .env or assistant_id in config/vendors.yaml. Found: {names}"
                    )
                data = assistants[0]

        self._assistant = data
        return data

    # ----------------------------------------------------------------- verify

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between what the bench requires and what is live.

        Compares the two things that change what the agent SAYS -- greeting and
        instructions -- verbatim modulo leading/trailing whitespace. Note the
        portal stores a curly apostrophe in "shop's"; the committed prompt file
        must match byte-for-byte or this reports a diff (which is the point: the
        published config receipt has to be the text that actually ran).
        """
        problems: list[str] = []
        assistant = self.assistant()

        live_greeting = (assistant.get("greeting") or "").strip()
        want_greeting = spec.greeting.strip()
        if live_greeting != want_greeting:
            problems.append(
                f"greeting mismatch\n    live: {live_greeting!r}\n    want: {want_greeting!r}"
            )

        live_prompt = (assistant.get("instructions") or "").strip()
        want_prompt = spec.system_prompt.strip()
        if live_prompt != want_prompt:
            problems.append(
                f"instructions mismatch\n    live: {live_prompt!r}\n    want: {want_prompt!r}"
            )

        # The agent must speak first, or there is no greeting to detect and the
        # whole turn-taking choreography has nothing to wait for.
        if not live_greeting:
            problems.append(
                "assistant has no greeting -- the bench needs 'Assistant speaks first' "
                "with non-empty greeting text (portal > assistant > Agent > Greeting Mode)"
            )

        if "telephony" not in (assistant.get("enabled_features") or []):
            problems.append(
                "assistant does not have the 'telephony' feature enabled -- it cannot "
                "answer phone calls"
            )

        # Pinned-stack requests are only meaningful if the vendor exposes them;
        # Telnyx does for model/voice/stt, which is why they are checkable here.
        if spec.model and assistant.get("model") != spec.model:
            problems.append(
                f"model mismatch: live {assistant.get('model')!r}, want {spec.model!r}"
            )
        if spec.stt:
            live_stt = (assistant.get("transcription") or {}).get("model")
            if live_stt != spec.stt:
                problems.append(f"stt mismatch: live {live_stt!r}, want {spec.stt!r}")
        if spec.tts:
            live_voice = (assistant.get("voice_settings") or {}).get("voice")
            if live_voice != spec.tts:
                problems.append(f"voice mismatch: live {live_voice!r}, want {spec.tts!r}")

        return problems

    # ------------------------------------------------------------ dial target

    def dial_target(self) -> DialTarget:
        """The number that reaches the assistant -- checked, not assumed.

        A hand-set number is the quietest way to publish a wrong measurement.
        The receipt describes the assistant pinned in vendors.yaml, so if the
        number reaches some OTHER assistant the report claims a configuration
        that had nothing to do with the numbers in it, and nothing about the
        output looks wrong. (Measured 2026-07-30: TELNYX_VENDOR_NUMBER pointed
        at an echo agent while the receipt described the pinned assistant. The
        run completed, every turn was usable, and the latencies were an echo
        bot's.) So an explicit number must still be attached to the pinned
        assistant's own TeXML application, exactly as a discovered one is.
        """
        explicit = self.block.get("number") or settings.telnyx_vendor_number
        assistant = self.assistant()
        app_id = (assistant.get("telephony_settings") or {}).get("default_texml_app_id")
        if not app_id:
            raise VendorNotReady(
                f"assistant {assistant.get('name')!r} has no telephony application"
            )

        with self._client() as client:
            response = client.get("/v2/phone_numbers",
                                  params={"filter[connection_id]": app_id})
        if response.status_code >= 300:
            raise VendorNotReady(
                f"could not list the numbers attached to assistant "
                f"{assistant.get('name')!r} (HTTP {response.status_code}), so the "
                f"number we would dial cannot be tied to the config we would "
                f"publish. Refusing rather than guessing."
            )
        numbers = response.json().get("data", [])
        active = [n["phone_number"] for n in numbers if n.get("status") == "active"]

        if explicit:
            if _digits(explicit) not in {_digits(n) for n in active}:
                raise VendorNotReady(
                    f"{explicit} is not attached to assistant "
                    f"{assistant.get('name')!r} ({assistant.get('id')}), whose "
                    f"TeXML app is {app_id}.\n"
                    f"  Attached numbers: {', '.join(active) or '(none)'}\n"
                    f"  Dialling it would measure a different agent while the "
                    f"published receipt described this one.\n"
                    f"  Either drop TELNYX_VENDOR_NUMBER (the right number is "
                    f"discovered automatically), or pin the assistant you "
                    f"actually mean in config/vendors.yaml."
                )
            return DialTarget(kind="pstn", value=explicit)

        if not active:
            raise VendorNotReady(
                f"no phone number is attached to assistant "
                f"{assistant.get('name')!r} (its TeXML app is {app_id}).\n"
                "  Attach one in the portal: AI > Assistants > your assistant > "
                "Calling tab, pick a voice-enabled number."
            )
        if len(active) > 1:
            log.info("assistant has %d numbers; using %s", len(active), active[0])
        return DialTarget(kind="pstn", value=active[0])

    # ---------------------------------------------------------------- receipt

    def applied_config(self) -> AppliedConfig:
        """The config receipt that ships with every measurement."""
        assistant = self.assistant()
        voice = assistant.get("voice_settings") or {}
        transcription = assistant.get("transcription") or {}
        stt_settings = transcription.get("settings") or {}
        interruption = assistant.get("interruption_settings") or {}
        speaking_plan = interruption.get("start_speaking_plan") or {}
        telephony = assistant.get("telephony_settings") or {}

        defaults_used = {
            "model": assistant.get("model"),
            "voice": voice.get("voice"),
            "voice_speed": voice.get("voice_speed"),
            "background_audio": voice.get("background_audio"),
            "stt_model": transcription.get("model"),
            "stt_language": transcription.get("language"),
            # The knobs that most directly set TTFAB on this platform: how long
            # the agent waits before deciding the caller's turn ended.
            "endpointing": {
                "eot_threshold": stt_settings.get("eot_threshold"),
                "eot_timeout_ms": stt_settings.get("eot_timeout_ms"),
                "eager_eot_threshold": stt_settings.get("eager_eot_threshold"),
                "start_speaking_wait_seconds": speaking_plan.get("wait_seconds"),
                "transcription_endpointing_plan": speaking_plan.get(
                    "transcription_endpointing_plan"
                ),
                "interrupt_prediction_threshold": interruption.get(
                    "interrupt_prediction_threshold"
                ),
            },
            "noise_suppression": telephony.get("noise_suppression"),
            # Why this matters: after this many seconds of caller silence the
            # assistant speaks unprompted. That is the `idle_filler` hazard the
            # analyzer discards on -- recorded here so the discard is explicable.
            "user_idle_reply_secs": telephony.get("user_idle_reply_secs"),
            "time_limit_secs": telephony.get("time_limit_secs"),
            "tools": [t.get("type") or t.get("name") for t in assistant.get("tools", [])],
            "version_id": assistant.get("version_id"),
            "assistant_id": assistant.get("id"),
        }

        unsupported = [
            # Reply determinism wants temperature 0. The Telnyx
            # assistant API exposes no LLM temperature (voice_settings.temperature
            # is a TTS knob), so this vendor cannot be pinned that way. Recorded
            # rather than footnoted -- this is exactly the incomparability the
            # receipt exists to make visible.
            "llm_temperature",
        ]

        return AppliedConfig(
            vendor=self.name,
            normalized={
                "greeting": (assistant.get("greeting") or "").strip(),
                "instructions": (assistant.get("instructions") or "").strip(),
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
        """Telnyx splits this across two resources and neither is sufficient.

        `/v2/ai/conversations` knows WHO called (metadata.from) but carries no
        cost; `/v2/detail_records` carries the cost but identifies the call only
        by call_control_id. So they are joined on that id: conversations supply
        the caller, detail records supply the money.

        Telnyx bills a 60-SECOND MINIMUM -- a 42 s call reports duration_sec 42
        and billed_sec 60. Both are kept: cost-per-minute computed against real
        duration is what the platform costs to run, computed against billed
        seconds is what the invoice says, and on short bench calls those differ
        by a lot. Nothing here rounds one into the other.
        """
        with self._client() as client:
            conversations = client.get("/v2/ai/conversations",
                                       params={"page[size]": 100})
            conversations.raise_for_status()
            by_control_id = {}
            for conversation in (conversations.json() or {}).get("data") or []:
                meta = conversation.get("metadata") or {}
                control_id = meta.get("call_control_id")
                if control_id:
                    by_control_id[control_id] = conversation

            records = client.get("/v2/detail_records",
                                 params={"filter[record_type]": "ai-voice-assistant",
                                         "page[size]": 250})
            records.raise_for_status()
            rows = (records.json() or {}).get("data") or []

        costs = []
        for row in rows:
            started = iso_to_epoch(row.get("created_at"))
            if started is None or not (since <= started <= until):
                continue
            conversation = by_control_id.get(row.get("call_control_id")) or {}
            meta = conversation.get("metadata") or {}
            cost = row.get("cost")
            billed = row.get("billed_sec")
            duration = row.get("duration_sec")
            notes = ["cost is the ai-voice-assistant detail record; a separate "
                     "PSTN leg may be billed under another record type"]
            if billed and duration and float(billed) > float(duration):
                notes.append(f"billed {billed}s for a {duration}s call "
                             f"(Telnyx bills a 60-second minimum)")
            costs.append(CallCost(
                vendor_call_id=str(row.get("conversation_id")
                                   or row.get("call_control_id") or ""),
                cost=float(cost) if cost is not None else None,
                currency=row.get("currency") or "USD",
                duration_s=float(duration) if duration is not None else None,
                billed_s=float(billed) if billed is not None else None,
                started_at=row.get("created_at"),
                caller=meta.get("from"),
                breakdown={"rate": row.get("rate"),
                           "rate_measured_in": row.get("rate_measured_in"),
                           "llm_model": row.get("llm_model"),
                           "stt_model": row.get("stt_model"),
                           "tts_model_id": row.get("tts_model_id")},
                source="GET /v2/detail_records[ai-voice-assistant] .cost",
                notes=tuple(notes),
            ))
        return costs
