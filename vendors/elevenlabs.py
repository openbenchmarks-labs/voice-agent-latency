"""ElevenLabs as a vendor under test.

The most transparent platform in this directory: it names its LLM, its ASR
provider and its TTS model, and exposes the LLM temperature. Nothing goes in the
receipt's `unsupported` list, which is itself a finding worth publishing next to
Bland (which names a tier and hides everything behind it).

It is also the only one that sells no phone numbers -- inbound arrives over a
number you bring from Twilio, Exotel or your own SIP trunk. See dial_target.

READ-ONLY by contract. This module issues GETs and nothing else. `verify_agent`
reports mismatches for a human to fix; it never repairs them.

Field map (from a live read of the account and the OpenAPI spec, 2026-07-30).
Everything the bench cares about is nested under `conversation_config`:
  GET /v1/convai/agents/{id}
    conversation_config.agent.first_message            the greeting
    conversation_config.agent.prompt.prompt            the system prompt
    conversation_config.agent.prompt.llm / .temperature
    conversation_config.asr.provider / .quality        (STT is named here)
    conversation_config.tts.model_id / .voice_id / .optimize_streaming_latency
    conversation_config.turn.*                         turn-taking, incl. the
                                                       filler hazard below
    version_id / branch_id                             versioned config
  GET /v1/convai/phone-numbers -> [{phone_number, assigned_agent{agent_id}, ...}]
  auth: an `xi-api-key: <key>` header

THE FILLER HAZARD. `conversation_config.turn.soft_timeout_config` makes the agent
speak a stall phrase (this account's default is "Hhmmmm...yeah.") when the LLM is
slow. That is not a cosmetic setting for this benchmark: the filler IS the first
audio, so TTFAB would measure how fast the platform says "hmm" rather than how
fast it answers -- and because it is real speech of real duration it would sail
past the analyzer's ttfab_content_ms guard, which only screens out short noise.
So `verify_agent` refuses to bench an agent with it enabled, rather than
silently publishing a flattering number.
"""

from __future__ import annotations

import logging

import httpx

from harness.config import settings

from .base import (AgentSpec, AppliedConfig, CallCost, DialTarget, digits,
                    epoch_to_iso, iso_to_epoch)

log = logging.getLogger(__name__)

API = "https://api.elevenlabs.io"

# soft_timeout_config uses -1 to mean "off"; anything >= 0 arms the filler.
FILLER_DISABLED_TIMEOUT = -1.0


class VendorNotReady(RuntimeError):
    """Raised with human-actionable instructions, never auto-repaired."""


def _slash(*parts) -> str | None:
    kept = [str(p) for p in parts if p]
    return "/".join(kept) or None


class ElevenLabsVendor:
    name = "elevenlabs"

    def __init__(self, block: dict | None = None):
        self.block = block or {}
        self.agent_id = (
            self.block.get("agent_id") or settings.elevenlabs_agent_id
        )
        self._agent: dict | None = None

    # ------------------------------------------------------------------ fetch

    def _client(self) -> httpx.Client:
        if not settings.elevenlabs_api_key:
            raise VendorNotReady("ELEVENLABS_API_KEY is not set in .env")
        return httpx.Client(
            base_url=API,
            headers={"xi-api-key": settings.elevenlabs_api_key},
            timeout=15.0,
        )

    @staticmethod
    def _items(payload) -> list[dict]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("agents") or payload.get("data") or []
        return []

    def agent(self, refresh: bool = False) -> dict:
        """The live agent config. Cached per adapter instance."""
        if self._agent is not None and not refresh:
            return self._agent

        with self._client() as client:
            if self.agent_id:
                response = client.get(f"/v1/convai/agents/{self.agent_id}")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot read agent {self.agent_id}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                data = response.json()
            else:
                response = client.get("/v1/convai/agents")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot list agents: {response.status_code} "
                        f"{response.text[:200]}"
                    )
                agents = self._items(response.json())
                if not agents:
                    raise VendorNotReady(
                        "no Conversational AI agents on this ElevenLabs account -- "
                        "create one, give it the Northwind prompt and greeting from "
                        "data/voice-bench/ttfab_scenarios.json, then set "
                        "ELEVENLABS_AGENT_ID in .env"
                    )
                if len(agents) > 1:
                    names = ", ".join(
                        f"{a.get('name')}={a.get('agent_id')}" for a in agents
                    )
                    raise VendorNotReady(
                        "multiple agents found; pin one with ELEVENLABS_AGENT_ID in "
                        f".env or agent_id in config/vendors.yaml. Found: {names}"
                    )
                # The list view is a summary; fetch the pinned agent in full so the
                # receipt quotes the real configuration.
                agent_id = agents[0].get("agent_id")
                detail = client.get(f"/v1/convai/agents/{agent_id}")
                data = detail.json() if detail.status_code < 300 else agents[0]

        self._agent = data
        return data

    # -------------------------------------------------------------- accessors

    def _config(self) -> dict:
        return self.agent().get("conversation_config") or {}

    def _agent_block(self) -> dict:
        return self._config().get("agent") or {}

    def _prompt_block(self) -> dict:
        return self._agent_block().get("prompt") or {}

    def _soft_timeout(self) -> dict:
        return (self._config().get("turn") or {}).get("soft_timeout_config") or {}

    def filler_is_armed(self) -> bool:
        """Whether the platform will speak a stall phrase before the real answer."""
        timeout = self._soft_timeout().get("timeout_seconds")
        return timeout is not None and float(timeout) >= 0

    # ----------------------------------------------------------------- verify

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between what the bench requires and what is live."""
        problems: list[str] = []
        agent_block = self._agent_block()
        prompt_block = self._prompt_block()

        live_greeting = (agent_block.get("first_message") or "").strip()
        want_greeting = spec.greeting.strip()
        if live_greeting != want_greeting:
            problems.append(
                f"greeting mismatch (conversation_config.agent.first_message)\n"
                f"    live: {live_greeting!r}\n    want: {want_greeting!r}"
            )

        live_prompt = (prompt_block.get("prompt") or "").strip()
        want_prompt = spec.system_prompt.strip()
        if live_prompt != want_prompt:
            problems.append(
                f"prompt mismatch (conversation_config.agent.prompt.prompt)\n"
                f"    live: {live_prompt!r}\n    want: {want_prompt!r}"
            )

        if not live_greeting:
            problems.append(
                "agent has no first_message -- with it empty the agent waits for the "
                "caller, so there is no greeting for live VAD to detect and every "
                "call would time out on the greeting instead of measuring"
            )

        # The measurement-corrupting one. Not a preference: a stall phrase becomes
        # the first audio, so TTFAB would time the filler rather than the answer,
        # and unlike a noise burst it is long enough to pass ttfab_content_ms too.
        if self.filler_is_armed():
            soft = self._soft_timeout()
            problems.append(
                "turn.soft_timeout_config is ARMED "
                f"(timeout_seconds={soft.get('timeout_seconds')}, "
                f"message={soft.get('message')!r}) -- the agent speaks a stall "
                "phrase when the LLM is slow, which would become the first audio "
                "and make TTFAB measure the filler instead of the answer. Set "
                f"timeout_seconds to {FILLER_DISABLED_TIMEOUT} to disable it."
            )

        if prompt_block.get("custom_llm"):
            problems.append(
                "agent uses a custom_llm endpoint, so the model under test is not "
                "the platform's own -- out of scope for the closed division, which "
                "measures the product as shipped"
            )

        if spec.model:
            live_model = prompt_block.get("llm")
            if live_model != spec.model:
                problems.append(
                    f"model mismatch: live {live_model!r}, want {spec.model!r}"
                )
        if spec.stt:
            asr = self._config().get("asr") or {}
            live_stt = _slash(asr.get("provider"), asr.get("quality"))
            if live_stt != spec.stt and asr.get("provider") != spec.stt:
                problems.append(f"stt mismatch: live {live_stt!r}, want {spec.stt!r}")
        if spec.tts:
            tts = self._config().get("tts") or {}
            live_voice = _slash(tts.get("model_id"), tts.get("voice_id"))
            if live_voice != spec.tts and tts.get("voice_id") != spec.tts:
                problems.append(
                    f"voice mismatch: live {live_voice!r}, want {spec.tts!r}"
                )

        return problems

    # ------------------------------------------------------------ dial target

    def dial_target(self) -> DialTarget:
        """The number assigned to this agent -- checked, not assumed.

        ElevenLabs sells no numbers: inbound arrives over one you bring from
        Twilio, Exotel or your own SIP trunk, which is why the not-ready message
        talks about importing rather than buying.

        An explicitly configured number is verified against the agent's
        assignment rather than trusted. A number reaching a DIFFERENT agent
        produces a run where every turn is usable and every latency is
        plausible, published against an agent that had nothing to do with it --
        which is what happened on Telnyx on 2026-07-30.
        """
        explicit = self.block.get("number") or settings.elevenlabs_phone_number
        agent = self.agent()
        agent_id = agent.get("agent_id") or self.agent_id

        # The agent echoes its own numbers; fall back to the account list.
        assigned = [n for n in (agent.get("phone_numbers") or []) if n.get("phone_number")]
        if not assigned:
            with self._client() as client:
                response = client.get("/v1/convai/phone-numbers")
            if response.status_code >= 300:
                raise VendorNotReady(
                    f"could not list this account's phone numbers (HTTP "
                    f"{response.status_code}), so the number we would dial cannot "
                    f"be tied to the config we would publish. Refusing rather "
                    f"than guessing."
                )
            numbers = self._items(response.json())
            assigned = [
                n for n in numbers
                if ((n.get("assigned_agent") or {}).get("agent_id") == agent_id)
                and n.get("supports_inbound", True)
            ]

        routed = [n.get("phone_number") for n in assigned if n.get("phone_number")]
        if explicit:
            if digits(explicit) not in {digits(n) for n in routed}:
                raise VendorNotReady(
                    f"{explicit} is not assigned to agent {agent.get('name')!r} "
                    f"({agent_id}).\n"
                    f"  Numbers that are: {', '.join(routed) or '(none)'}\n"
                    f"  Dialling it would measure a different agent while the "
                    f"published receipt described this one.\n"
                    f"  Either drop ELEVENLABS_PHONE_NUMBER (the right number is "
                    f"discovered automatically), or pin the agent you actually "
                    f"mean in config/vendors.yaml."
                )
            return DialTarget(kind="pstn", value=explicit)

        if not assigned:
            raise VendorNotReady(
                f"no phone number is assigned to agent {agent.get('name')!r} "
                f"({agent_id}).\n"
                "  ElevenLabs does not sell numbers -- import one you already own:\n"
                "    Twilio: POST /v1/convai/phone-numbers with phone_number, label, "
                "sid, token\n"
                "    SIP trunk: POST /v1/convai/phone-numbers with phone_number, "
                "label + trunk config\n"
                "  then assign it to this agent.\n"
                "  NOTE: routing it over the bench's own carrier keeps both legs on "
                "one network, which is not the PSTN path the other vendors are "
                "measured on -- prefer an independent trunk."
            )
        if len(assigned) > 1:
            log.info("agent has %d numbers; using %s",
                     len(assigned), assigned[0].get("phone_number"))
        return DialTarget(kind="pstn", value=assigned[0]["phone_number"])

    # ---------------------------------------------------------------- receipt

    def applied_config(self) -> AppliedConfig:
        """The config receipt that ships with every measurement."""
        agent = self.agent()
        config = self._config()
        agent_block = self._agent_block()
        prompt_block = self._prompt_block()
        asr = config.get("asr") or {}
        tts = config.get("tts") or {}
        turn = config.get("turn") or {}
        soft = self._soft_timeout()

        defaults_used = {
            "model": prompt_block.get("llm"),
            "model_temperature": prompt_block.get("temperature"),
            "model_max_tokens": prompt_block.get("max_tokens"),
            "reasoning_effort": prompt_block.get("reasoning_effort"),
            "backup_llm": (prompt_block.get("backup_llm_config") or {}).get("llm"),
            "voice": _slash(tts.get("model_id"), tts.get("voice_id")),
            "voice_speed": tts.get("speed"),
            "tts_optimize_streaming_latency": tts.get("optimize_streaming_latency"),
            "stt_model": _slash(asr.get("provider"), asr.get("quality")),
            "stt_audio_format": asr.get("user_input_audio_format"),
            "language": agent_block.get("language"),
            # The knobs that most directly set TTFAB on this platform.
            "endpointing": {
                "turn_mode": turn.get("mode"),
                "turn_model": turn.get("turn_model"),
                "turn_eagerness": turn.get("turn_eagerness"),
                "turn_timeout_s": turn.get("turn_timeout"),
                "initial_wait_time": turn.get("initial_wait_time"),
                "speculative_turn": turn.get("speculative_turn"),
                "background_voice_detection": (config.get("vad") or {}).get(
                    "background_voice_detection"
                ),
                "disable_first_message_interruptions": agent_block.get(
                    "disable_first_message_interruptions"
                ),
            },
            # Recorded prominently because an armed filler does not slow a
            # measurement down, it makes one look FAST -- the stall phrase becomes
            # the first audio. The gate refuses to run with it on; this proves it
            # was off for the run.
            "filler_armed": self.filler_is_armed(),
            "filler_timeout_seconds": soft.get("timeout_seconds"),
            "filler_message": soft.get("message"),
            "max_duration_s": (config.get("conversation") or {}).get(
                "max_duration_seconds"
            ),
            "tools": [
                t.get("name") or t.get("type")
                for t in (prompt_block.get("tools") or [])
            ],
            "built_in_tools": sorted(
                key for key, value in (prompt_block.get("built_in_tools") or {}).items()
                if value
            ),
            "agent_id": agent.get("agent_id"),
            "version_id": agent.get("version_id"),
            "branch_id": agent.get("branch_id"),
        }

        # Deliberately empty, and that is the point: this platform names its LLM,
        # its ASR provider and its TTS model, and exposes temperature. Publishing an
        # empty list next to Bland's three-item one is the comparison.
        unsupported: list[str] = []

        return AppliedConfig(
            vendor=self.name,
            normalized={
                "greeting": (agent_block.get("first_message") or "").strip(),
                "instructions": (prompt_block.get("prompt") or "").strip(),
                "model_requested": self.block.get("agent", {}).get("model"),
                "stt_requested": self.block.get("agent", {}).get("stt"),
                "tts_requested": self.block.get("agent", {}).get("tts"),
            },
            raw=agent,
            defaults_used=defaults_used,
            unsupported=unsupported,
        )

    # ----------------------------------------------------------------- cost
    def call_costs(self, since: float, until: float) -> list[CallCost]:
        """Two calls: the list has no cost, the conversation detail does.

        `cost` is in ElevenLabs credits and `cost_fiat` is the dollar figure --
        we take the dollars so the board compares one currency, and keep the
        credits in the breakdown because the credit price is what a subscriber
        actually spends down.

        The account tier travels with the number. On a free or promotional tier
        `cost_fiat` is what the conversation WOULD list at, not what was
        charged, and a benchmark that printed that as a paid price would be
        quoting a discount as a rate.
        """
        with self._client() as client:
            listing = client.get("/v1/convai/conversations",
                                 params={"page_size": 100})
            listing.raise_for_status()
            conversations = (listing.json() or {}).get("conversations") or []

            costs = []
            for summary in conversations:
                started = summary.get("start_time_unix_secs")
                if started is None or not (since <= float(started) <= until):
                    continue
                conversation_id = summary.get("conversation_id")
                detail = client.get(f"/v1/convai/conversations/{conversation_id}")
                if detail.status_code >= 300:
                    continue
                meta = (detail.json() or {}).get("metadata") or {}
                charging = meta.get("charging") or {}
                phone = meta.get("phone_call") or {}
                tier = charging.get("tier")
                notes = ["cost_fiat is the USD figure; `cost` is in credits"]
                if tier and tier != "paid":
                    notes.append(f"account tier is {tier!r} -- cost_fiat is the "
                                 f"list-price equivalent, not an amount billed")
                costs.append(CallCost(
                    vendor_call_id=str(conversation_id or ""),
                    cost=meta.get("cost_fiat"),
                    currency="USD",
                    duration_s=meta.get("call_duration_secs"),
                    billed_s=None,
                    started_at=epoch_to_iso(float(started)),
                    caller=phone.get("external_number"),
                    breakdown={"credits": meta.get("cost"),
                               "tier": tier,
                               "llm_usage": charging.get("llm_usage")},
                    source="GET /v1/convai/conversations/{id} .metadata.cost_fiat",
                    notes=tuple(notes),
                ))
        return costs
