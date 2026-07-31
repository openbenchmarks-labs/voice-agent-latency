"""Retell AI agent as a vendor under test.

Retell is the two-object case. The agent holds voice, language and the
turn-taking knobs; its `response_engine` points at a SEPARATE retell-llm object
that holds the prompt (`general_prompt`), the greeting (`begin_message`) and the
model. Both have to be read to know what the agent will say, and both go in the
receipt -- an agent id alone does not identify a configuration here.

READ-ONLY by contract. This module issues GETs and nothing else. `verify_agent`
reports mismatches for a human to fix in the dashboard; it never repairs them.

Facts verified live against the account (2026-07-30):
  GET /list-agents                  -> agents (bare array)
  GET /get-agent/{agent_id}         -> voice_id, language, interruption_sensitivity,
                                       max_call_duration_ms, response_engine{llm_id},
                                       version; optional latency knobs appear only
                                       when set
  GET /get-retell-llm/{llm_id}      -> model, model_temperature, model_high_priority,
                                       start_speaker, begin_message, general_prompt
  GET /list-phone-numbers           -> phone_number, inbound_agent_id /
                                       inbound_agents[], outbound_agent_id

`start_speaker` is this platform's version of "who talks first", and the blank
dashboard template ships it as "user" -- which would leave the bench's live VAD
with no greeting to wait for. It is checked, not assumed.
"""

from __future__ import annotations

import logging

import httpx

from harness.config import settings

from .base import (AgentSpec, AppliedConfig, CallCost, DialTarget, digits,
                    epoch_to_iso, iso_to_epoch)

log = logging.getLogger(__name__)

API = "https://api.retellai.com"

# The agent must speak first or there is no greeting to detect and the whole
# turn-taking choreography has nothing to trigger on.
REQUIRED_START_SPEAKER = "agent"


class VendorNotReady(RuntimeError):
    """Raised with human-actionable instructions, never auto-repaired."""


class RetellVendor:
    name = "retell"

    def __init__(self, block: dict | None = None):
        self.block = block or {}
        self.agent_id = self.block.get("agent_id") or settings.retell_agent_id
        self._agent: dict | None = None
        self._llm: dict | None = None

    # ------------------------------------------------------------------ fetch

    def _client(self) -> httpx.Client:
        if not settings.retell_api_key:
            raise VendorNotReady("RETELL_API_KEY is not set in .env")
        return httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {settings.retell_api_key}"},
            timeout=15.0,
        )

    @staticmethod
    def _items(payload) -> list[dict]:
        """Retell list endpoints return a bare array; tolerate {"data": [...]}."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("data") or payload.get("results") or []
        return []

    def agent(self, refresh: bool = False) -> dict:
        """The live agent config. Cached per adapter instance."""
        if self._agent is not None and not refresh:
            return self._agent

        with self._client() as client:
            if self.agent_id:
                response = client.get(f"/get-agent/{self.agent_id}")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot read agent {self.agent_id}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                data = response.json()
            else:
                response = client.get("/list-agents")
                if response.status_code >= 300:
                    raise VendorNotReady(
                        f"cannot list agents: {response.status_code} "
                        f"{response.text[:200]}"
                    )
                agents = self._items(response.json())
                if not agents:
                    raise VendorNotReady(
                        "no agents on this Retell account -- create one in the "
                        "dashboard, give it the Northwind prompt and greeting from "
                        "data/voice-bench/ttfab_scenarios.json, then set "
                        "RETELL_AGENT_ID in .env"
                    )
                if len(agents) > 1:
                    names = ", ".join(
                        f"{a.get('agent_name')}={a.get('agent_id')}" for a in agents
                    )
                    raise VendorNotReady(
                        "multiple agents found; pin one with RETELL_AGENT_ID in .env "
                        f"or agent_id in config/vendors.yaml. Found: {names}"
                    )
                data = agents[0]

        self._agent = data
        return data

    def llm(self, refresh: bool = False) -> dict:
        """The response engine behind the agent: where the prompt actually lives.

        A non-retell-llm engine (custom-llm, conversation-flow) keeps its prompt
        somewhere this adapter cannot read, so it is refused rather than reported
        as an empty prompt -- which would look like drift instead of the
        unsupported configuration it is.
        """
        if self._llm is not None and not refresh:
            return self._llm

        engine = self.agent().get("response_engine") or {}
        kind = engine.get("type")
        if kind != "retell-llm":
            raise VendorNotReady(
                f"agent's response_engine is {kind!r}; this adapter reads the "
                "'retell-llm' engine, whose prompt it can verify. Point the agent "
                "at a Retell LLM, or extend vendors/retell.py for that engine."
            )
        llm_id = engine.get("llm_id")
        if not llm_id:
            raise VendorNotReady("agent's response_engine has no llm_id")

        with self._client() as client:
            response = client.get(f"/get-retell-llm/{llm_id}")
        if response.status_code >= 300:
            raise VendorNotReady(
                f"cannot read retell-llm {llm_id}: {response.status_code} "
                f"{response.text[:200]}"
            )
        self._llm = response.json()
        return self._llm

    # ----------------------------------------------------------------- verify

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between what the bench requires and what is live."""
        problems: list[str] = []
        agent = self.agent()
        llm = self.llm()

        live_greeting = (llm.get("begin_message") or "").strip()
        want_greeting = spec.greeting.strip()
        if live_greeting != want_greeting:
            problems.append(
                f"greeting mismatch (retell-llm.begin_message)\n"
                f"    live: {live_greeting!r}\n    want: {want_greeting!r}"
            )

        live_prompt = (llm.get("general_prompt") or "").strip()
        want_prompt = spec.system_prompt.strip()
        if live_prompt != want_prompt:
            problems.append(
                f"prompt mismatch (retell-llm.general_prompt)\n"
                f"    live: {live_prompt!r}\n    want: {want_prompt!r}"
            )

        if not live_greeting:
            problems.append(
                "retell-llm has no begin_message -- the bench needs the agent to "
                "speak first or live VAD has no greeting to wait for (dashboard > "
                "agent > Begin Message)"
            )

        start_speaker = llm.get("start_speaker")
        if start_speaker and start_speaker != REQUIRED_START_SPEAKER:
            problems.append(
                f"start_speaker is {start_speaker!r}, must be "
                f"{REQUIRED_START_SPEAKER!r} -- with 'user' the agent waits for the "
                "caller and never delivers a greeting to measure"
            )

        if spec.model:
            live_model = llm.get("model")
            if live_model != spec.model:
                problems.append(
                    f"model mismatch: live {live_model!r}, want {spec.model!r}"
                )
        if spec.tts:
            live_voice = agent.get("voice_id")
            if live_voice != spec.tts:
                problems.append(
                    f"voice mismatch: live {live_voice!r}, want {spec.tts!r}"
                )
        if spec.stt:
            # Retell does not expose an STT provider/model on the agent -- only
            # stt_mode and vocab hints -- so a pinned STT cannot be honoured here.
            problems.append(
                f"stt {spec.stt!r} cannot be pinned on this platform: Retell "
                "exposes no STT provider or model on the agent (see "
                "applied_config().unsupported)"
            )

        return problems

    # ------------------------------------------------------------ dial target

    def dial_target(self) -> DialTarget:
        """The number whose INBOUND agent is this agent -- checked, not assumed.

        Inbound is what matters: the bench dials the vendor. A number bound only
        as `outbound_agent_id` would never reach it.

        An explicitly configured number is verified against that binding rather
        than trusted. A number reaching a DIFFERENT agent produces a run where
        every turn is usable and every latency is plausible, published against
        an agent that had nothing to do with it -- which is what happened on
        Telnyx on 2026-07-30.
        """
        explicit = self.block.get("number") or settings.retell_phone_number
        agent_id = self.agent().get("agent_id") or self.agent_id

        with self._client() as client:
            response = client.get("/list-phone-numbers")
        if response.status_code >= 300:
            raise VendorNotReady(
                f"could not list this account's phone numbers (HTTP "
                f"{response.status_code}), so the number we would dial cannot be "
                f"tied to the config we would publish. Refusing rather than "
                f"guessing."
            )
        numbers = self._items(response.json())

        def _inbound_ids(number: dict) -> list[str]:
            # Retell has both shapes: a single inbound_agent_id and a weighted
            # inbound_agents list. Read both so a weighted binding still counts.
            ids = []
            if number.get("inbound_agent_id"):
                ids.append(number["inbound_agent_id"])
            for entry in number.get("inbound_agents") or []:
                if isinstance(entry, dict) and entry.get("agent_id"):
                    ids.append(entry["agent_id"])
                elif isinstance(entry, str):
                    ids.append(entry)
            return ids

        attached = [n for n in numbers if agent_id in _inbound_ids(n)]
        routed = [n.get("phone_number") for n in attached if n.get("phone_number")]

        if explicit:
            if digits(explicit) not in {digits(n) for n in routed}:
                raise VendorNotReady(
                    f"{explicit} is not bound to agent {agent_id} as its inbound "
                    f"agent.\n"
                    f"  Numbers that are: {', '.join(routed) or '(none)'}\n"
                    f"  Dialling it would measure a different agent while the "
                    f"published receipt described this one.\n"
                    f"  Either drop RETELL_PHONE_NUMBER (the right number is "
                    f"discovered automatically), or pin the agent you actually "
                    f"mean in config/vendors.yaml."
                )
            return DialTarget(kind="pstn", value=explicit)

        if not attached:
            outbound_only = [
                n.get("phone_number") for n in numbers
                if n.get("outbound_agent_id") == agent_id
            ]
            hint = ""
            if outbound_only:
                hint = (
                    f"\n  {outbound_only[0]} has this agent as its OUTBOUND agent "
                    "only -- the bench dials in, so it must be the inbound agent."
                )
            raise VendorNotReady(
                f"no phone number has agent {agent_id} as its inbound agent."
                f"{hint}\n"
                "  Bind one in the dashboard: Phone Numbers > pick a number > "
                "Inbound Agent = this agent."
            )
        if len(attached) > 1:
            log.info("agent has %d inbound numbers; using %s",
                     len(attached), attached[0].get("phone_number"))

        number = attached[0].get("phone_number")
        if not number:
            raise VendorNotReady(
                "the bound number has no `phone_number` field -- set "
                "RETELL_PHONE_NUMBER in .env with the number that reaches it"
            )
        return DialTarget(kind="pstn", value=number)

    # ---------------------------------------------------------------- receipt

    def applied_config(self) -> AppliedConfig:
        """The config receipt that ships with every measurement."""
        agent = self.agent()
        llm = self.llm()

        defaults_used = {
            "model": llm.get("model"),
            # Retell exposes LLM temperature (Telnyx does not), so the
            # reply-determinism request is satisfiable here. None means the
            # platform default is in force.
            "model_temperature": llm.get("model_temperature"),
            "model_high_priority": llm.get("model_high_priority"),
            "voice": agent.get("voice_id"),
            "voice_model": agent.get("voice_model"),
            "voice_speed": agent.get("voice_speed"),
            "language": agent.get("language"),
            "ambient_sound": agent.get("ambient_sound"),
            "denoising_mode": agent.get("denoising_mode"),
            # The knobs that most directly set TTFAB on this platform. Retell
            # returns a field only when it has been set, so None here means "not
            # overridden, platform default in force" -- which is the closed
            # division's answer anyway.
            "endpointing": {
                "responsiveness": agent.get("responsiveness"),
                "interruption_sensitivity": agent.get("interruption_sensitivity"),
                "enable_backchannel": agent.get("enable_backchannel"),
                "begin_message_delay_ms": agent.get("begin_message_delay_ms"),
                "stt_mode": agent.get("stt_mode"),
                "vocab_specialization": agent.get("vocab_specialization"),
            },
            "start_speaker": llm.get("start_speaker"),
            # Why this matters: the analyzer discards a call whose vendor speaks
            # unprompted (`idle_filler`); this is the threshold that would cause it.
            "reminder_trigger_ms": agent.get("reminder_trigger_ms"),
            "reminder_max_count": agent.get("reminder_max_count"),
            "end_call_after_silence_ms": agent.get("end_call_after_silence_ms"),
            "max_call_duration_ms": agent.get("max_call_duration_ms"),
            "tools": [
                t.get("type") or t.get("name")
                for t in (llm.get("general_tools") or [])
            ],
            # Both halves are versioned independently, and the prompt lives in the
            # llm -- an agent version alone would not identify what was said.
            "agent_id": agent.get("agent_id"),
            "agent_version": agent.get("version"),
            "llm_id": llm.get("llm_id"),
            "llm_version": llm.get("version"),
            "agent_last_modified": agent.get("last_modification_timestamp"),
            "llm_last_modified": llm.get("last_modification_timestamp"),
        }

        unsupported = [
            # Retell owns the speech-to-text path and exposes no provider/model
            # for it (only stt_mode and vocabulary hints), so the stack cannot be
            # equalised against a vendor that does -- recorded rather than
            # footnoted, same as llm_temperature on Telnyx.
            "stt_provider",
        ]

        return AppliedConfig(
            vendor=self.name,
            normalized={
                "greeting": (llm.get("begin_message") or "").strip(),
                "instructions": (llm.get("general_prompt") or "").strip(),
                "model_requested": self.block.get("agent", {}).get("model"),
                "stt_requested": self.block.get("agent", {}).get("stt"),
                "tts_requested": self.block.get("agent", {}).get("tts"),
            },
            # Both objects, because neither alone describes the run.
            raw={"agent": agent, "retell_llm": llm},
            defaults_used=defaults_used,
            unsupported=unsupported,
        )

    # ----------------------------------------------------------------- cost
    def call_costs(self, since: float, until: float) -> list[CallCost]:
        """Retell reports `combined_cost` in CENTS, not dollars.

        Verified against its own breakdown on 2026-07-31: a 51 s call returned
        combined_cost 12.55 with product unit_prices summing to 0.2167 per
        second (0.2167 x 51 = 11.05, plus a 1.5 flat item = 12.55). At dollars
        that is $12.55 for a 51-second call, which is off by 100x and would put
        Retell two orders of magnitude above every other platform on this board.
        The conversion is applied here and recorded in `notes`.
        """
        with self._client() as client:
            response = client.post("/v2/list-calls", json={"limit": 100})
            response.raise_for_status()
            items = self._items(response.json())

        costs = []
        for call in items:
            started_ms = call.get("start_timestamp")
            if started_ms is None:
                continue
            started = float(started_ms) / 1000.0
            if not (since <= started <= until):
                continue
            block = call.get("call_cost") or {}
            platform_cents, telephony_cents = platform_cost(block)
            duration_ms = call.get("duration_ms")
            notes = ["combined_cost is reported in cents; divided by 100"]
            if telephony_cents:
                notes.append(f"telephony excluded ({telephony_cents / 100.0:.4f} "
                             f"USD): the PSTN leg is the bench's own carrier "
                             f"cost, and no other platform's figure includes one")
            costs.append(CallCost(
                vendor_call_id=str(call.get("call_id") or ""),
                cost=(platform_cents / 100.0) if platform_cents is not None else None,
                currency="USD",
                duration_s=(float(duration_ms) / 1000.0
                            if duration_ms is not None
                            else block.get("total_duration_seconds")),
                billed_s=block.get("total_duration_seconds"),
                started_at=epoch_to_iso(started),
                caller=call.get("from_number"),
                breakdown={"product_costs": block.get("product_costs") or [],
                           "combined_cost_as_reported": block.get("combined_cost"),
                           "telephony_excluded": telephony_cents,
                           "total_duration_unit_price":
                               block.get("total_duration_unit_price")},
                source="POST /v2/list-calls .call_cost.product_costs "
                       "(telephony excluded)",
                notes=tuple(notes),
            ))
        return costs


#: Products that are the CARRIER leg, not the platform. Retell resells its PSTN
#: through Twilio and folds the charge into combined_cost; nothing else on this
#: board does. Telnyx's ai-voice-assistant record excludes PSTN, Vapi reports
#: transport 0, and an ElevenLabs number is billed to your own Twilio account --
#: so leaving Retell's in would compare four platform prices against one
#: platform-plus-carrier price. The bench's own carrier (Plivo) is the
#: instrument, and an instrument is not part of what a vendor charges.
TELEPHONY_PRODUCT = "telephony"


def platform_cost(block: dict) -> tuple[float | None, float]:
    """(platform cents, telephony cents) from a Retell call_cost block.

    Falls back to combined_cost when the per-product breakdown is absent: a
    figure that includes telephony is closer to the truth than no figure, and
    the caller can tell the difference because telephony comes back 0.
    """
    products = block.get("product_costs") or []
    telephony = sum(
        float(p.get("cost") or 0)
        for p in products
        if TELEPHONY_PRODUCT in str(p.get("product") or "").lower()
    )
    combined = block.get("combined_cost")
    if not products:
        return (float(combined) if combined is not None else None), 0.0
    if combined is None:
        return sum(float(p.get("cost") or 0) for p in products) - telephony, telephony
    return float(combined) - telephony, telephony
