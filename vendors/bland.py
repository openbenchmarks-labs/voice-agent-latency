"""Bland AI as a vendor under test.

Bland is the third distinct shape in this directory, and the one that breaks the
pattern the others share. There is no agent object behind an inbound call: the
configuration -- prompt, first_sentence, voice, model, interruption knobs --
lives ON the phone number itself (POST /v1/inbound/{number} takes them inline
and accepts no agent_id). The benchable unit is therefore the number, so
BLAND_PHONE_NUMBER is what pins this vendor, where the others pin an agent id.

That has a practical consequence worth stating: a Bland inbound number is a
$15/month subscription, so unlike Vapi (free number) or Retell (~$2) there is no
way to have a verifiable Bland agent without an active paid number. There is no
free tier of "the thing under test" here.

READ-ONLY by contract. This module issues GETs and nothing else. `verify_agent`
reports mismatches for a human to fix; it never repairs them.

Facts from the API and docs (2026-07-30):
  GET /v1/inbound            -> {"inbound_numbers": [ {phone_number, prompt,
                                first_sentence, voice_id, interruption_threshold,
                                max_duration, record, ...} ]}
  GET /v1/inbound/{number}   -> that number's configuration
  auth: a raw `authorization: <key>` header, no Bearer prefix
  model values are Bland's own tiers ("enhanced", "base"), not LLM identities

Two things this platform does not expose, both recorded in the receipt rather
than footnoted: which LLM is behind a tier name, and anything about the STT.
"""

from __future__ import annotations

import logging

import httpx

from harness.config import settings

from .base import (AgentSpec, AppliedConfig, CallCost, DialTarget, digits,
                    iso_to_epoch)

log = logging.getLogger(__name__)

API = "https://api.bland.ai"


class VendorNotReady(RuntimeError):
    """Raised with human-actionable instructions, never auto-repaired."""


class BlandVendor:
    name = "bland"

    def __init__(self, block: dict | None = None):
        self.block = block or {}
        self.number = self.block.get("number") or settings.bland_phone_number
        self._config: dict | None = None

    # ------------------------------------------------------------------ fetch

    def _client(self) -> httpx.Client:
        if not settings.bland_api_key:
            raise VendorNotReady("BLAND_API_KEY is not set in .env")
        return httpx.Client(
            base_url=API,
            # Bland takes the key raw, not as a Bearer token.
            headers={"authorization": settings.bland_api_key},
            timeout=15.0,
            # Bland is migrating endpoint paths and answers old ones with 308s.
            follow_redirects=True,
        )

    @staticmethod
    def _numbers(payload) -> list[dict]:
        if isinstance(payload, dict):
            return payload.get("inbound_numbers") or payload.get("data") or []
        if isinstance(payload, list):
            return payload
        return []

    def config(self, refresh: bool = False) -> dict:
        """The inbound number's live configuration -- the agent, on this platform.

        Cached per adapter instance.
        """
        if self._config is not None and not refresh:
            return self._config

        with self._client() as client:
            response = client.get("/v1/inbound")
            if response.status_code >= 300:
                raise VendorNotReady(
                    f"cannot list inbound numbers: {response.status_code} "
                    f"{response.text[:200]}"
                )
            numbers = self._numbers(response.json())

            if not numbers:
                raise VendorNotReady(
                    "this Bland account has no inbound numbers, and on this platform "
                    "the number IS the agent -- its prompt and greeting live on the "
                    "number, not on a separate agent object. There is nothing to "
                    "verify or dial until one exists.\n"
                    "  A Bland inbound number is a $15/month subscription: buy one "
                    "with tools/setup_bland_agent.py --buy-number --area-code <NNN>, "
                    "or add one you own through Bland's BYO-Twilio flow.\n"
                    "  Then set BLAND_PHONE_NUMBER in .env."
                )

            if self.number:
                wanted = _digits(self.number)
                match = [n for n in numbers if _digits(n.get("phone_number")) == wanted]
                if not match:
                    have = ", ".join(str(n.get("phone_number")) for n in numbers)
                    raise VendorNotReady(
                        f"BLAND_PHONE_NUMBER {self.number} is not an inbound number "
                        f"on this account. Have: {have}"
                    )
                data = match[0]
            elif len(numbers) > 1:
                have = ", ".join(str(n.get("phone_number")) for n in numbers)
                raise VendorNotReady(
                    "multiple inbound numbers on this account; pin the one under "
                    "test with BLAND_PHONE_NUMBER in .env or `number` in "
                    f"config/vendors.yaml. Have: {have}"
                )
            else:
                data = numbers[0]

            # The list view is documented to carry the full configuration, but ask
            # for the number's own record when it is available: the list is a
            # summary and a receipt should quote the authoritative source.
            detail = client.get(f"/v1/inbound/{_digits(data.get('phone_number'))}")
            if detail.status_code < 300:
                payload = detail.json()
                if isinstance(payload, dict):
                    data = payload.get("inbound_number") or payload.get("data") or data

        self._config = data
        return data

    # ----------------------------------------------------------------- verify

    def verify_agent(self, spec: AgentSpec) -> list[str]:
        """Mismatches between what the bench requires and what is live."""
        problems: list[str] = []
        config = self.config()

        # A pathway replaces the prompt with a graph whose text this adapter
        # cannot read, so refuse rather than report an empty-prompt "diff" that
        # would send someone hunting for a mismatch that does not exist.
        if config.get("pathway_id"):
            raise VendorNotReady(
                f"number {config.get('phone_number')} is driven by pathway "
                f"{config['pathway_id']}, whose conversation text this adapter "
                "cannot read or verify. Point the number at a prompt, or extend "
                "vendors/bland.py to walk pathways."
            )

        live_greeting = (config.get("first_sentence") or "").strip()
        want_greeting = spec.greeting.strip()
        if live_greeting != want_greeting:
            problems.append(
                f"greeting mismatch (first_sentence)\n"
                f"    live: {live_greeting!r}\n    want: {want_greeting!r}"
            )

        live_prompt = (config.get("prompt") or "").strip()
        want_prompt = spec.system_prompt.strip()
        if live_prompt != want_prompt:
            problems.append(
                f"prompt mismatch\n    live: {live_prompt!r}\n    want: {want_prompt!r}"
            )

        if not live_greeting:
            problems.append(
                "number has no first_sentence -- without it Bland lets the model "
                "open the call, so the greeting differs every call and live VAD has "
                "no fixed utterance to wait for. Set first_sentence to the bench "
                "greeting."
            )

        if spec.model:
            # Bland's `model` is a tier name of its own ("enhanced", "base"), so a
            # pin naming an LLM cannot be satisfied or even compared here.
            live_model = config.get("model")
            if live_model != spec.model:
                problems.append(
                    f"model mismatch: live tier {live_model!r}, want {spec.model!r} "
                    "(note: Bland exposes tiers, not LLM identities -- see "
                    "applied_config().unsupported)"
                )
        if spec.tts:
            live_voice = config.get("voice") or config.get("voice_id")
            if str(live_voice) != str(spec.tts):
                problems.append(
                    f"voice mismatch: live {live_voice!r}, want {spec.tts!r}"
                )
        if spec.stt:
            problems.append(
                f"stt {spec.stt!r} cannot be pinned on this platform: Bland exposes "
                "no STT provider or model (see applied_config().unsupported)"
            )

        return problems

    # ------------------------------------------------------------ dial target

    def dial_target(self) -> DialTarget:
        """The number that reaches the agent -- which here is the agent itself."""
        config = self.config()
        number = config.get("phone_number")
        if not number:
            raise VendorNotReady(
                "the inbound number record has no `phone_number` field; set "
                "BLAND_PHONE_NUMBER in .env"
            )
        return DialTarget(kind="pstn", value=_e164(number))

    # ---------------------------------------------------------------- receipt

    def applied_config(self) -> AppliedConfig:
        """The config receipt that ships with every measurement."""
        config = self.config()

        defaults_used = {
            # A tier, not an LLM name. Recorded as what it is.
            "model_tier": config.get("model"),
            "temperature": config.get("temperature"),
            "voice": config.get("voice") or config.get("voice_id"),
            "voice_settings": config.get("voice_settings"),
            "language": config.get("language"),
            "background_track": config.get("background_track"),
            "noise_cancellation": config.get("noise_cancellation"),
            # The knobs that most directly set TTFAB on this platform. A null means
            # not overridden -- the platform default is in force, which is the
            # closed division's answer anyway.
            "endpointing": {
                "interruption_threshold": config.get("interruption_threshold"),
                "block_interruptions": config.get("block_interruptions"),
                # Documented as a latency/quality trade-off on inbound numbers, so
                # it belongs next to the endpointing knobs rather than buried.
                "reduce_latency": config.get("reduce_latency"),
                "interruptibility": config.get("interruptibility"),
                "resumption_speed": config.get("resumption_speed"),
                "keywords": config.get("keywords"),
            },
            # Bland speaks this after a silence, which is the `idle_filler` hazard
            # the analyzer discards on -- recorded so the discard is explicable.
            "silence_end_message": config.get("silence_end_message"),
            "max_duration_min": config.get("max_duration"),
            "record": config.get("record"),
            "tools": [
                t.get("name") or t.get("type") for t in (config.get("tools") or [])
            ],
            "pathway_id": config.get("pathway_id"),
            # There is no agent id to pin on this platform; the number is the
            # identity, and `created_at` is the only version-ish field it carries.
            "phone_number": config.get("phone_number"),
            "created_at": config.get("created_at"),
        }

        unsupported = [
            # Bland names capability tiers ("enhanced"), never the model behind
            # them, so it cannot be stack-equalised against a vendor that does --
            # and a tier can be re-pointed at a different LLM without the receipt
            # changing, which is exactly the kind of thing the receipt exists to
            # make visible.
            "llm_identity",
            # No STT provider or model is exposed anywhere in the configuration.
            "stt_provider",
            # No per-config version or revision field exists, so a silent edit to
            # the number's prompt cannot be detected from the receipt alone the way
            # a version_id change would show it on the other platforms.
            "config_version",
        ]

        return AppliedConfig(
            vendor=self.name,
            normalized={
                "greeting": (config.get("first_sentence") or "").strip(),
                "instructions": (config.get("prompt") or "").strip(),
                "model_requested": self.block.get("agent", {}).get("model"),
                "stt_requested": self.block.get("agent", {}).get("stt"),
                "tts_requested": self.block.get("agent", {}).get("tts"),
            },
            raw=config,
            defaults_used=defaults_used,
            unsupported=unsupported,
        )

    # ----------------------------------------------------------------- cost
    def call_costs(self, since: float, until: float) -> list[CallCost]:
        """Bland reports `price` in dollars and `call_length` in MINUTES.

        The unit mismatch is the trap: every other platform on this board
        reports seconds, so passing call_length through unconverted would make
        Bland look 60x cheaper per minute.
        """
        with self._client() as client:
            response = client.get("/v1/calls", params={"limit": 100})
            response.raise_for_status()
            payload = response.json()
        items = (payload.get("calls") if isinstance(payload, dict)
                 else payload) or []

        costs = []
        for call in items:
            started = iso_to_epoch(call.get("started_at"))
            if started is None or not (since <= started <= until):
                continue
            minutes = call.get("call_length")
            costs.append(CallCost(
                vendor_call_id=str(call.get("call_id") or call.get("c_id") or ""),
                cost=call.get("price"),
                currency="USD",
                duration_s=(float(minutes) * 60.0) if minutes is not None else None,
                billed_s=None,
                started_at=call.get("started_at"),
                caller=call.get("from"),
                breakdown={},
                source="GET /v1/calls .price",
                notes=("call_length is reported in minutes; converted to seconds",
                       "Bland publishes no cost breakdown -- the figure is a "
                       "single all-in price"),
            ))
        return costs


#: Bland accepts a number with or without the + prefix; compare on digits.
#: Shared with every other adapter that checks a configured number.
_digits = digits

def _e164(number: str) -> str:
    text = str(number).strip()
    return text if text.startswith("+") else f"+{_digits(text)}"
