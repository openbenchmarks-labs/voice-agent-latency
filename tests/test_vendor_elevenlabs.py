"""ElevenLabs adapter: the nested config read, the filler gate, and the receipt.

The fixture is trimmed from a live read of the account (2026-07-30), so the field
nesting is real. What this suite proves is the comparison plumbing; whether the
committed config matches the LIVE agent is the verify_agent gate's job at bench
time, which needs the network.

The interesting test here is the filler gate. Every other vendor's gate catches
configurations that would make a measurement FAIL; this one catches a
configuration that would make a measurement LOOK GOOD -- the agent speaks a stall
phrase, that becomes the first audio, and TTFAB times the "hmm" instead of the
answer.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from vendors.base import AgentSpec, AppliedConfig, DialTarget
from vendors.registry import load_vendor_config, spec_from_config

_COMMITTED = spec_from_config(load_vendor_config("elevenlabs"))

AGENT_ID = "agent_1101kyt41p3hft4aaepgwt7rc4sj"

AGENT_JSON = {
    "agent_id": AGENT_ID,
    "name": "northwind-elevenlabs-bench",
    "version_id": "agtvrsn_8301kyqx8fqvfks9tp0rm93d0c2z",
    "branch_id": "agtbrch_0601kyqx8fn8ebxrwfnms9m5yvzk",
    "phone_numbers": [],
    "conversation_config": {
        "agent": {
            "first_message": _COMMITTED.greeting,
            "language": "en",
            "disable_first_message_interruptions": False,
            "prompt": {
                "prompt": _COMMITTED.system_prompt,
                "llm": "gemini-2.5-flash",
                "temperature": 0.0,
                "max_tokens": -1,
                "tools": [],
                "built_in_tools": {"end_call": None, "language_detection": None},
                "custom_llm": None,
            },
        },
        "asr": {"provider": "scribe_realtime", "quality": "high",
                "user_input_audio_format": "pcm_16000"},
        "tts": {"model_id": "eleven_flash_v2", "voice_id": "cjVigY5qzO86Huf0OWal",
                "optimize_streaming_latency": 3, "speed": 1.0},
        "turn": {
            "mode": "turn", "turn_model": "turn_v3", "turn_eagerness": "normal",
            "turn_timeout": 7.0, "initial_wait_time": None,
            "speculative_turn": False,
            # -1 means off. This is the state the bench requires.
            "soft_timeout_config": {"timeout_seconds": -1.0,
                                    "message": "Hhmmmm...yeah."},
        },
        "vad": {"background_voice_detection": False},
        "conversation": {"max_duration_seconds": 600},
    },
}

NUMBERS_JSON = [
    {"phone_number": "+14155550123", "phone_number_id": "pn_1",
     "provider": "twilio", "supports_inbound": True,
     "assigned_agent": {"agent_id": AGENT_ID, "agent_name": "northwind"}},
    {"phone_number": "+14155559999", "phone_number_id": "pn_2",
     "provider": "twilio", "supports_inbound": True,
     "assigned_agent": {"agent_id": "agent_other"}},
]


def _mock_vendor(agent: dict | None = None, numbers=None, block: dict | None = None):
    from vendors.elevenlabs import ElevenLabsVendor

    agent = agent if agent is not None else AGENT_JSON
    numbers = numbers if numbers is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convai/phone-numbers":
            return httpx.Response(200, json=numbers)
        if path.startswith("/v1/convai/agents/"):
            return httpx.Response(200, json=agent)
        if path == "/v1/convai/agents":
            return httpx.Response(200, json={"agents": [agent]})
        return httpx.Response(404, json={"detail": path})

    vendor = ElevenLabsVendor(block or {"agent_id": agent["agent_id"]})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.elevenlabs.io")
    return vendor


def _live_spec() -> AgentSpec:
    return spec_from_config(load_vendor_config("elevenlabs"))


def _with_turn(**turn_updates) -> dict:
    agent = copy.deepcopy(AGENT_JSON)
    agent["conversation_config"]["turn"].update(turn_updates)
    return agent


# --------------------------------------------------------------- contract ----


def test_adapter_satisfies_the_vendor_contract():
    vendor = _mock_vendor()
    assert vendor.name == "elevenlabs"
    for method in ("verify_agent", "dial_target", "applied_config"):
        assert callable(getattr(vendor, method))


def test_adapter_only_issues_get_requests():
    from vendors.elevenlabs import ElevenLabsVendor

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/v1/convai/phone-numbers":
            return httpx.Response(200, json=NUMBERS_JSON)
        return httpx.Response(200, json=AGENT_JSON)

    vendor = ElevenLabsVendor({"agent_id": AGENT_ID})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.elevenlabs.io")

    vendor.verify_agent(_live_spec())
    vendor.dial_target()
    vendor.applied_config()

    assert methods, "no requests were made"
    assert set(methods) == {"GET"}, f"non-GET traffic: {set(methods)}"


def test_auth_uses_the_xi_api_key_header():
    from harness.config import settings
    from vendors.elevenlabs import ElevenLabsVendor

    settings.elevenlabs_api_key = "sk_test"
    try:
        headers = ElevenLabsVendor({})._client().headers
    finally:
        settings.elevenlabs_api_key = None
    assert headers["xi-api-key"] == "sk_test"


# --------------------------------------------------------- the filler gate ---


def test_an_armed_filler_fails_the_gate():
    """The one gate here that catches a flattering measurement rather than a
    failed one: the stall phrase becomes the first audio, so TTFAB would time the
    filler instead of the answer -- and being real speech of real duration it
    would pass the analyzer's short-noise guard too."""
    agent = _with_turn(soft_timeout_config={"timeout_seconds": 2.0,
                                            "message": "Hhmmmm...yeah."})
    problems = _mock_vendor(agent=agent).verify_agent(_live_spec())
    assert any("soft_timeout_config is ARMED" in p for p in problems)
    assert any("Hhmmmm...yeah." in p for p in problems)


def test_zero_timeout_counts_as_armed():
    """0 seconds is the most aggressive filler, not 'off'; only -1 disables it."""
    agent = _with_turn(soft_timeout_config={"timeout_seconds": 0.0, "message": "uh"})
    assert _mock_vendor(agent=agent).filler_is_armed() is True


def test_disabled_filler_passes():
    assert _mock_vendor().filler_is_armed() is False
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_receipt_proves_the_filler_was_off_for_the_run():
    used = _mock_vendor().applied_config().defaults_used
    assert used["filler_armed"] is False
    assert used["filler_timeout_seconds"] == -1.0
    # The message is recorded even when disabled, so a later run that turned it on
    # is comparable against this one.
    assert used["filler_message"] == "Hhmmmm...yeah."


# ------------------------------------------------------------------ verify ---


def test_committed_config_matches_a_conforming_agent():
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_prompt_drift_is_reported_not_repaired():
    agent = copy.deepcopy(AGENT_JSON)
    agent["conversation_config"]["agent"]["prompt"]["prompt"] = "You are helpful."
    problems = _mock_vendor(agent=agent).verify_agent(_live_spec())
    assert any("prompt mismatch" in p for p in problems)


def test_greeting_drift_is_reported():
    agent = copy.deepcopy(AGENT_JSON)
    agent["conversation_config"]["agent"]["first_message"] = "Yo."
    problems = _mock_vendor(agent=agent).verify_agent(_live_spec())
    assert any("greeting mismatch" in p for p in problems)


def test_empty_first_message_fails_the_gate():
    agent = copy.deepcopy(AGENT_JSON)
    agent["conversation_config"]["agent"]["first_message"] = ""
    problems = _mock_vendor(agent=agent).verify_agent(_live_spec())
    assert any("no first_message" in p for p in problems)


def test_a_custom_llm_is_out_of_scope_for_the_closed_division():
    """Measuring someone else's endpoint through this platform is not measuring
    the product as shipped."""
    agent = copy.deepcopy(AGENT_JSON)
    agent["conversation_config"]["agent"]["prompt"]["custom_llm"] = {
        "url": "https://example.com/v1"}
    problems = _mock_vendor(agent=agent).verify_agent(_live_spec())
    assert any("custom_llm" in p for p in problems)


def test_pinned_stack_is_checked_when_requested():
    spec = AgentSpec(system_prompt=_COMMITTED.system_prompt,
                     greeting=_COMMITTED.greeting,
                     model="gpt-4o", stt="deepgram", tts="eleven_turbo_v2")
    problems = _mock_vendor().verify_agent(spec)
    assert any("model mismatch" in p for p in problems)
    assert any("stt mismatch" in p for p in problems)
    assert any("voice mismatch" in p for p in problems)


def test_pinned_stack_accepts_the_bare_provider_or_voice_id():
    spec = AgentSpec(system_prompt=_COMMITTED.system_prompt,
                     greeting=_COMMITTED.greeting,
                     model="gemini-2.5-flash", stt="scribe_realtime",
                     tts="cjVigY5qzO86Huf0OWal")
    assert _mock_vendor().verify_agent(spec) == []


# ------------------------------------------------------------- dial target ---


def test_dial_target_uses_the_number_assigned_to_this_agent():
    assert _mock_vendor(numbers=NUMBERS_JSON).dial_target() == DialTarget(
        kind="pstn", value="+14155550123")


def test_dial_target_prefers_the_numbers_echoed_on_the_agent():
    agent = copy.deepcopy(AGENT_JSON)
    agent["phone_numbers"] = [{"phone_number": "+14155551111"}]
    assert _mock_vendor(agent=agent).dial_target().value == "+14155551111"


def test_an_explicit_number_is_accepted_only_if_it_is_assigned_here():
    vendor = _mock_vendor(numbers=NUMBERS_JSON,
                          block={"agent_id": AGENT_ID, "number": "+14155550123"})
    assert vendor.dial_target().value == "+14155550123"


def test_an_explicit_number_assigned_elsewhere_is_refused():
    """It really happened on Telnyx (2026-07-30): the configured number reached
    a different agent while the receipt described the pinned one. Every turn
    was usable and the latencies were plausible."""
    from vendors.elevenlabs import VendorNotReady

    vendor = _mock_vendor(numbers=NUMBERS_JSON,
                          block={"agent_id": AGENT_ID, "number": "+14155559999"})
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    message = str(exc.value)
    assert "+14155559999" in message and "not assigned" in message
    assert "+14155550123" in message, "must name what IS assigned"


def test_no_number_explains_that_this_vendor_sells_none():
    """Money alone cannot unblock this one, unlike every other vendor here, so the
    error has to name the import paths instead of a purchase command."""
    from vendors.elevenlabs import VendorNotReady

    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=[NUMBERS_JSON[1]]).dial_target()
    message = str(exc.value)
    assert "does not sell numbers" in message
    assert "Twilio" in message and "SIP trunk" in message
    # Must NOT suggest setting the variable -- it no longer bypasses the check.
    assert "ELEVENLABS_PHONE_NUMBER" not in message
    # And the entanglement warning, since the obvious shortcut is to route it over
    # the bench's own carrier.
    assert "one network" in message


# ----------------------------------------------------------------- receipt ---


def test_receipt_records_the_vendors_own_choices():
    config = _mock_vendor().applied_config()
    assert isinstance(config, AppliedConfig)
    used = config.defaults_used
    assert used["model"] == "gemini-2.5-flash"
    assert used["model_temperature"] == 0.0
    assert used["stt_model"] == "scribe_realtime/high"
    assert used["voice"] == "eleven_flash_v2/cjVigY5qzO86Huf0OWal"
    assert used["tts_optimize_streaming_latency"] == 3
    assert used["endpointing"]["turn_model"] == "turn_v3"
    assert used["endpointing"]["turn_timeout_s"] == 7.0
    assert used["version_id"] == AGENT_JSON["version_id"]


def test_nothing_is_unsupported_and_that_is_the_finding():
    """This platform names its LLM, ASR provider, TTS model and temperature. The
    empty list is the direct comparison to Bland's three entries."""
    assert _mock_vendor().applied_config().unsupported == []


def test_receipt_hash_tracks_config_changes():
    first = _mock_vendor().applied_config()
    assert first.sha256 == _mock_vendor().applied_config().sha256
    assert _mock_vendor(
        agent=_with_turn(turn_timeout=3.0)).applied_config().sha256 != first.sha256


def test_receipt_is_json_serialisable():
    payload = json.loads(json.dumps(_mock_vendor().applied_config().as_dict()))
    assert payload["vendor"] == "elevenlabs"
    assert payload["raw"]["conversation_config"]["asr"]["provider"] == "scribe_realtime"


# ------------------------------------------------------------- the prompt ----


def test_all_five_vendors_are_verified_against_the_same_agent():
    specs = [spec_from_config(load_vendor_config(name))
             for name in ("telnyx", "vapi", "retell", "bland", "elevenlabs")]
    assert len({s.system_prompt for s in specs}) == 1
    assert len({s.greeting for s in specs}) == 1
