"""Vapi adapter: the verify gate, dial-target discovery, and the receipt.

The fixture below is built from Vapi's live OpenAPI schema
(https://api.vapi.ai/api-json, fetched 2026-07-30) -- field names and enum
values are real, the values are representative. It is NOT a capture from a live
account: no Vapi assistant exists yet (VAPI_API_KEY is unset), so what this
suite proves is the comparison plumbing, exactly as the Telnyx suite does.
Whether the committed config matches a LIVE assistant is checked by the
verify_agent gate at bench time, which needs the network.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from vendors.base import AgentSpec, AppliedConfig, DialTarget
from vendors.registry import load_vendor_config, spec_from_config

REPO = Path(__file__).resolve().parent.parent

# Same source of truth as the vendor config, so a prompt edit cannot fail this
# suite for a reason that has nothing to do with the plumbing under test.
_COMMITTED = spec_from_config(load_vendor_config("vapi"))

ASSISTANT_ID = "d9a1f2c3-4b5e-4a67-8c90-1de2f3a4b5c6"

ASSISTANT_JSON = {
    "id": ASSISTANT_ID,
    "name": "northwind-vapi",
    "firstMessage": _COMMITTED.greeting,
    "firstMessageMode": "assistant-speaks-first",
    "firstMessageInterruptionsEnabled": False,
    "model": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "messages": [{"role": "system", "content": _COMMITTED.system_prompt}],
        "tools": [],
    },
    "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
    "voice": {"provider": "11labs", "voiceId": "21m00Tcm4TlvDq8ikWAM", "speed": 1.0},
    "startSpeakingPlan": {
        "waitSeconds": 0.4,
        "smartEndpointingEnabled": True,
        "smartEndpointingPlan": {"provider": "livekit"},
        "transcriptionEndpointingPlan": {
            "onPunctuationSeconds": 0.1,
            "onNoPunctuationSeconds": 1.5,
            "onNumberSeconds": 0.5,
        },
    },
    "stopSpeakingPlan": {"numWords": 0, "voiceSeconds": 0.2, "backoffSeconds": 1.0},
    "backgroundSound": "off",
    "maxDurationSeconds": 1800,
    "updatedAt": "2026-07-30T04:00:00.000Z",
}

# Vapi list endpoints return a bare array.
NUMBERS_JSON = [
    {
        "id": "0f1e2d3c-4b5a-4968-8776-655443332211",
        "number": "+14155550123",
        "provider": "vapi",
        "assistantId": ASSISTANT_ID,
        "status": "active",
    },
    {
        "id": "9988aabb-ccdd-4eef-8011-223344556677",
        "number": "+14155559999",
        "provider": "vapi",
        "assistantId": "some-other-assistant",
        "status": "active",
    },
]


def _mock_vendor(assistant: dict | None = None, numbers=None, block: dict | None = None):
    """A VapiVendor whose HTTP client is a scripted transport."""
    from vendors.vapi import VapiVendor

    assistant = assistant if assistant is not None else ASSISTANT_JSON
    numbers = numbers if numbers is not None else NUMBERS_JSON

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/phone-number":
            return httpx.Response(200, json=numbers)
        if path.startswith("/assistant/"):
            return httpx.Response(200, json=assistant)
        if path == "/assistant":
            return httpx.Response(200, json=[assistant])
        return httpx.Response(404, json={"message": path})

    vendor = VapiVendor(block or {"assistant_id": assistant["id"]})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.vapi.ai")
    return vendor


def _live_spec() -> AgentSpec:
    return spec_from_config(load_vendor_config("vapi"))


# --------------------------------------------------------------- contract ----


def test_adapter_satisfies_the_vendor_contract():
    vendor = _mock_vendor()
    assert vendor.name == "vapi"
    for method in ("verify_agent", "dial_target", "applied_config"):
        assert callable(getattr(vendor, method))


def test_adapter_only_issues_get_requests():
    """The read-only guarantee on the wire, for every public method."""
    from vendors.vapi import VapiVendor

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/phone-number":
            return httpx.Response(200, json=NUMBERS_JSON)
        return httpx.Response(200, json=ASSISTANT_JSON)

    vendor = VapiVendor({"assistant_id": ASSISTANT_ID})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.vapi.ai")

    vendor.verify_agent(_live_spec())
    vendor.dial_target()
    vendor.applied_config()

    assert methods, "no requests were made"
    assert set(methods) == {"GET"}, f"non-GET traffic: {set(methods)}"


# ------------------------------------------------------------------ verify ---


def test_committed_config_matches_a_conforming_assistant():
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_prompt_drift_is_reported_not_repaired():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["model"]["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("system prompt mismatch" in p for p in problems)


def test_greeting_drift_is_reported():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["firstMessage"] = "Yo."
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("greeting mismatch" in p for p in problems)


def test_missing_first_message_fails_the_gate():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["firstMessage"] = ""
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("no firstMessage" in p for p in problems)


@pytest.mark.parametrize("mode", [
    "assistant-waits-for-user",
    "assistant-speaks-first-with-model-generated-message",
])
def test_wrong_first_message_mode_fails_the_gate(mode):
    """Waiting for the user leaves VAD nothing to detect; a model-generated
    greeting differs every call, so neither the gate nor TTFG would mean anything."""
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["firstMessageMode"] = mode
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("firstMessageMode" in p for p in problems)


def test_multiple_system_messages_are_refused():
    """Which one is in force is not ours to guess, and merging them would make
    the receipt misdescribe the run."""
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["model"]["messages"].append(
        {"role": "system", "content": "Also always upsell."}
    )
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("system messages" in p for p in problems)


def test_pinned_stack_is_checked_when_requested():
    spec = AgentSpec(
        system_prompt=_COMMITTED.system_prompt,
        greeting=_COMMITTED.greeting,
        model="anthropic/claude-haiku-4-5",
        stt="assembly/universal",
        tts="cartesia/sonic",
    )
    problems = _mock_vendor().verify_agent(spec)
    assert any("model mismatch" in p for p in problems)
    assert any("stt mismatch" in p for p in problems)
    assert any("voice mismatch" in p for p in problems)


def test_pinned_stack_accepts_bare_model_name():
    """`provider/model` and the bare model name are the same pin."""
    spec = AgentSpec(
        system_prompt=_COMMITTED.system_prompt,
        greeting=_COMMITTED.greeting,
        model="gpt-4o",
        stt="nova-3",
    )
    assert _mock_vendor().verify_agent(spec) == []


# ------------------------------------------------------------- dial target ---


def test_dial_target_discovers_the_number_routed_to_this_assistant():
    """Two numbers on the account; only one has our assistantId."""
    assert _mock_vendor().dial_target() == DialTarget(kind="pstn", value="+14155550123")


def test_an_explicit_number_is_accepted_only_if_it_routes_here():
    """A hand-set number that genuinely reaches this assistant is fine."""
    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_ID,
                                 "number": "+14155550123"})
    assert vendor.dial_target().value == "+14155550123"


def test_an_explicit_number_routed_elsewhere_is_refused():
    """The quietest way to publish a wrong measurement, and it really happened
    on Telnyx (2026-07-30): the configured number reached an echo agent while
    the receipt described the pinned assistant. Every turn was usable and the
    latencies were plausible, so nothing in the output looked wrong."""
    from vendors.vapi import VendorNotReady

    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_ID,
                                 "number": "+14155559999"})  # other assistant's
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    message = str(exc.value)
    assert "+14155559999" in message
    assert "does not route" in message
    # It must name what DOES route here, or the reader cannot act on it.
    assert "+14155550123" in message


def test_explicit_number_matching_ignores_formatting():
    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_ID,
                                 "number": "14155550123"})
    assert vendor.dial_target()


def test_an_unlistable_account_refuses_rather_than_guesses():
    """If the numbers cannot be listed, the number we would dial cannot be tied
    to the config we would publish."""
    from vendors.vapi import VapiVendor, VendorNotReady

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/assistant"):
            return httpx.Response(200, json=ASSISTANT_JSON)
        return httpx.Response(503, json={"message": "down"})

    vendor = VapiVendor({"assistant_id": ASSISTANT_ID, "number": "+14155550123"})
    vendor._client = lambda: httpx.Client(transport=httpx.MockTransport(handler),
                                          base_url="https://api.vapi.ai")
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    assert "Refusing rather than guessing" in str(exc.value)


def test_number_routed_elsewhere_gives_actionable_instructions():
    from vendors.vapi import VendorNotReady

    numbers = [dict(NUMBERS_JSON[1])]  # only the other assistant's number
    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=numbers).dial_target()
    message = str(exc.value)
    assert "Inbound" in message and "dashboard" in message
    # Must NOT suggest setting the variable -- it no longer bypasses the check.
    assert "VAPI_PHONE_NUMBER" not in message


def test_inactive_numbers_are_not_dialled():
    from vendors.vapi import VendorNotReady

    numbers = [dict(NUMBERS_JSON[0], status="blocked")]
    with pytest.raises(VendorNotReady):
        _mock_vendor(numbers=numbers).dial_target()


def test_no_assistants_on_the_account_says_what_to_create():
    from vendors.vapi import VapiVendor, VendorNotReady

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    vendor = VapiVendor({})            # unpinned -> lists
    vendor.assistant_id = None
    vendor._client = lambda: httpx.Client(transport=httpx.MockTransport(handler),
                                          base_url="https://api.vapi.ai")
    with pytest.raises(VendorNotReady) as exc:
        vendor.assistant()
    assert "VAPI_ASSISTANT_ID" in str(exc.value)


# ----------------------------------------------------------------- receipt ---


def test_receipt_records_the_vendors_own_choices():
    config = _mock_vendor().applied_config()
    assert isinstance(config, AppliedConfig)
    used = config.defaults_used

    assert used["model"] == "openai/gpt-4o"
    assert used["stt_model"] == "deepgram/nova-3"
    assert used["voice"] == "11labs/21m00Tcm4TlvDq8ikWAM"
    # The knobs that set TTFAB on this platform.
    assert used["endpointing"]["wait_seconds"] == 0.4
    assert used["endpointing"]["on_no_punctuation_seconds"] == 1.5
    assert used["endpointing"]["smart_endpointing_provider"] == "livekit"
    assert used["first_message_mode"] == "assistant-speaks-first"
    assert used["assistant_id"] == ASSISTANT_ID
    # Plans were echoed by the API here, so that is what the receipt attributes.
    assert used["endpointing_source"] == {
        "start_speaking_plan": "api", "stop_speaking_plan": "api",
    }


def test_unset_speaking_plans_record_the_documented_default_and_say_so():
    """Vapi returns null for plans it never overrode while still applying its
    documented defaults. Recording null would describe the knobs that set TTFAB
    as unknown; recording the number without provenance would pass an assumption
    off as an observation."""
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed.pop("startSpeakingPlan")
    changed.pop("stopSpeakingPlan")
    used = _mock_vendor(assistant=changed).applied_config().defaults_used

    assert used["endpointing"]["wait_seconds"] == 0.4
    assert used["endpointing"]["on_no_punctuation_seconds"] == 1.5
    assert used["endpointing"]["stop_backoff_seconds"] == 1.0
    assert used["endpointing_source"] == {
        "start_speaking_plan": "vapi-documented-default",
        "stop_speaking_plan": "vapi-documented-default",
    }


def test_receipt_records_the_temperature_telnyx_cannot_expose():
    """The asymmetry is data, not a footnote: reply determinism is pinnable here
    and is not pinnable on Telnyx."""
    assert _mock_vendor().applied_config().defaults_used["model_temperature"] == 0.7


def test_receipt_declares_what_this_vendor_cannot_pin():
    config = _mock_vendor().applied_config()
    assert "idle_reply_threshold" in config.unsupported
    # ...and records no idle threshold, so an `idle_filler` discard on this
    # vendor cannot be blamed on a configured idle reply.
    assert config.defaults_used["user_idle_reply_secs"] is None


def test_receipt_hash_tracks_config_changes():
    first = _mock_vendor().applied_config()
    assert first.sha256 == _mock_vendor().applied_config().sha256

    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["startSpeakingPlan"]["waitSeconds"] = 0.9
    assert _mock_vendor(assistant=changed).applied_config().sha256 != first.sha256


def test_receipt_is_json_serialisable():
    payload = json.loads(json.dumps(_mock_vendor().applied_config().as_dict()))
    assert payload["vendor"] == "vapi"
    assert payload["raw"]["model"]["model"] == "gpt-4o"


# ------------------------------------------------------------- the prompt ----


def test_the_vendor_prompt_comes_from_the_scenarios_file():
    spec = _live_spec()
    assert "Northwind" in spec.system_prompt
    assert spec.greeting.startswith("Thanks for calling Northwind Internet")


def test_both_vendors_are_verified_against_the_same_agent():
    """Same prompt, same greeting, different platforms -- otherwise the
    leaderboard compares two different tasks."""
    telnyx = spec_from_config(load_vendor_config("telnyx"))
    vapi = spec_from_config(load_vendor_config("vapi"))
    assert telnyx.system_prompt == vapi.system_prompt
    assert telnyx.greeting == vapi.greeting
