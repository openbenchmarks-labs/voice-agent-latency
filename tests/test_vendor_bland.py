"""Bland adapter: the number-is-the-agent read, the verify gate, and the receipt.

Fixtures are shaped from the documented GET /v1/inbound response and a live read
of the account's agent schema (2026-07-30). What this suite proves is the
comparison plumbing; whether the committed config matches the LIVE number is the
verify_agent gate's job at bench time, which needs the network.

What makes this vendor different: there is no agent object at all. The prompt and
greeting live on the phone number, so the number is the identity, and an account
with no number has nothing to verify -- a state the other adapters cannot be in.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from vendors.base import AgentSpec, AppliedConfig, DialTarget
from vendors.registry import load_vendor_config, spec_from_config

_COMMITTED = spec_from_config(load_vendor_config("bland"))

NUMBER = "+14155550123"

CONFIG_JSON = {
    "created_at": "2026-07-30T04:00:00.000Z",
    "phone_number": NUMBER,
    "prompt": _COMMITTED.system_prompt,
    "first_sentence": _COMMITTED.greeting,
    "model": "enhanced",
    "voice": "Valentine Experimental",
    "temperature": None,
    "interruption_threshold": None,
    "block_interruptions": False,
    "reduce_latency": True,
    "noise_cancellation": True,
    "silence_end_message": None,
    "max_duration": 30,
    "record": False,
    "webhook": None,
    "tools": None,
    "pathway_id": None,
}


def _mock_vendor(config: dict | None = None, numbers=None, block: dict | None = None,
                 detail_404: bool = False):
    from vendors.bland import BlandVendor

    config = config if config is not None else CONFIG_JSON
    listing = numbers if numbers is not None else [config]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/inbound":
            return httpx.Response(200, json={"inbound_numbers": listing})
        if path.startswith("/v1/inbound/"):
            if detail_404:
                return httpx.Response(404, json={"status": "error",
                                                 "message": "Number not found"})
            return httpx.Response(200, json={"inbound_number": config})
        return httpx.Response(404, json={"status": "error", "message": path})

    vendor = BlandVendor(block if block is not None else {})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.bland.ai")
    return vendor


def _live_spec() -> AgentSpec:
    return spec_from_config(load_vendor_config("bland"))


# --------------------------------------------------------------- contract ----


def test_adapter_satisfies_the_vendor_contract():
    vendor = _mock_vendor()
    assert vendor.name == "bland"
    for method in ("verify_agent", "dial_target", "applied_config"):
        assert callable(getattr(vendor, method))


def test_adapter_only_issues_get_requests():
    from vendors.bland import BlandVendor

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/v1/inbound":
            return httpx.Response(200, json={"inbound_numbers": [CONFIG_JSON]})
        return httpx.Response(200, json={"inbound_number": CONFIG_JSON})

    vendor = BlandVendor({})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.bland.ai")

    vendor.verify_agent(_live_spec())
    vendor.dial_target()
    vendor.applied_config()

    assert methods, "no requests were made"
    assert set(methods) == {"GET"}, f"non-GET traffic: {set(methods)}"


def test_auth_header_is_the_raw_key_not_a_bearer_token():
    """Bland rejects nothing but also ignores Bearer; the documented form is raw.
    Pinned because a silent auth change would look like an empty account."""
    from harness.config import settings
    from vendors.bland import BlandVendor

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"inbound_numbers": [CONFIG_JSON]})

    settings.bland_api_key = "org_testkey"
    try:
        vendor = BlandVendor({})
        real_client = vendor._client
        vendor._client = lambda: httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://api.bland.ai",
            headers=real_client().headers,
        )
        vendor.config()
    finally:
        settings.bland_api_key = None
    assert seen["authorization"] == "org_testkey"


# ------------------------------------------------- the number is the agent ---


def test_an_account_with_no_number_has_nothing_to_verify_and_says_the_cost():
    """A state the other vendors cannot be in: no number means no agent at all.
    The error has to say so, and say what it costs, or it reads as a bug."""
    from vendors.bland import VendorNotReady

    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=[]).config()
    message = str(exc.value)
    assert "number IS the agent" in message
    assert "$15/month" in message
    assert "BLAND_PHONE_NUMBER" in message


def test_multiple_numbers_are_refused_rather_than_guessed():
    from vendors.bland import VendorNotReady

    other = dict(CONFIG_JSON, phone_number="+14155559999")
    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=[CONFIG_JSON, other]).config()
    assert "BLAND_PHONE_NUMBER" in str(exc.value)


def test_pinned_number_selects_among_several_ignoring_plus_prefix():
    other = dict(CONFIG_JSON, phone_number="+14155559999")
    vendor = _mock_vendor(numbers=[other, CONFIG_JSON],
                          block={"number": "14155550123"})
    assert vendor.config()["phone_number"] == NUMBER


def test_a_pinned_number_not_on_the_account_is_named():
    from vendors.bland import VendorNotReady

    vendor = _mock_vendor(block={"number": "+19998887777"})
    with pytest.raises(VendorNotReady) as exc:
        vendor.config()
    assert "+19998887777" in str(exc.value)


def test_detail_endpoint_failure_falls_back_to_the_list_record():
    """The list is documented to carry the full config, so a 404 on the detail
    view must not take the whole run down."""
    vendor = _mock_vendor(detail_404=True)
    assert vendor.config()["phone_number"] == NUMBER
    assert vendor.verify_agent(_live_spec()) == []


def test_a_pathway_driven_number_is_refused_not_reported_as_empty():
    """A pathway replaces the prompt with a graph this adapter cannot read;
    reporting an empty prompt would look like drift and send someone hunting for
    a mismatch that does not exist."""
    from vendors.bland import VendorNotReady

    changed = dict(CONFIG_JSON, pathway_id="path_123", prompt=None)
    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(config=changed).verify_agent(_live_spec())
    assert "pathway" in str(exc.value)


# ------------------------------------------------------------------ verify ---


def test_committed_config_matches_a_conforming_number():
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_prompt_drift_is_reported_not_repaired():
    changed = dict(CONFIG_JSON, prompt="You are a helpful assistant.")
    problems = _mock_vendor(config=changed).verify_agent(_live_spec())
    assert any("prompt mismatch" in p for p in problems)


def test_greeting_drift_is_reported():
    changed = dict(CONFIG_JSON, first_sentence="Yo.")
    problems = _mock_vendor(config=changed).verify_agent(_live_spec())
    assert any("greeting mismatch" in p for p in problems)


def test_missing_first_sentence_fails_the_gate():
    """Without it Bland lets the model open the call, so the greeting differs
    every call and there is no fixed utterance for live VAD to wait for."""
    changed = dict(CONFIG_JSON, first_sentence=None)
    problems = _mock_vendor(config=changed).verify_agent(_live_spec())
    assert any("no first_sentence" in p for p in problems)


def test_pinned_voice_is_checked():
    spec = AgentSpec(system_prompt=_COMMITTED.system_prompt,
                     greeting=_COMMITTED.greeting, tts="Nova")
    assert any("voice mismatch" in p
               for p in _mock_vendor().verify_agent(spec))


def test_a_pinned_model_reports_that_the_platform_names_tiers_not_models():
    spec = AgentSpec(system_prompt=_COMMITTED.system_prompt,
                     greeting=_COMMITTED.greeting, model="gpt-4.1")
    problems = _mock_vendor().verify_agent(spec)
    assert any("tiers, not LLM identities" in p for p in problems)


def test_a_pinned_stt_is_refused_because_the_platform_exposes_none():
    spec = AgentSpec(system_prompt=_COMMITTED.system_prompt,
                     greeting=_COMMITTED.greeting, stt="deepgram/nova-3")
    problems = _mock_vendor().verify_agent(spec)
    assert any("cannot be pinned" in p for p in problems)


# ------------------------------------------------------------- dial target ---


def test_dial_target_is_the_number_itself():
    assert _mock_vendor().dial_target() == DialTarget(kind="pstn", value=NUMBER)


def test_dial_target_normalises_a_missing_plus():
    changed = dict(CONFIG_JSON, phone_number="14155550123")
    assert _mock_vendor(config=changed).dial_target().value == NUMBER


# ----------------------------------------------------------------- receipt ---


def test_receipt_records_the_vendors_own_choices():
    used = _mock_vendor().applied_config().defaults_used
    assert used["model_tier"] == "enhanced"
    assert used["voice"] == "Valentine Experimental"
    assert used["phone_number"] == NUMBER
    assert used["endpointing"]["reduce_latency"] is True
    assert used["max_duration_min"] == 30


def test_receipt_declares_the_three_things_this_platform_hides():
    """Incomparability as data. The tier one matters most: 'enhanced' can be
    re-pointed at a different LLM without anything in the receipt changing."""
    config = _mock_vendor().applied_config()
    assert isinstance(config, AppliedConfig)
    for item in ("llm_identity", "stt_provider", "config_version"):
        assert item in config.unsupported


def test_receipt_hash_tracks_config_changes():
    first = _mock_vendor().applied_config()
    assert first.sha256 == _mock_vendor().applied_config().sha256

    changed = dict(CONFIG_JSON, interruption_threshold=200)
    assert _mock_vendor(config=changed).applied_config().sha256 != first.sha256


def test_receipt_is_json_serialisable():
    payload = json.loads(json.dumps(_mock_vendor().applied_config().as_dict()))
    assert payload["vendor"] == "bland"
    assert payload["raw"]["model"] == "enhanced"


# ------------------------------------------------------------- the prompt ----


def test_all_four_vendors_are_verified_against_the_same_agent():
    specs = [spec_from_config(load_vendor_config(name))
             for name in ("telnyx", "vapi", "retell", "bland")]
    assert len({s.system_prompt for s in specs}) == 1
    assert len({s.greeting for s in specs}) == 1
