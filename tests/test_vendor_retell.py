"""Retell adapter: the two-object read, the verify gate, and the receipt.

The fixtures are shaped from live reads of the account (2026-07-30): the agent
and its retell-llm as the API actually returns them, trimmed to the fields the
adapter reads. What this suite proves is the comparison plumbing; whether the
committed config matches the LIVE agent is the verify_agent gate's job at bench
time, which needs the network.

The thing that makes this vendor different from telnyx and vapi is that the
prompt and greeting do not live on the agent at all -- they live on the separate
response engine -- so "read the agent" is not enough to know what it will say.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from vendors.base import AgentSpec, AppliedConfig, DialTarget
from vendors.registry import load_vendor_config, spec_from_config

_COMMITTED = spec_from_config(load_vendor_config("retell"))

AGENT_ID = "agent_0ffe15ecf1e57f2503b34658ac"
LLM_ID = "llm_07a33cfd375e8e43ee628ff3195e"

AGENT_JSON = {
    "agent_id": AGENT_ID,
    "agent_name": "northwind-retell-bench",
    "voice_id": "retell-Cimo",
    "language": "en-US",
    "interruption_sensitivity": 0.9,
    "max_call_duration_ms": 3600000,
    "response_engine": {"type": "retell-llm", "llm_id": LLM_ID, "version": 0},
    "version": 0,
    "last_modification_timestamp": 1785360069151,
}

LLM_JSON = {
    "llm_id": LLM_ID,
    "model": "gpt-4.1",
    "model_temperature": None,
    "model_high_priority": False,
    "start_speaker": "agent",
    "begin_message": _COMMITTED.greeting,
    "general_prompt": _COMMITTED.system_prompt,
    "general_tools": [],
    "version": 0,
    "last_modification_timestamp": 1785360069151,
}

# The weighted shape; the singular inbound_agent_id was deprecated 2026-03-31.
NUMBERS_JSON = [
    {
        "phone_number": "+14155550123",
        "inbound_agents": [{"agent_id": AGENT_ID, "weight": 1}],
        "outbound_agent_id": None,
    },
    {
        "phone_number": "+14155559999",
        "inbound_agents": [{"agent_id": "agent_someone_else", "weight": 1}],
    },
]


def _mock_vendor(agent: dict | None = None, llm: dict | None = None,
                 numbers=None, block: dict | None = None):
    from vendors.retell import RetellVendor

    agent = agent if agent is not None else AGENT_JSON
    llm = llm if llm is not None else LLM_JSON
    numbers = numbers if numbers is not None else NUMBERS_JSON

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/list-phone-numbers":
            return httpx.Response(200, json=numbers)
        if path.startswith("/get-retell-llm/"):
            return httpx.Response(200, json=llm)
        if path.startswith("/get-agent/"):
            return httpx.Response(200, json=agent)
        if path == "/list-agents":
            return httpx.Response(200, json=[agent])
        return httpx.Response(404, json={"message": path})

    vendor = RetellVendor(block or {"agent_id": agent["agent_id"]})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.retellai.com")
    return vendor


def _live_spec() -> AgentSpec:
    return spec_from_config(load_vendor_config("retell"))


# --------------------------------------------------------------- contract ----


def test_adapter_satisfies_the_vendor_contract():
    vendor = _mock_vendor()
    assert vendor.name == "retell"
    for method in ("verify_agent", "dial_target", "applied_config"):
        assert callable(getattr(vendor, method))


def test_adapter_only_issues_get_requests():
    from vendors.retell import RetellVendor

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        path = request.url.path
        if path == "/list-phone-numbers":
            return httpx.Response(200, json=NUMBERS_JSON)
        if path.startswith("/get-retell-llm/"):
            return httpx.Response(200, json=LLM_JSON)
        return httpx.Response(200, json=AGENT_JSON)

    vendor = RetellVendor({"agent_id": AGENT_ID})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.retellai.com")

    vendor.verify_agent(_live_spec())
    vendor.dial_target()
    vendor.applied_config()

    assert methods, "no requests were made"
    assert set(methods) == {"GET"}, f"non-GET traffic: {set(methods)}"


# --------------------------------------------------------- the two objects ---


def test_prompt_is_read_from_the_response_engine_not_the_agent():
    """The distinguishing feature of this platform: an agent id alone does not
    identify what the agent will say."""
    vendor = _mock_vendor()
    assert "Northwind" in vendor.llm()["general_prompt"]
    assert "general_prompt" not in vendor.agent()


def test_a_non_retell_llm_engine_is_refused_not_reported_as_empty():
    """Refusing says 'unsupported configuration'; an empty prompt would look like
    drift and send someone hunting for a diff that does not exist."""
    from vendors.retell import VendorNotReady

    changed = copy.deepcopy(AGENT_JSON)
    changed["response_engine"] = {"type": "conversation-flow",
                                  "conversation_flow_id": "flow_x"}
    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(agent=changed).verify_agent(_live_spec())
    assert "conversation-flow" in str(exc.value)


# ------------------------------------------------------------------ verify ---


def test_committed_config_matches_a_conforming_agent():
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_prompt_drift_is_reported_not_repaired():
    changed = copy.deepcopy(LLM_JSON)
    changed["general_prompt"] = "You are a helpful assistant."
    problems = _mock_vendor(llm=changed).verify_agent(_live_spec())
    assert any("prompt mismatch" in p for p in problems)


def test_greeting_drift_is_reported():
    changed = copy.deepcopy(LLM_JSON)
    changed["begin_message"] = "Yo."
    problems = _mock_vendor(llm=changed).verify_agent(_live_spec())
    assert any("greeting mismatch" in p for p in problems)


def test_missing_begin_message_fails_the_gate():
    changed = copy.deepcopy(LLM_JSON)
    changed["begin_message"] = ""
    problems = _mock_vendor(llm=changed).verify_agent(_live_spec())
    assert any("no begin_message" in p for p in problems)


def test_start_speaker_user_fails_the_gate():
    """The blank dashboard template ships start_speaker='user'. With it the agent
    waits for the caller, so there is no greeting to detect and every call would
    time out on the greeting instead of producing a measurement."""
    changed = copy.deepcopy(LLM_JSON)
    changed["start_speaker"] = "user"
    problems = _mock_vendor(llm=changed).verify_agent(_live_spec())
    assert any("start_speaker" in p for p in problems)


def test_pinned_model_and_voice_are_checked():
    spec = AgentSpec(
        system_prompt=_COMMITTED.system_prompt,
        greeting=_COMMITTED.greeting,
        model="claude-haiku-4-5",
        tts="retell-Andrew",
    )
    problems = _mock_vendor().verify_agent(spec)
    assert any("model mismatch" in p for p in problems)
    assert any("voice mismatch" in p for p in problems)


def test_a_pinned_stt_is_refused_because_the_platform_cannot_honour_it():
    """Silently ignoring the pin would publish a receipt claiming an equalised
    stack that was never equalised."""
    spec = AgentSpec(
        system_prompt=_COMMITTED.system_prompt,
        greeting=_COMMITTED.greeting,
        stt="deepgram/nova-3",
    )
    problems = _mock_vendor().verify_agent(spec)
    assert any("cannot be pinned" in p for p in problems)


# ------------------------------------------------------------- dial target ---


def test_dial_target_discovers_the_weighted_inbound_binding():
    assert _mock_vendor().dial_target() == DialTarget(kind="pstn",
                                                      value="+14155550123")


def test_dial_target_also_reads_the_deprecated_singular_field():
    """Older numbers still come back with inbound_agent_id."""
    numbers = [{"phone_number": "+14155550123", "inbound_agent_id": AGENT_ID}]
    assert _mock_vendor(numbers=numbers).dial_target().value == "+14155550123"


def test_an_explicit_number_is_accepted_only_if_it_is_bound_inbound_here():
    vendor = _mock_vendor(block={"agent_id": AGENT_ID, "number": "+14155550123"})
    assert vendor.dial_target().value == "+14155550123"


def test_an_explicit_number_bound_to_another_agent_is_refused():
    """It really happened on Telnyx (2026-07-30): the configured number reached
    a different agent while the receipt described the pinned one. Every turn
    was usable and the latencies were plausible."""
    from vendors.retell import VendorNotReady

    vendor = _mock_vendor(block={"agent_id": AGENT_ID, "number": "+14155559999"})
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    message = str(exc.value)
    assert "+14155559999" in message and "not bound" in message
    assert "+14155550123" in message, "must name what IS bound"


def test_an_unlistable_account_refuses_rather_than_guesses():
    from vendors.retell import RetellVendor, VendorNotReady

    def handler(request: httpx.Request) -> httpx.Response:
        if "phone" in request.url.path:
            return httpx.Response(503, json={"message": "down"})
        if "llm" in request.url.path:
            return httpx.Response(200, json=LLM_JSON)
        return httpx.Response(200, json=AGENT_JSON)

    vendor = RetellVendor({"agent_id": AGENT_ID, "number": "+14155550123"})
    vendor._client = lambda: httpx.Client(transport=httpx.MockTransport(handler),
                                          base_url="https://api.retellai.com")
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    assert "Refusing rather than guessing" in str(exc.value)


def test_outbound_only_binding_is_not_dialled_and_says_why():
    """The bench dials IN. A number bound outbound-only never reaches the agent,
    and that is a one-field fix worth naming rather than 'no number found'."""
    from vendors.retell import VendorNotReady

    numbers = [{"phone_number": "+14155550123", "outbound_agent_id": AGENT_ID}]
    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=numbers).dial_target()
    message = str(exc.value)
    assert "OUTBOUND" in message and "+14155550123" in message
    # Must NOT suggest setting the variable -- it no longer bypasses the check.
    assert "RETELL_PHONE_NUMBER" not in message


def test_no_number_at_all_gives_actionable_instructions():
    from vendors.retell import VendorNotReady

    with pytest.raises(VendorNotReady) as exc:
        _mock_vendor(numbers=[]).dial_target()
    assert "Inbound Agent" in str(exc.value)


# ----------------------------------------------------------------- receipt ---


def test_receipt_carries_both_objects_and_both_versions():
    """Neither object alone describes the run: the prompt lives in the llm and
    they are versioned independently."""
    config = _mock_vendor().applied_config()
    assert isinstance(config, AppliedConfig)
    assert sorted(config.raw) == ["agent", "retell_llm"]
    used = config.defaults_used
    assert used["agent_id"] == AGENT_ID and used["llm_id"] == LLM_ID
    assert used["agent_version"] == 0 and used["llm_version"] == 0


def test_receipt_records_the_vendors_own_choices():
    used = _mock_vendor().applied_config().defaults_used
    assert used["model"] == "gpt-4.1"
    assert used["voice"] == "retell-Cimo"
    assert used["language"] == "en-US"
    assert used["start_speaker"] == "agent"
    assert used["endpointing"]["interruption_sensitivity"] == 0.9


def test_receipt_declares_that_stt_cannot_be_pinned():
    """Incomparability as data: this platform owns its STT and names no provider,
    so it cannot be stack-equalised against one that does."""
    assert "stt_provider" in _mock_vendor().applied_config().unsupported


def test_receipt_hash_tracks_changes_in_either_object():
    first = _mock_vendor().applied_config()
    assert first.sha256 == _mock_vendor().applied_config().sha256

    agent_changed = copy.deepcopy(AGENT_JSON)
    agent_changed["voice_id"] = "retell-Andrew"
    assert _mock_vendor(agent=agent_changed).applied_config().sha256 != first.sha256

    llm_changed = copy.deepcopy(LLM_JSON)
    llm_changed["model"] = "gpt-4o"
    assert _mock_vendor(llm=llm_changed).applied_config().sha256 != first.sha256


def test_receipt_is_json_serialisable():
    payload = json.loads(json.dumps(_mock_vendor().applied_config().as_dict()))
    assert payload["vendor"] == "retell"
    assert payload["raw"]["retell_llm"]["model"] == "gpt-4.1"


# ------------------------------------------------------------- the prompt ----


def test_all_vendors_are_verified_against_the_same_agent():
    """Same prompt, same greeting across platforms -- otherwise the leaderboard
    compares different tasks."""
    specs = [spec_from_config(load_vendor_config(name))
             for name in ("telnyx", "vapi", "retell")]
    assert len({s.system_prompt for s in specs}) == 1
    assert len({s.greeting for s in specs}) == 1


# --------------------------------------------------------------------------- #
# Cost: which products are the platform's, and which are the carrier's
# --------------------------------------------------------------------------- #

from vendors.retell import platform_cost  # noqa: E402


BREAKDOWN = {
    "combined_cost": 12.55,
    "product_costs": [
        {"product": "retell_voice_engine", "cost": 4.675},
        {"product": "platform_tts", "cost": 1.275},
        {"product": "gpt_4_1", "cost": 3.825},
        {"product": "us_twilio_telephony", "cost": 1.275},
        {"product": "gpt_4_1_text_testing", "cost": 1.5},
    ],
}


def test_the_carrier_leg_is_excluded_from_the_platform_price():
    """Retell resells PSTN through Twilio and folds it into combined_cost;
    nothing else on this board does. Leaving it in would compare four platform
    prices against one platform-plus-carrier price."""
    platform, telephony = platform_cost(BREAKDOWN)

    assert telephony == pytest.approx(1.275)
    assert platform == pytest.approx(11.275)


def test_every_non_telephony_product_is_kept():
    """Only the carrier leg comes out. The LLM, TTS and voice-engine charges
    are what the platform costs and all belong in the figure."""
    platform, _ = platform_cost(BREAKDOWN)

    assert platform == pytest.approx(4.675 + 1.275 + 3.825 + 1.5)


def test_a_call_with_no_breakdown_falls_back_to_the_combined_figure():
    """A figure that includes telephony beats no figure -- and telephony coming
    back 0 is how the caller tells the two situations apart."""
    platform, telephony = platform_cost({"combined_cost": 9.0})

    assert (platform, telephony) == (9.0, 0.0)


def test_a_missing_combined_total_is_summed_from_the_products():
    platform, telephony = platform_cost(
        {"product_costs": BREAKDOWN["product_costs"]})

    assert platform == pytest.approx(11.275)
    assert telephony == pytest.approx(1.275)


def test_no_cost_at_all_is_none_rather_than_zero():
    """Zero would read as free; None reads as not yet billed."""
    assert platform_cost({}) == (None, 0.0)
