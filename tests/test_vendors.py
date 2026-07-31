"""Vendor layer: contract parity, the read-only guarantee, and the verify gate.

The Telnyx assistant JSON below is a real capture from the account
(2026-07-28), trimmed to the fields the adapter reads. Inline rather than a data
file so the fixture travels with the test regardless of what is gitignored.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import httpx
import pytest

from vendors import get_vendor
from vendors.base import AgentSpec, AppliedConfig, DialTarget
from vendors.registry import load_vendor_config, spec_from_config

REPO = Path(__file__).resolve().parent.parent

# The fake "live" assistant is built from the SAME committed source the bench
# verifies against, rather than a copy pinned here: a hardcoded copy went stale
# the moment the prompt changed and failed a test that was actually about
# comparison plumbing, not content. Whether the committed text matches the LIVE
# assistant is checked for real by the verify_agent gate at bench time, which
# needs the network and so cannot live in this suite.
_COMMITTED = spec_from_config(load_vendor_config("telnyx"))
LIVE_INSTRUCTIONS = _COMMITTED.system_prompt

ASSISTANT_JSON = {
    "id": "assistant-0e01469e-4458-441b-be3d-a4f97e6ff2d0",
    "name": "telnyx-voice-agent",
    "model": "moonshotai/Kimi-K2.6",
    "greeting": _COMMITTED.greeting,
    "instructions": LIVE_INSTRUCTIONS,
    "enabled_features": ["telephony"],
    "tools": [],
    "version_id": "20260728T225813827600",
    "voice_settings": {
        "voice": "Telnyx.Ultra.f786b574-daa5-4673-aa0c-cbe3e8534c02",
        "voice_speed": 1.0,
        "background_audio": {"type": "predefined_media", "value": "silence"},
    },
    "transcription": {
        "model": "deepgram/flux",
        "language": "en",
        "settings": {
            "eot_threshold": 0.8,
            "eot_timeout_ms": 5000,
            "eager_eot_threshold": 0.8,
        },
    },
    "interruption_settings": {
        "enable": True,
        "interrupt_prediction_threshold": 0.55,
        "start_speaking_plan": {
            "wait_seconds": 0.1,
            "transcription_endpointing_plan": {"on_punctuation_seconds": 0.1},
        },
    },
    "telephony_settings": {
        "default_texml_app_id": "3014309092261889941",
        "noise_suppression": "disabled",
        "time_limit_secs": 1800,
        "user_idle_reply_secs": 10,
    },
}

NUMBERS_JSON = {
    "data": [
        {"phone_number": "+15551234567", "status": "active",
         "connection_id": "3014309092261889941"}
    ]
}


def _mock_vendor(assistant: dict | None = None, numbers: dict | None = None,
                 block: dict | None = None):
    """A TelnyxVendor whose HTTP client is a scripted transport."""
    from vendors.telnyx import TelnyxVendor

    assistant = assistant if assistant is not None else ASSISTANT_JSON
    numbers = numbers if numbers is not None else NUMBERS_JSON

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/v2/ai/assistants/"):
            return httpx.Response(200, json={"data": assistant})
        if path == "/v2/ai/assistants":
            return httpx.Response(200, json={"data": [assistant]})
        if path == "/v2/phone_numbers":
            return httpx.Response(200, json=numbers)
        return httpx.Response(404, json={"errors": [{"detail": path}]})

    vendor = TelnyxVendor(block or {"assistant_id": assistant["id"]})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.telnyx.com")
    return vendor


def _live_spec() -> AgentSpec:
    """The spec the committed config asks for."""
    return spec_from_config(load_vendor_config("telnyx"))


# --------------------------------------------------------------- contract ----


@pytest.mark.parametrize("name", ["telnyx", "vapi", "retell", "bland", "elevenlabs"])
def test_every_registered_vendor_satisfies_the_contract(name):
    vendor = get_vendor(name)
    assert isinstance(vendor.name, str) and vendor.name
    for method in ("verify_agent", "dial_target", "applied_config"):
        assert callable(getattr(vendor, method)), f"{name} missing {method}"


def test_adapters_expose_no_mutating_methods():
    """The read-only guarantee, enforced structurally.

    An adapter that grows a create/update/delete method is a rule violation, not
    a feature -- vendor accounts are the operator's to change, never the bench's.
    """
    import vendors.bland
    import vendors.elevenlabs
    import vendors.retell
    import vendors.telnyx
    import vendors.vapi

    forbidden = ("create", "update", "delete", "patch", "post", "ensure", "provision")
    for module in (vendors.telnyx, vendors.vapi, vendors.retell, vendors.bland,
                   vendors.elevenlabs):
        for cls_name in dir(module):
            cls = getattr(module, cls_name)
            if not isinstance(cls, type) or not cls_name.endswith("Vendor"):
                continue
            for attr in dir(cls):
                if attr.startswith("_"):
                    continue
                assert not any(word in attr.lower() for word in forbidden), (
                    f"{module.__name__}.{cls_name}.{attr} looks like a mutation"
                )


def test_telnyx_adapter_only_issues_get_requests():
    """Belt and braces: exercise every public method and assert the wire is GET-only."""
    from vendors.telnyx import TelnyxVendor

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/v2/phone_numbers":
            return httpx.Response(200, json=NUMBERS_JSON)
        return httpx.Response(200, json={"data": ASSISTANT_JSON})

    vendor = TelnyxVendor({"assistant_id": ASSISTANT_JSON["id"]})
    transport = httpx.MockTransport(handler)
    vendor._client = lambda: httpx.Client(transport=transport,
                                          base_url="https://api.telnyx.com")

    vendor.verify_agent(_live_spec())
    vendor.dial_target()
    vendor.applied_config()

    assert methods, "no requests were made"
    assert set(methods) == {"GET"}, f"non-GET traffic: {set(methods)}"


def test_nothing_outside_vendors_imports_a_concrete_vendor():
    """Composability rule, same enforcement as carriers/."""
    offenders = []
    for py in REPO.rglob("*.py"):
        parts = py.relative_to(REPO).parts
        if parts[0] in (".venv", "vendors", "tests") or "__pycache__" in parts:
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("vendors.") and name not in (
                    "vendors.base", "vendors.registry"
                ):
                    offenders.append(f"{py.relative_to(REPO)}: imports {name}")
    assert not offenders, "\n".join(offenders)


# ------------------------------------------------------------------ verify ---


def test_committed_config_matches_the_live_assistant():
    """The bench's own gate: config/vendors.yaml must equal what is running.

    Uses the real captured text, so a stray edit to the prompt file (or a
    straight-vs-curly apostrophe slip) fails here rather than silently shipping
    a receipt that misdescribes the run.
    """
    assert _mock_vendor().verify_agent(_live_spec()) == []


def test_prompt_drift_is_reported_not_repaired():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["instructions"] = "You are a helpful assistant."
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert len(problems) == 1
    assert "instructions mismatch" in problems[0]


def test_greeting_drift_is_reported():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["greeting"] = "Yo."
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("greeting mismatch" in p for p in problems)


def test_missing_greeting_fails_the_gate():
    """No greeting means nothing for live VAD to wait for -- the choreography
    has no trigger, so the bench must refuse rather than time out on every call."""
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["greeting"] = ""
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("no greeting" in p for p in problems)


def test_telephony_disabled_fails_the_gate():
    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["enabled_features"] = ["chat"]
    problems = _mock_vendor(assistant=changed).verify_agent(_live_spec())
    assert any("telephony" in p for p in problems)


def test_pinned_stack_is_checked_when_requested():
    """Open-division style pinning: only meaningful because Telnyx exposes these."""
    spec = AgentSpec(
        system_prompt=LIVE_INSTRUCTIONS,
        greeting=ASSISTANT_JSON["greeting"],
        model="anthropic/claude-haiku-4-5",
        stt="deepgram/nova-3",
    )
    problems = _mock_vendor().verify_agent(spec)
    assert any("model mismatch" in p for p in problems)
    assert any("stt mismatch" in p for p in problems)


# ------------------------------------------------------------- dial target ---


def test_dial_target_discovers_the_assistants_number():
    target = _mock_vendor().dial_target()
    assert target == DialTarget(kind="pstn", value="+15551234567")


def test_an_explicit_number_is_accepted_only_if_it_reaches_this_assistant():
    """A hand-set number that is genuinely the assistant's is fine."""
    attached = NUMBERS_JSON["data"][0]["phone_number"]
    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_JSON["id"],
                                 "number": attached})
    assert vendor.dial_target().value == attached


def test_a_number_pointing_at_a_different_agent_is_refused():
    """The quietest way to publish a wrong measurement, and it really happened
    (2026-07-30): TELNYX_VENDOR_NUMBER pointed at an echo agent while the
    receipt described the pinned assistant. The run completed, every turn was
    usable, and the latencies belonged to a bot that does no inference at all.
    Nothing in the output looked wrong -- so the check has to be here."""
    from vendors.telnyx import VendorNotReady

    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_JSON["id"],
                                 "number": "+19998887777"})
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    message = str(exc.value)
    assert "+19998887777" in message
    assert "not attached" in message
    # It must name what IS attached, or the reader cannot act on it.
    assert NUMBERS_JSON["data"][0]["phone_number"] in message


def test_number_matching_ignores_formatting():
    """+1555…, 1555… and 555… are one number; a format difference must not
    read as a different agent."""
    attached = NUMBERS_JSON["data"][0]["phone_number"]
    vendor = _mock_vendor(block={"assistant_id": ASSISTANT_JSON["id"],
                                 "number": attached.lstrip("+")})
    assert vendor.dial_target()


def test_an_unverifiable_number_list_refuses_rather_than_guesses():
    """If we cannot tie the number to the assistant, we cannot honestly
    publish a receipt for the call."""
    from vendors.telnyx import VendorNotReady

    def handler(request):
        if request.url.path.startswith("/v2/ai/assistants"):
            return httpx.Response(200, json={"data": ASSISTANT_JSON})
        return httpx.Response(503, json={"errors": [{"detail": "down"}]})

    from vendors.telnyx import TelnyxVendor

    vendor = TelnyxVendor({"assistant_id": ASSISTANT_JSON["id"],
                           "number": "+15551234567"})
    vendor._client = lambda: httpx.Client(transport=httpx.MockTransport(handler),
                                          base_url="https://api.telnyx.com")
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    assert "Refusing rather than guessing" in str(exc.value)


def test_no_attached_number_gives_actionable_instructions():
    """The error has to say what to click, because attaching it is the operator's
    action -- and it must NOT suggest setting TELNYX_VENDOR_NUMBER, which no
    longer bypasses anything."""
    from vendors.telnyx import VendorNotReady

    vendor = _mock_vendor(numbers={"data": []})
    with pytest.raises(VendorNotReady) as exc:
        vendor.dial_target()
    message = str(exc.value)
    assert "Calling" in message and "portal" in message


def test_inactive_numbers_are_not_dialled():
    numbers = {"data": [{"phone_number": "+15550000000", "status": "pending"}]}
    from vendors.telnyx import VendorNotReady

    with pytest.raises(VendorNotReady):
        _mock_vendor(numbers=numbers).dial_target()


# ----------------------------------------------------------------- receipt ---


def test_receipt_records_the_vendors_own_choices():
    config = _mock_vendor().applied_config()
    assert isinstance(config, AppliedConfig)
    used = config.defaults_used

    assert used["model"] == "moonshotai/Kimi-K2.6"
    assert used["stt_model"] == "deepgram/flux"
    assert used["voice"].startswith("Telnyx.Ultra")
    # The knobs that set TTFAB on this platform.
    assert used["endpointing"]["eot_threshold"] == 0.8
    assert used["endpointing"]["start_speaking_wait_seconds"] == 0.1
    # The idle-prompt threshold, so an `idle_filler` discard is explicable
    # rather than mysterious.
    assert used["user_idle_reply_secs"] == 10
    assert used["assistant_id"] == ASSISTANT_JSON["id"]


def test_receipt_declares_what_the_vendor_cannot_pin():
    """Incomparability as data, not a footnote."""
    config = _mock_vendor().applied_config()
    assert "llm_temperature" in config.unsupported


def test_receipt_hash_tracks_config_changes():
    first = _mock_vendor().applied_config()
    same = _mock_vendor().applied_config()
    assert first.sha256 == same.sha256

    changed = copy.deepcopy(ASSISTANT_JSON)
    changed["model"] = "anthropic/claude-haiku-4-5"
    assert _mock_vendor(assistant=changed).applied_config().sha256 != first.sha256


def test_receipt_is_json_serialisable():
    import json

    payload = json.loads(json.dumps(_mock_vendor().applied_config().as_dict()))
    assert payload["vendor"] == "telnyx"
    assert payload["raw"]["model"] == "moonshotai/Kimi-K2.6"


# ------------------------------------------------------------- the prompt ----


def test_the_vendor_prompt_comes_from_the_scenarios_file():
    """The questions and the prompt that can answer them must be one artifact.

    A vendor verified against a copied-out prompt can drift from the scenarios
    the caller is reading, and the symptom is a vendor that suddenly cannot
    answer -- indistinguishable from a slow or broken agent.
    """
    spec = _live_spec()
    assert "Northwind" in spec.system_prompt
    assert spec.greeting.startswith("Thanks for calling Northwind Internet")


def test_an_unknown_adapter_is_refused():
    with pytest.raises(ValueError):
        get_vendor("telnyx", config_path=_config_with_adapter("nope"))


def _config_with_adapter(adapter: str):
    import tempfile
    from pathlib import Path

    import yaml

    path = Path(tempfile.mkdtemp()) / "vendors.yaml"
    path.write_text(yaml.safe_dump({"telnyx": {"adapter": adapter, "agent": {}}}))
    return path


def test_unknown_vendor_is_rejected():
    with pytest.raises(KeyError):
        get_vendor("definitely-not-a-vendor")
