#!/usr/bin/env python3
"""Provision the Vapi agent under test, from the committed config.

This is a TOOL, not part of the bench. The vendor adapters are read-only by
contract (tests/test_vendors.py enforces it structurally) because a bench that
can rewrite the thing it measures cannot publish a trustworthy receipt. Setup is
a separate, explicit, human-invoked action -- this script -- and it is the only
place in the voice tree that writes to a vendor account.

What it does, idempotently:
  1. finds or creates an assistant named --name
  2. leaves the stack (model/voice/transcriber) at whatever Vapi gives a new
     signup, then patches ONLY the system prompt into that default model object
     -- the closed division measures defaults as shipped, so choosing a model
     here would change what is being measured
  3. verifies greeting + firstMessageMode are what the bench's gate requires
  4. attaches a phone number (creating a free US one if the account has none)
  5. re-reads everything and runs the bench's own verify gate

Afterwards `python -m harness.bench --vendor vapi` can verify and dial it.

Usage:
    .venv/bin/python tools/setup_vapi_agent.py [--name northwind-vapi-bench]
        [--area-code 415] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

from harness.config import settings                       # noqa: E402
from vendors import get_vendor                            # noqa: E402
from vendors.registry import load_vendor_config, spec_from_config  # noqa: E402

API = "https://api.vapi.ai"
DEFAULT_NAME = "northwind-vapi-bench"
REQUIRED_FIRST_MESSAGE_MODE = "assistant-speaks-first"

# Vapi's API, unlike its dashboard, applies NO default stack: an assistant
# created over HTTP comes back with model/transcriber/voice all null, and a
# patch that omits model.provider is rejected. So the defaults have to be
# materialised here, and they are copied verbatim from what the dashboard's
# blank assistant template produced, read 2026-07-30. (Create a blank assistant
# in the dashboard and read it back to re-derive them for yourself.)
# That is the closed-division intent (what a new signup gets) made explicit
# rather than silently chosen by us; the receipt records these as the defaults
# in force, and `--stack-from <assistant_id>` re-derives them if Vapi moves.
BLANK_TEMPLATE_DEFAULTS = {
    "model": {"provider": "openai", "model": "gpt-4.1"},
    "transcriber": {"provider": "soniox", "model": "stt-rt-v5", "language": "en"},
    "voice": {"provider": "vapi", "voiceId": "Elliot"},
}


def client() -> httpx.Client:
    if not settings.vapi_api_key:
        sys.exit("VAPI_API_KEY is not set in .env")
    return httpx.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
        timeout=30.0,
    )


def items(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("data") or payload.get("results") or []
    return []


def check(response: httpx.Response, what: str) -> dict:
    if response.status_code >= 300:
        sys.exit(f"{what} failed: {response.status_code} {response.text[:400]}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--area-code", default=None,
                        help="Desired area code for a new free US number")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing")
    parser.add_argument("--stack-from", default=None, metavar="ASSISTANT_ID",
                        help="Re-derive the default stack from a dashboard-created "
                             "assistant instead of the recorded template defaults")
    args = parser.parse_args()

    spec = spec_from_config(load_vendor_config("vapi"))
    if not spec.system_prompt or not spec.greeting:
        sys.exit("vendors.yaml gave an empty prompt or greeting")
    print(f"agent under test: {args.name}")
    print(f"  greeting: {spec.greeting!r}")
    print(f"  prompt:   {len(spec.system_prompt)} chars from the scenarios file")
    if args.dry_run:
        print("\n--dry-run: no writes")

    with client() as api:
        existing = [
            a for a in items(check(api.get("/assistant"), "list assistants"))
            if a.get("name") == args.name
        ]
        if len(existing) > 1:
            sys.exit(f"{len(existing)} assistants named {args.name!r}; "
                     "delete the duplicates in the dashboard first")

        if existing:
            assistant = existing[0]
            print(f"\nfound assistant {assistant['id']}")
        else:
            if args.dry_run:
                print("\nwould create the assistant (Vapi picks the default stack)")
                return
            # Created minimal on purpose: whatever model/voice/transcriber Vapi
            # assigns IS the product under test. We never choose them.
            assistant = check(api.post("/assistant", json={
                "name": args.name,
                "firstMessage": spec.greeting,
                "firstMessageMode": REQUIRED_FIRST_MESSAGE_MODE,
            }), "create assistant")
            print(f"\ncreated assistant {assistant['id']}")

        defaults = dict(BLANK_TEMPLATE_DEFAULTS)
        if args.stack_from:
            source = check(api.get(f"/assistant/{args.stack_from}"), "read stack source")
            source_model = source.get("model") or {}
            defaults = {
                "model": {k: v for k, v in source_model.items()
                          if k in ("provider", "model")},
                "transcriber": source.get("transcriber") or {},
                "voice": source.get("voice") or {},
            }
            # Vapi rejects a model patch that omits `provider`, and a source
            # assistant that never had a stack would silently produce one --
            # failing later, on the patch, with a less obvious message.
            if not defaults["model"].get("provider"):
                sys.exit(f"--stack-from {args.stack_from} has no model.provider; "
                         "it has no stack to copy. Pick an assistant created in "
                         "the dashboard.")
            print(f"  stack derived from {args.stack_from}")

        model = dict(assistant.get("model") or {})
        if not model.get("provider"):
            # API-created assistants come back with no stack at all; fill it with
            # the platform's template defaults rather than a choice of our own.
            model = {**defaults["model"], **model}
            print(f"  applying template defaults: "
                  f"model={defaults['model']['provider']}/{defaults['model']['model']} "
                  f"transcriber={defaults['transcriber'].get('provider')}/"
                  f"{defaults['transcriber'].get('model')} "
                  f"voice={defaults['voice'].get('provider')}/"
                  f"{defaults['voice'].get('voiceId')}")
        else:
            print(f"  live stack: model={model.get('provider')}/{model.get('model')} "
                  f"transcriber={(assistant.get('transcriber') or {}).get('provider')} "
                  f"voice={(assistant.get('voice') or {}).get('provider')}")

        # Patch only what the bench requires, preserving the default stack.
        messages = [m for m in (model.get("messages") or [])
                    if (m.get("role") or "").lower() != "system"]
        messages.insert(0, {"role": "system", "content": spec.system_prompt})
        patch: dict = {}
        if not (assistant.get("transcriber") or {}).get("provider"):
            patch["transcriber"] = defaults["transcriber"]
        if not (assistant.get("voice") or {}).get("provider"):
            patch["voice"] = defaults["voice"]
        current_system = next(
            (m.get("content") for m in (model.get("messages") or [])
             if (m.get("role") or "").lower() == "system"), None)
        if (current_system or "").strip() != spec.system_prompt.strip():
            patch["model"] = {**model, "messages": messages}
        if (assistant.get("firstMessage") or "").strip() != spec.greeting.strip():
            patch["firstMessage"] = spec.greeting
        if assistant.get("firstMessageMode") != REQUIRED_FIRST_MESSAGE_MODE:
            patch["firstMessageMode"] = REQUIRED_FIRST_MESSAGE_MODE

        if patch and not args.dry_run:
            assistant = check(
                api.patch(f"/assistant/{assistant['id']}", json=patch),
                "patch assistant",
            )
            print(f"  patched: {', '.join(sorted(patch))}")
        elif patch:
            print(f"  would patch: {', '.join(sorted(patch))}")
        else:
            print("  already matches the committed config")

        # ------------------------------------------------------------- number
        numbers = items(check(api.get("/phone-number", params={"limit": 100}),
                              "list numbers"))
        attached = [n for n in numbers if n.get("assistantId") == assistant["id"]]
        if attached:
            print(f"\nnumber already routed: {attached[0].get('number')}")
            number = attached[0]
        elif args.dry_run:
            print("\nwould attach a number (creating a free US one if needed)")
            return
        else:
            free = [n for n in numbers if not n.get("assistantId")]
            if free:
                number = check(
                    api.patch(f"/phone-number/{free[0]['id']}",
                              json={"assistantId": assistant["id"]}),
                    "attach number",
                )
                print(f"\nattached existing number {number.get('number')}")
            else:
                body = {"provider": "vapi", "assistantId": assistant["id"]}
                if args.area_code:
                    body["numberDesiredAreaCode"] = args.area_code
                number = check(api.post("/phone-number", json=body),
                               "create number")
                print(f"\ncreated free number {number.get('number')}")
                for _ in range(10):
                    if (number.get("status") or "active") == "active":
                        break
                    time.sleep(3)
                    number = check(api.get(f"/phone-number/{number['id']}"),
                                   "poll number")
                print(f"  status: {number.get('status')}")

    # ------------------------------------------------------------------ verify
    # The bench's own gate, run here so setup ends with the same verdict the
    # bench will reach rather than a promise that it should.
    settings.vapi_assistant_id = assistant["id"]
    problems = get_vendor("vapi").verify_agent(spec)
    if problems:
        print("\nVERIFY FAILED -- the bench would refuse to run:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nverify: clean -- the bench will accept this assistant")

    print("\nPin the assistant in config/vendors.yaml (vapi block):")
    print(f"  assistant_id: {assistant['id']}")
    print("Or set it in .env:")
    print(f"  VAPI_ASSISTANT_ID={assistant['id']}")
    print(f"\nThe number ({number.get('number')}) is DISCOVERED from the "
          "assistant's inbound routing -- do not set VAPI_PHONE_NUMBER unless "
          "you mean to override it, and it is verified against this assistant "
          "either way.")
    print(json.dumps({"assistant_id": assistant["id"],
                      "number": number.get("number")}))


if __name__ == "__main__":
    main()
