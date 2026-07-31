#!/usr/bin/env python3
"""Provision the Telnyx agent under test, from the committed config.

This is a TOOL, not part of the bench. The vendor adapters are read-only by
contract (tests/test_vendors.py enforces it structurally) because a bench that
can rewrite the thing it measures cannot publish a trustworthy receipt. Setup is
a separate, explicit, human-invoked action -- this script -- and it is the only
place in the voice tree that writes to a vendor account.

What it does, idempotently:
  1. finds or creates an assistant named --name
  2. patches ONLY greeting + instructions + the telephony feature, leaving the
     stack (model/voice/transcription) at whatever Telnyx gave it -- the closed
     division measures defaults as shipped, so choosing a model here would
     change what is being measured
  3. resolves the TeXML application that routes inbound calls to the assistant
  4. checks a voice-enabled number is attached to that application, and with
     --buy-number purchases and attaches one
  5. re-reads everything and runs the bench's own verify gate

Telnyx's inbound path has one more link than the other platforms, and it is
where a run was lost before: assistant -> telephony_settings.default_texml_app_id
-> a phone number whose connection_id is that app. Owning the number is not
enough. On 2026-07-29 the far-end number was silently taken over by another
connection, so calls reached the wrong agent and the symptom (slope -0.13)
looked like a broken instrument rather than broken wiring. Step 4 checks the
link rather than assuming it.

Afterwards `python -m harness.bench --vendor telnyx` can verify and dial it.

Usage:
    .venv/bin/python tools/setup_telnyx_agent.py [--name northwind-telnyx-bench]
        [--adopt ASSISTANT_ID] [--buy-number] [--area-code 415] [--dry-run]
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

API = "https://api.telnyx.com"
DEFAULT_NAME = "northwind-telnyx-bench"

# Telnyx's create-assistant API requires a model; there is no "give me your
# default" option, so one has to be named here. This is what the account's
# existing bench assistant runs (assistant-0e01469e-…, read 2026-07-30), i.e.
# the platform's own choice rather than ours -- the same reasoning as the Vapi
# script's BLANK_TEMPLATE_DEFAULTS. The receipt records whatever is actually in
# force, and `--stack-from <assistant_id>` re-derives this if Telnyx moves.
TEMPLATE_DEFAULTS = {"model": "moonshotai/Kimi-K2.6"}


def client() -> httpx.Client:
    if not settings.telnyx_api_key:
        sys.exit("TELNYX_API_KEY is not set in .env")
    return httpx.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {settings.telnyx_api_key}"},
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
    payload = response.json()
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing")
    parser.add_argument("--adopt", default=None, metavar="ASSISTANT_ID",
                        help="Use an existing assistant instead of matching on "
                             "--name (e.g. one built in the portal)")
    parser.add_argument("--buy-number", action="store_true",
                        help="Purchase a US local voice number (COSTS MONEY) "
                             "and attach it to the assistant's TeXML app")
    parser.add_argument("--area-code", default=None,
                        help="3-digit US area code for --buy-number")
    parser.add_argument("--stack-from", default=None, metavar="ASSISTANT_ID",
                        help="Re-derive the default model from a "
                             "portal-created assistant")
    args = parser.parse_args()

    spec = spec_from_config(load_vendor_config("telnyx"))
    if not spec.system_prompt or not spec.greeting:
        sys.exit("vendors.yaml gave an empty prompt or greeting")
    print(f"agent under test: {args.name}")
    print(f"  greeting: {spec.greeting!r}")
    print(f"  prompt:   {len(spec.system_prompt)} chars from the scenarios file")
    if args.dry_run:
        print("\n--dry-run: no writes")

    with client() as api:
        defaults = dict(TEMPLATE_DEFAULTS)
        if args.stack_from:
            source = check(api.get(f"/v2/ai/assistants/{args.stack_from}"),
                           "read stack source")
            if source.get("model"):
                defaults["model"] = source["model"]
            print(f"  stack derived from {args.stack_from}")
        print(f"  template default: model={defaults['model']}")

        # ---------------------------------------------------------- assistant
        if args.adopt:
            assistant = check(api.get(f"/v2/ai/assistants/{args.adopt}"),
                              "read adopted assistant")
            print(f"\nadopted assistant {assistant['id']} "
                  f"({assistant.get('name')!r})")
        else:
            existing = [a for a in items(check(api.get("/v2/ai/assistants"),
                                               "list assistants"))
                        if a.get("name") == args.name]
            if len(existing) > 1:
                sys.exit(f"{len(existing)} assistants named {args.name!r}; "
                         "delete the duplicates in the portal, or pass --adopt "
                         "with the id you mean")
            if existing:
                assistant = existing[0]
                print(f"\nfound assistant {assistant['id']}")
            elif args.dry_run:
                print("\nwould create the assistant "
                      f"(model {defaults['model']}, telephony enabled)")
                return
            else:
                assistant = check(api.post("/v2/ai/assistants", json={
                    "name": args.name,
                    "model": defaults["model"],
                    "instructions": spec.system_prompt,
                    "greeting": spec.greeting,
                    # Without this the assistant cannot answer a phone call at
                    # all, and verify_agent refuses the run.
                    "enabled_features": ["telephony"],
                }), "create assistant")
                print(f"\ncreated assistant {assistant['id']}")

        # Patch only what the bench requires; never the stack.
        features = list(assistant.get("enabled_features") or [])
        patch: dict = {}
        if (assistant.get("greeting") or "").strip() != spec.greeting.strip():
            patch["greeting"] = spec.greeting
        if (assistant.get("instructions") or "").strip() != spec.system_prompt.strip():
            patch["instructions"] = spec.system_prompt
        if "telephony" not in features:
            patch["enabled_features"] = features + ["telephony"]

        if patch and not args.dry_run:
            assistant = check(
                api.patch(f"/v2/ai/assistants/{assistant['id']}", json=patch),
                "patch assistant",
            )
            print(f"  patched: {', '.join(sorted(patch))}")
        elif patch:
            print(f"  would patch: {', '.join(sorted(patch))}")
        else:
            print("  already matches the committed config")
        print(f"  live stack: model={assistant.get('model')} "
              f"voice={(assistant.get('voice_settings') or {}).get('voice')} "
              f"stt={(assistant.get('transcription') or {}).get('model')}")

        # --------------------------------------------------------- texml app
        # Telnyx creates this app when telephony is enabled; it is what routes
        # an inbound call to the assistant. No app means no inbound path, and
        # nothing downstream can compensate.
        app_id = (assistant.get("telephony_settings") or {}).get(
            "default_texml_app_id")
        if not app_id:
            print("\nNO TEXML APPLICATION on this assistant, so nothing routes "
                  "an inbound call to it.")
            print("  Telnyx provisions one when telephony is enabled -- give it "
                  "a moment and re-run, or open the assistant in the portal "
                  "(AI > Assistants > Calling) once to force it.")
            print(f"\nTELNYX_ASSISTANT_ID={assistant['id']}")
            return
        print(f"\ntexml app: {app_id}")

        # ------------------------------------------------------------- number
        attached = [n for n in items(check(
            api.get("/v2/phone_numbers", params={"filter[connection_id]": app_id}),
            "list numbers on the app"))
            if n.get("status") == "active"]

        if attached:
            number = attached[0]["phone_number"]
            print(f"number already routed: {number}")
            if len(attached) > 1:
                print(f"  ({len(attached)} attached; the adapter dials the first)")
        elif args.dry_run:
            print("would attach a number (--buy-number purchases one)")
            return
        elif not args.buy_number:
            print("NO NUMBER attached to this assistant's TeXML app, so it "
                  "cannot be dialled.")
            print("  Attach a voice-enabled number you already own in the "
                  "portal (Numbers > pick one > Connection = this TeXML app),")
            print("  or re-run with --buy-number [--area-code 415] to purchase "
                  "one. Telnyx numbers are NOT free.")
            print(f"\nTELNYX_ASSISTANT_ID={assistant['id']}")
            return
        else:
            search = {"filter[country_code]": "US",
                      "filter[phone_number_type]": "local",
                      "filter[features]": "voice",
                      "filter[limit]": 3}
            if args.area_code:
                search["filter[national_destination_code]"] = args.area_code
            candidates = [d["phone_number"] for d in items(check(
                api.get("/v2/available_phone_numbers", params=search),
                "number search"))]
            if not candidates:
                sys.exit("no purchasable numbers matched -- try another area code")
            number = candidates[0]
            print(f"buying {number} ...")
            order = check(api.post("/v2/number_orders",
                                   json={"phone_numbers": [{"phone_number": number}]}),
                          "number order")
            print(f"  order {order.get('id')} status={order.get('status')}")

            # It lands in inventory asynchronously; it cannot be attached before
            # it does.
            phone_id = None
            deadline = time.time() + 60
            while time.time() < deadline:
                owned = items(check(api.get("/v2/phone_numbers",
                                            params={"filter[phone_number]": number}),
                                    "poll inventory"))
                if owned:
                    phone_id = owned[0]["id"]
                    break
                time.sleep(2.0)
            if not phone_id:
                sys.exit(f"{number} never appeared in inventory; check the portal")

            assigned = check(api.patch(f"/v2/phone_numbers/{phone_id}",
                                       json={"connection_id": app_id}),
                             "attach number")
            print(f"  attached to TeXML app {assigned.get('connection_id')}")

    # ------------------------------------------------------------------ verify
    # The bench's own gate, run here so setup ends with the same verdict the
    # bench will reach rather than a promise that it should.
    settings.telnyx_assistant_id = assistant["id"]
    problems = get_vendor("telnyx").verify_agent(spec)
    if problems:
        print("\nVERIFY FAILED -- the bench would refuse to run:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nverify: clean -- the bench will accept this assistant")

    print("\nPin the assistant in config/vendors.yaml (telnyx block):")
    print(f"  assistant_id: {assistant['id']}")
    print("Or set it in .env:")
    print(f"  TELNYX_ASSISTANT_ID={assistant['id']}")
    print(f"\nThe number ({number}) is DISCOVERED from the assistant's TeXML "
          "app -- do not set TELNYX_VENDOR_NUMBER unless you mean to override "
          "it, and it is verified against this assistant either way.")
    print(json.dumps({"assistant_id": assistant["id"],
                      "texml_app_id": app_id,
                      "number": number}))


if __name__ == "__main__":
    main()
