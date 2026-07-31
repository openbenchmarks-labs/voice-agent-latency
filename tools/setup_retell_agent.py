#!/usr/bin/env python3
"""Provision the Retell agent under test, from the committed config.

A TOOL, not part of the bench: vendor adapters are read-only by contract
(tests/test_vendors.py enforces it structurally), because a bench that can
rewrite the thing it measures cannot publish a trustworthy receipt.

Retell needs two objects, created in order:
  1. a retell-llm holding general_prompt, begin_message, start_speaker="agent"
  2. an agent whose response_engine points at that llm, holding voice + language

The stack (model, voice, language) is copied from what the dashboard's
single-prompt template produced rather than chosen here -- the
closed division measures defaults as shipped. `--stack-from <agent_id>`
re-derives them.

Phone numbers are NOT free on this platform, so buying one is opt-in:
    --buy-number --area-code 415
Without --buy-number the tool sets everything else up and tells you what is
left. Use --import-number to bind a number you already own instead.

Usage:
    .venv/bin/python tools/setup_retell_agent.py [--dry-run]
    .venv/bin/python tools/setup_retell_agent.py --buy-number --area-code 415
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._bootstrap import require_voice_venv           # noqa: E402

require_voice_venv()

from harness.config import settings                       # noqa: E402
from vendors import get_vendor                            # noqa: E402
from vendors.registry import load_vendor_config, spec_from_config  # noqa: E402

API = "https://api.retellai.com"
DEFAULT_NAME = "northwind-retell-bench"
REQUIRED_START_SPEAKER = "agent"

# Copied verbatim from what the dashboard's blank single-prompt template and its
# llm produced, read 2026-07-30. (Create one in the dashboard and read it back to
# re-derive them for yourself.) Retell's create endpoints
# require voice_id and a model, so unlike a dashboard click there is no "leave it
# to the platform" option; these are the platform's own template values, made
# explicit rather than silently chosen by us.
TEMPLATE_DEFAULTS = {
    "model": "gpt-4.1",
    "voice_id": "retell-Cimo",
    "language": "en-US",
}

# Retell provisions through a third-party carrier. Default to twilio: telnyx is
# itself a vendor under test, and putting one benched vendor's PSTN leg on
# another's network is exactly the entanglement the method avoids on the caller
# side.
NUMBER_PROVIDER = "twilio"


def client() -> httpx.Client:
    if not settings.retell_api_key:
        sys.exit("RETELL_API_KEY is not set in .env")
    return httpx.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {settings.retell_api_key}"},
        timeout=30.0,
    )


def items(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("data") or payload.get("results") or []
    return []


def inbound_binding(agent_id: str) -> list[dict]:
    """The weighted inbound binding. The singular `inbound_agent_id` field was
    deprecated 2026-03-31 and is now rejected outright
    (docs.retellai.com/deprecation-notice/2026/03-31_phone_number_agent_fields);
    weights must sum to 1."""
    return [{"agent_id": agent_id, "weight": 1}]


def check(response: httpx.Response, what: str) -> dict:
    if response.status_code >= 300:
        sys.exit(f"{what} failed: {response.status_code} {response.text[:400]}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing")
    parser.add_argument("--buy-number", action="store_true",
                        help="Purchase a number (COSTS MONEY on this platform)")
    parser.add_argument("--area-code", default=None,
                        help="3-digit US area code for --buy-number")
    parser.add_argument("--import-number", default=None, metavar="E164",
                        help="Bind a number you already own instead of buying")
    parser.add_argument("--stack-from", default=None, metavar="AGENT_ID",
                        help="Re-derive model/voice/language from a "
                             "dashboard-created agent")
    args = parser.parse_args()

    spec = spec_from_config(load_vendor_config("retell"))
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
            source = check(api.get(f"/get-agent/{args.stack_from}"), "read stack source")
            source_llm_id = (source.get("response_engine") or {}).get("llm_id")
            source_llm = check(api.get(f"/get-retell-llm/{source_llm_id}"),
                               "read stack source llm") if source_llm_id else {}
            defaults = {
                "model": source_llm.get("model") or TEMPLATE_DEFAULTS["model"],
                "voice_id": source.get("voice_id") or TEMPLATE_DEFAULTS["voice_id"],
                "language": source.get("language") or TEMPLATE_DEFAULTS["language"],
            }
            print(f"  stack derived from {args.stack_from}")
        print(f"  template defaults: model={defaults['model']} "
              f"voice={defaults['voice_id']} language={defaults['language']}")

        existing = [a for a in items(check(api.get("/list-agents"), "list agents"))
                    if a.get("agent_name") == args.name]
        if len(existing) > 1:
            sys.exit(f"{len(existing)} agents named {args.name!r}; "
                     "delete the duplicates in the dashboard first")

        llm_body = {
            "general_prompt": spec.system_prompt,
            "begin_message": spec.greeting,
            "start_speaker": REQUIRED_START_SPEAKER,
            "model": defaults["model"],
        }

        if existing:
            agent = existing[0]
            llm_id = (agent.get("response_engine") or {}).get("llm_id")
            print(f"\nfound agent {agent['agent_id']} (llm {llm_id})")
            if args.dry_run:
                print("would patch the llm to match the committed prompt/greeting")
                return
            llm = check(api.patch(f"/update-retell-llm/{llm_id}", json=llm_body),
                        "update retell-llm")
            print("  llm updated: prompt, begin_message, start_speaker")
        else:
            if args.dry_run:
                print("\nwould create: retell-llm, then agent pointing at it")
                return
            llm = check(api.post("/create-retell-llm", json=llm_body),
                        "create retell-llm")
            print(f"\ncreated retell-llm {llm['llm_id']}")
            agent = check(api.post("/create-agent", json={
                "agent_name": args.name,
                "voice_id": defaults["voice_id"],
                "language": defaults["language"],
                "response_engine": {"type": "retell-llm", "llm_id": llm["llm_id"]},
            }), "create agent")
            print(f"created agent {agent['agent_id']}")

        # ------------------------------------------------------------- number
        numbers = items(check(api.get("/list-phone-numbers"), "list numbers"))

        def inbound_ids(number: dict) -> list[str]:
            ids = [number["inbound_agent_id"]] if number.get("inbound_agent_id") else []
            for entry in number.get("inbound_agents") or []:
                ids.append(entry.get("agent_id") if isinstance(entry, dict) else entry)
            return [i for i in ids if i]

        bound = [n for n in numbers if agent["agent_id"] in inbound_ids(n)]
        number: dict | None = bound[0] if bound else None

        if number:
            print(f"\nnumber already bound inbound: {number.get('phone_number')}")
        elif args.import_number:
            number = check(api.post("/import-phone-number", json={
                "phone_number": args.import_number,
                "inbound_agents": inbound_binding(agent["agent_id"]),
            }), "import number")
            print(f"\nimported + bound {number.get('phone_number')}")
        elif args.buy_number:
            if not args.area_code:
                sys.exit("--buy-number needs --area-code (3-digit US)")
            body = {
                "area_code": int(args.area_code),
                "inbound_agents": inbound_binding(agent["agent_id"]),
                "number_provider": NUMBER_PROVIDER,
            }
            number = check(api.post("/create-phone-number", json=body),
                           "create number")
            print(f"\npurchased {number.get('phone_number')} "
                  f"via {NUMBER_PROVIDER}")
        else:
            free = [n for n in numbers if not inbound_ids(n)]
            print("\nno number bound to this agent.")
            if free:
                print(f"  unbound numbers on the account: "
                      f"{[n.get('phone_number') for n in free]} -- bind one in the "
                      f"dashboard, or re-run with --import-number <E164>")
            else:
                print("  Retell numbers are NOT free. Re-run with "
                      "--buy-number --area-code 415 to purchase one, or "
                      "--import-number <E164> to bind one you own.")
            print(f"\nRETELL_AGENT_ID={agent['agent_id']}")
            return

    settings.retell_agent_id = agent["agent_id"]
    problems = get_vendor("retell").verify_agent(spec)
    if problems:
        print("\nVERIFY FAILED -- the bench would refuse to run:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nverify: clean -- the bench will accept this agent")

    print("\nPin the agent in config/vendors.yaml (retell block):")
    print(f"  agent_id: {agent['agent_id']}")
    print("Or set it in .env:")
    print(f"  RETELL_AGENT_ID={agent['agent_id']}")
    print(f"\nThe number ({number.get('phone_number')}) is DISCOVERED from the "
          "agent's inbound binding -- do not set RETELL_PHONE_NUMBER unless you "
          "mean to override it, and it is verified against this agent either way.")
    print(json.dumps({"agent_id": agent["agent_id"],
                      "llm_id": llm.get("llm_id"),
                      "number": number.get("phone_number")}))


if __name__ == "__main__":
    main()
