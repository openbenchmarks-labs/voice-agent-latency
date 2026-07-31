#!/usr/bin/env python3
"""Provision the ElevenLabs agent under test, from the committed config.

A TOOL, not part of the bench: vendor adapters are read-only by contract
(tests/test_vendors.py enforces it structurally), because a bench that can
rewrite the thing it measures cannot publish a trustworthy receipt.

What it does, idempotently:
  1. finds or creates a Conversational AI agent named --name
  2. sets the Northwind prompt and greeting inside conversation_config
  3. DISARMS turn.soft_timeout_config. Left on, the agent speaks a stall phrase
     when the LLM is slow, and that filler becomes the first audio -- TTFAB would
     time the "hmm" instead of the answer, and it is long enough to slip past the
     analyzer's short-noise guard. This is a correctness requirement, not tuning.
  4. leaves the stack (llm, asr, tts) exactly as the platform set it

It does NOT get you a number: ElevenLabs sells none. Inbound needs a number
imported from Twilio, Exotel or your own SIP trunk, which needs that provider's
credentials -- so it stays a deliberate human step and the script just says so.

Usage:
    .venv/bin/python tools/setup_elevenlabs_agent.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
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

API = "https://api.elevenlabs.io"
DEFAULT_NAME = "northwind-elevenlabs-bench"
FILLER_DISABLED_TIMEOUT = -1.0


def client() -> httpx.Client:
    if not settings.elevenlabs_api_key:
        sys.exit("ELEVENLABS_API_KEY is not set in .env")
    return httpx.Client(
        base_url=API,
        headers={"xi-api-key": settings.elevenlabs_api_key},
        timeout=30.0,
    )


def check(response: httpx.Response, what: str) -> dict:
    if response.status_code >= 300:
        sys.exit(f"{what} failed: {response.status_code} {response.text[:400]}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing")
    parser.add_argument("--adopt", default=None, metavar="AGENT_ID",
                        help="Configure this existing agent instead of creating one")
    args = parser.parse_args()

    spec = spec_from_config(load_vendor_config("elevenlabs"))
    if not spec.system_prompt or not spec.greeting:
        sys.exit("vendors.yaml gave an empty prompt or greeting")
    print(f"agent under test: {args.name}")
    print(f"  greeting: {spec.greeting!r}")
    print(f"  prompt:   {len(spec.system_prompt)} chars from the scenarios file")
    if args.dry_run:
        print("\n--dry-run: no writes")

    with client() as api:
        listing = check(api.get("/v1/convai/agents"), "list agents")
        agents = listing.get("agents") or []

        if args.adopt:
            match = [a for a in agents if a.get("agent_id") == args.adopt]
            if not match:
                sys.exit(f"agent {args.adopt} is not on this account")
            agent_id = args.adopt
        else:
            named = [a for a in agents if a.get("name") == args.name]
            if len(named) > 1:
                sys.exit(f"{len(named)} agents named {args.name!r}; "
                         "delete the duplicates first")
            agent_id = named[0]["agent_id"] if named else None

        if agent_id is None:
            if args.dry_run:
                print("\nwould create the agent (platform picks the default stack)")
                return
            created = check(api.post("/v1/convai/agents/create", json={
                "name": args.name,
                "conversation_config": {
                    "agent": {
                        "first_message": spec.greeting,
                        "prompt": {"prompt": spec.system_prompt},
                    },
                },
            }), "create agent")
            agent_id = created.get("agent_id")
            print(f"\ncreated agent {agent_id}")

        agent = check(api.get(f"/v1/convai/agents/{agent_id}"), "read agent")
        print(f"\nagent {agent_id} ({agent.get('name')})")

        config = copy.deepcopy(agent.get("conversation_config") or {})
        agent_block = config.setdefault("agent", {})
        prompt_block = agent_block.setdefault("prompt", {})
        turn = config.setdefault("turn", {})
        soft = turn.setdefault("soft_timeout_config", {})
        asr = config.get("asr") or {}
        tts = config.get("tts") or {}

        print(f"  platform stack: llm={prompt_block.get('llm')} "
              f"asr={asr.get('provider')} tts={tts.get('model_id')}")

        changes = []
        if (prompt_block.get("prompt") or "").strip() != spec.system_prompt.strip():
            prompt_block["prompt"] = spec.system_prompt
            changes.append("prompt")
        if (agent_block.get("first_message") or "").strip() != spec.greeting.strip():
            agent_block["first_message"] = spec.greeting
            changes.append("first_message")

        timeout = soft.get("timeout_seconds")
        if timeout is None or float(timeout) >= 0:
            print(f"  filler is ARMED (timeout={timeout}, "
                  f"message={soft.get('message')!r}) -- disabling: it would become "
                  f"the first audio and make TTFAB time the stall phrase")
            soft["timeout_seconds"] = FILLER_DISABLED_TIMEOUT
            changes.append("soft_timeout_config (disarmed)")

        if not changes:
            print("  already matches the committed config")
        elif args.dry_run:
            print(f"  would set: {', '.join(changes)}")
            return
        else:
            check(api.patch(f"/v1/convai/agents/{agent_id}",
                            json={"conversation_config": config}),
                  "update agent")
            print(f"  set: {', '.join(changes)}")

        numbers = check(api.get("/v1/convai/phone-numbers"), "list numbers")
        assigned = [
            n for n in (numbers if isinstance(numbers, list) else [])
            if (n.get("assigned_agent") or {}).get("agent_id") == agent_id
        ]

    settings.elevenlabs_agent_id = agent_id
    problems = get_vendor("elevenlabs").verify_agent(spec)
    if problems:
        print("\nVERIFY FAILED -- the bench would refuse to run:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nverify: clean -- the bench will accept this agent")

    print("\nPin the agent in config/vendors.yaml (elevenlabs block):")
    print(f"  agent_id: {agent_id}")
    print("Or set it in .env:")
    print(f"  ELEVENLABS_AGENT_ID={agent_id}")
    if assigned:
        print(f"\nThe number ({assigned[0].get('phone_number')}) is DISCOVERED "
              "from the agent's assignment -- do not set "
              "ELEVENLABS_PHONE_NUMBER unless you mean to override it, and it "
              "is verified against this agent either way.")
    else:
        print("\nNOT DIALABLE YET. ElevenLabs sells no phone numbers, so this is the "
              "one vendor money alone cannot unblock:")
        print("  Twilio:    POST /v1/convai/phone-numbers "
              "{phone_number, label, sid, token}")
        print("  SIP trunk: POST /v1/convai/phone-numbers "
              "{phone_number, label, + trunk config}")
        print("  Then assign it to this agent; the bench discovers it from "
              "there.")
        print("  Prefer a trunk independent of the bench's own carrier: routing "
              "both legs over one network is not the PSTN path the other vendors "
              "are measured on.")
    print(json.dumps({"agent_id": agent_id,
                      "number": assigned[0].get("phone_number") if assigned else None}))


if __name__ == "__main__":
    main()
