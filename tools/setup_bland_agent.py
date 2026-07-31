#!/usr/bin/env python3
"""Provision the Bland agent under test, from the committed config.

A TOOL, not part of the bench: vendor adapters are read-only by contract
(tests/test_vendors.py enforces it structurally), because a bench that can
rewrite the thing it measures cannot publish a trustworthy receipt.

Bland has no agent object behind an inbound call -- the prompt and greeting live
ON the number -- so this script configures a number rather than creating an
agent. Consequences:

  * With no inbound number there is nothing to configure. Buying one is a
    $15/MONTH SUBSCRIPTION, so it is opt-in behind --buy-number and the tool
    prints the account balance and asks for --yes-i-accept-the-cost first.
  * Anything already on the number (webhook, tools, transfer numbers) is left
    alone: only prompt, first_sentence and the fields the bench requires are set.

Usage:
    .venv/bin/python tools/setup_bland_agent.py --dry-run
    .venv/bin/python tools/setup_bland_agent.py                    # configure
    .venv/bin/python tools/setup_bland_agent.py --buy-number \\
        --area-code 415 --yes-i-accept-the-cost
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

API = "https://api.bland.ai"
NUMBER_MONTHLY_USD = 15


def client() -> httpx.Client:
    if not settings.bland_api_key:
        sys.exit("BLAND_API_KEY is not set in .env")
    return httpx.Client(
        base_url=API,
        # Bland takes the key raw, not as a Bearer token.
        headers={"authorization": settings.bland_api_key},
        timeout=30.0,
        # Bland is migrating paths (/v1/inbound/purchase now 308s to
        # /numbers/purchase). A 308 preserves method and body by spec, so
        # following it is safe for POSTs and survives the migration.
        follow_redirects=True,
    )


def check(response: httpx.Response, what: str) -> dict:
    if response.status_code >= 300:
        sys.exit(f"{what} failed: {response.status_code} {response.text[:400]}")
    payload = response.json()
    if isinstance(payload, dict) and payload.get("status") == "error":
        sys.exit(f"{what} failed: {payload.get('message')}")
    return payload


def digits(number: str | None) -> str:
    return "".join(ch for ch in str(number or "") if ch.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", default=None,
                        help="Inbound number to configure (default: the only one)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing")
    parser.add_argument("--buy-number", action="store_true",
                        help=f"Purchase an inbound number (${NUMBER_MONTHLY_USD}/MONTH)")
    parser.add_argument("--area-code", default=None,
                        help="3-digit area code for --buy-number")
    parser.add_argument("--yes-i-accept-the-cost", action="store_true",
                        help=f"Required with --buy-number: a recurring "
                             f"${NUMBER_MONTHLY_USD}/month charge")
    args = parser.parse_args()

    spec = spec_from_config(load_vendor_config("bland"))
    if not spec.system_prompt or not spec.greeting:
        sys.exit("vendors.yaml gave an empty prompt or greeting")
    print("agent under test: a Bland INBOUND NUMBER (the number is the agent)")
    print(f"  greeting: {spec.greeting!r}")
    print(f"  prompt:   {len(spec.system_prompt)} chars from the scenarios file")
    if args.dry_run:
        print("\n--dry-run: no writes")

    with client() as api:
        me = check(api.get("/v1/me"), "read account")
        balance = (me.get("billing") or {}).get("current_balance")
        print(f"  account balance: ${balance}")

        numbers = check(api.get("/v1/inbound"), "list numbers").get(
            "inbound_numbers", [])
        print(f"  inbound numbers: {[n.get('phone_number') for n in numbers] or 'none'}")

        target: dict | None = None
        if args.number:
            match = [n for n in numbers if digits(n.get("phone_number")) == digits(args.number)]
            if not match:
                sys.exit(f"{args.number} is not an inbound number on this account")
            target = match[0]
        elif len(numbers) == 1:
            target = numbers[0]
        elif len(numbers) > 1:
            sys.exit("multiple inbound numbers; pick one with --number")

        if target is None:
            if not args.buy_number:
                print(
                    f"\nNo inbound number, and on Bland the number IS the agent --"
                    f" there is nothing to configure without one."
                    f"\n  A number is a ${NUMBER_MONTHLY_USD}/month subscription."
                    f" To buy one:"
                    f"\n    tools/setup_bland_agent.py --buy-number --area-code 415"
                    f" --yes-i-accept-the-cost"
                    f"\n  Or add a number you own through Bland's BYO-Twilio flow"
                    f" (/v1/inbound/insert), then re-run."
                )
                return
            if not args.yes_i_accept_the_cost:
                sys.exit(
                    f"--buy-number is a recurring ${NUMBER_MONTHLY_USD}/month "
                    f"charge on the stored payment method. Re-run with "
                    f"--yes-i-accept-the-cost to confirm."
                )
            if args.dry_run:
                print(f"\nwould purchase a number (${NUMBER_MONTHLY_USD}/month)")
                return
            body = {"area_code": args.area_code or "415"}
            bought = check(api.post("/v1/inbound/purchase", json=body),
                           "purchase number")
            number = bought.get("phone_number") or bought.get("number")
            print(f"\npurchased {number} (${NUMBER_MONTHLY_USD}/month)")
            target = {"phone_number": number}

        phone = target.get("phone_number")
        wanted = {
            "prompt": spec.system_prompt,
            # Without this Bland lets the model open the call, so the greeting
            # would differ every call and live VAD would have no fixed utterance
            # to wait for. It is a bench requirement, not a tuning choice.
            "first_sentence": spec.greeting,
        }
        drift = {
            key: value for key, value in wanted.items()
            if (target.get(key) or "").strip() != value.strip()
        }

        if not drift:
            print(f"\n{phone} already matches the committed config")
        elif args.dry_run:
            print(f"\nwould set on {phone}: {', '.join(sorted(drift))}")
            return
        else:
            # The update endpoint requires `prompt`, so send both fields rather
            # than only the drifted one.
            check(api.post(f"/v1/inbound/{digits(phone)}", json=wanted),
                  "update inbound number")
            print(f"\nset on {phone}: {', '.join(sorted(drift))}")

    settings.bland_phone_number = phone
    problems = get_vendor("bland").verify_agent(spec)
    if problems:
        print("\nVERIFY FAILED -- the bench would refuse to run:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nverify: clean -- the bench will accept this number")

    # Unlike every other vendor here, the NUMBER is the agent's identity on
    # Bland -- there is no agent object behind an inbound call -- so this is
    # the pin, not an override of discovery.
    print("\nPut this in .env:")
    print(f"  BLAND_PHONE_NUMBER={phone}")
    print(json.dumps({"number": phone}))


if __name__ == "__main__":
    main()
