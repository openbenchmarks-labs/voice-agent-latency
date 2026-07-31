"""name -> adapter, resolved from config/vendors.yaml.

The only place concrete vendor classes appear (same rule as carriers/base.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.config import data_root

from .base import AgentSpec, VendorAdapter

_PKG_ROOT = Path(__file__).resolve().parent.parent      # the runner root
CONFIG_PATH = _PKG_ROOT / "config" / "vendors.yaml"


def load_vendor_config(name: str, config_path: Path | None = None) -> dict:
    """The vendor's YAML block, with the system prompt inlined.

    `system_prompt_from_scenarios` (a repo-root-relative path) takes the prompt
    from the benchmark's own scenarios file, so the text the vendor is verified
    against and the text the questions were written for cannot drift apart.
    `system_prompt_file` is the escape hatch for a vendor whose prompt is not a
    scenario prompt.
    """
    path = config_path or CONFIG_PATH
    config = yaml.safe_load(path.read_text())
    if name not in config:
        raise KeyError(f"vendor {name!r} not in {path} (have: {sorted(config)})")
    block = dict(config[name])

    agent = dict(block.get("agent", {}))
    prompt_file = agent.pop("system_prompt_file", None)
    scenarios_file = agent.pop("system_prompt_from_scenarios", None)
    if scenarios_file:
        scenarios_path = (data_root() / scenarios_file).resolve()
        scenarios = json.loads(scenarios_path.read_text())
        agent["system_prompt"] = scenarios.get("system_prompt", "").strip()
    elif prompt_file:
        agent["system_prompt"] = (path.parent.parent / prompt_file).read_text().strip()
    block["agent"] = agent
    return block


def spec_from_config(block: dict) -> AgentSpec:
    agent = block.get("agent", {})
    return AgentSpec(
        system_prompt=agent.get("system_prompt", ""),
        greeting=agent.get("greeting", ""),
        model=agent.get("model"),
        stt=agent.get("stt"),
        tts=agent.get("tts"),
    )


def get_vendor(name: str, config_path: Path | None = None) -> VendorAdapter:
    block = load_vendor_config(name, config_path)
    adapter_kind = block.get("adapter", name)

    if adapter_kind == "telnyx":
        from .telnyx import TelnyxVendor

        return TelnyxVendor(block)
    if adapter_kind == "vapi":
        from .vapi import VapiVendor

        return VapiVendor(block)
    if adapter_kind == "retell":
        from .retell import RetellVendor

        return RetellVendor(block)
    if adapter_kind == "bland":
        from .bland import BlandVendor

        return BlandVendor(block)
    if adapter_kind == "elevenlabs":
        from .elevenlabs import ElevenLabsVendor

        return ElevenLabsVendor(block)
    raise ValueError(f"unknown vendor adapter {adapter_kind!r}")
