"""Vendor abstraction: who ANSWERS our calls and gets measured.

Mirror of carriers/ (who places calls). The harness never knows which vendor is
on the other end -- it dials a number and measures what happens. Swapping
vendors is one adapter file plus one YAML block; nothing outside vendors/ may
import a concrete vendor module (enforced by test).

Two rules baked into the interface:

1. READ-ONLY. Adapters verify and snapshot; they never create or modify
   anything in a vendor account. Anything requiring a write is printed as
   instructions for the operator to run. (`verify_agent`, not `ensure_agent`.)

2. THE RECEIPT. Every measurement ships with an AppliedConfig -- the vendor's
   live configuration, hashed. `defaults_used` records what the vendor chose
   for knobs we left alone (closed division: defaults ARE the product);
   `unsupported` records knobs the vendor does not expose at all, so
   incomparability is data, not a footnote.
"""

from .base import AgentSpec, AppliedConfig, DialTarget, VendorAdapter
from .registry import get_vendor, load_vendor_config

__all__ = [
    "AgentSpec",
    "AppliedConfig",
    "DialTarget",
    "VendorAdapter",
    "get_vendor",
    "load_vendor_config",
]
