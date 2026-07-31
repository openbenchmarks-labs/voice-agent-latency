"""Carrier (CPaaS) abstraction: who originates our calls.

One implementation: Plivo. The method requires that the instrument not be a
competitor of the measured, and Telnyx -- an early development carrier -- is a
vendor under test, so its carrier adapter was removed.
The Protocol stays because it is the seam the harness talks through, and
because a second neutral carrier is a plausible future (carrier-effect
cross-checks).

Nothing outside carriers/ may import a concrete carrier module.
"""

from .base import Carrier, get_carrier

__all__ = ["Carrier", "get_carrier"]
