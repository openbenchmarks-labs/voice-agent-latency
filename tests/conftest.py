"""Shared fixture building.

Fixtures are assembled once per test session into a tmp dir rather than committed
as WAVs: the inputs (speech clips in analyzer/fixtures/sources/) are committed, and
assembly is deterministic, so this keeps the repo small without giving up
reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from analyzer.fixtures import build_all

# The suite must be hermetic: no test may depend on a developer's .env. Set
# before any harness import so Settings never sees a missing base URL.
os.environ.setdefault("PUBLIC_BASE_URL", "https://bench.invalid")

# Settings loads .env directly, so clearing os.environ is not
# enough -- the fields have to be nulled on the instance. Anything here is a
# value that would otherwise make a test's verdict depend on whose laptop it
# ran on. TELNYX_VENDOR_NUMBER earned its place: with a real number in .env,
# the dial-target tests silently exercised that number instead of the mocked
# account, and passed or failed accordingly.
#
# ONE list, applied twice: once at import (before any adapter reads settings)
# and once per test (so a test that sets a value cannot leak into the next).
# Two hand-maintained lists drifted apart within a day of existing.
LEAKY_SETTINGS = (
    # Carrier.
    "plivo_from_number", "plivo_auth_id", "plivo_auth_token",
    # Which agent, and which number reaches it, per vendor. A real value here
    # makes a discovery test pass without discovering anything.
    "telnyx_vendor_number", "telnyx_assistant_id",
    "vapi_phone_number", "vapi_assistant_id",
    "retell_phone_number", "retell_agent_id",
    "bland_phone_number",
    "elevenlabs_phone_number", "elevenlabs_agent_id",
    # API keys. Every adapter mocks its transport, so these should never be
    # read -- but if a mock is ever incomplete, the failure must be the same on
    # every machine. With a key present an unmocked client makes a REAL request
    # against a live account; without one it raises VendorNotReady immediately.
    "telnyx_api_key", "vapi_api_key", "retell_api_key",
    "bland_api_key", "elevenlabs_api_key",
    # Object storage. A real connection string here would let an incompletely
    # mocked upload test WRITE to the public container -- the one credential in
    # this list whose leak is a publish, not just a read.
    "azure_storage_connection_string", "azure_public_base_url",
)

from harness.config import settings as _settings  # noqa: E402

for _leaky in LEAKY_SETTINGS:
    setattr(_settings, _leaky, None)


@pytest.fixture(autouse=True)
def _no_vendor_overrides_from_env(monkeypatch):
    """The per-test half of the hermeticity above.

    The import-time pass covers adapters that read settings at import; this
    covers anything a test mutates, and re-asserts the nulls for tests that run
    after one which set a value deliberately.
    """
    from harness.config import settings

    for field in LEAKY_SETTINGS:
        monkeypatch.setattr(settings, field, None, raising=False)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixtures")
    build_all(out)
    return out


@pytest.fixture(scope="session")
def truths(fixture_dir) -> dict[str, dict]:
    import json

    return {
        p.stem.replace(".truth", ""): json.loads(p.read_text())
        for p in fixture_dir.glob("*.truth.json")
    }


def load_fixture(fixture_dir: Path, truth: dict):
    """Return (near, far, reference, rate) for a fixture."""
    audio, rate = sf.read(fixture_dir / truth["wav"], dtype="int16")
    ref, ref_rate = sf.read(fixture_dir / truth["reference_wav"], dtype="int16")
    assert ref_rate == rate
    near = audio[:, truth["channels"]["near"]]
    far = audio[:, truth["channels"]["far"]]
    return np.ascontiguousarray(near), np.ascontiguousarray(far), ref, rate
