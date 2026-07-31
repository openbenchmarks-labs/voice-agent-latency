"""Settings, mirroring .env.example.

Credentials are optional
at import time so the app can boot (and tests can run) without a filled .env --
anything that actually needs Plivo fails loudly at the point of use instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# The runner's own root directory. Everything the runner reads or
# writes by default -- .env, runs/, logs/ -- is anchored here, not to the
# process CWD, so a tool works the same whichever directory you run it from.
_PKG_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """The root that `data/…` paths are relative to.

    The runner is published standalone as openbenchmarks-labs/voice-agent-latency,
    They ship inside the tree at data/voice-bench/, so the runner is
    self-contained. The fallback one level up exists because the same code runs
    embedded in a larger repo that keeps its scenarios at its own root; checking
    for the local copy first means one tree works in both layouts with no build
    step and no environment variable, and the committed
    `system_prompt_from_scenarios:` paths in config/vendors.yaml stay
    byte-identical either way -- so the config receipt's sha256 does not move.
    """
    if (_PKG_ROOT / "data" / "voice-bench").is_dir():
        return _PKG_ROOT
    return _PKG_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PKG_ROOT / ".env"),
        extra="ignore",
    )

    # Which CPaaS originates calls. Plivo is the only carrier: the caller side
    # is Plivo XML (harness/dialog.py) and the method forbids measuring a vendor
    # over its own network (Telnyx is a vendor under test).
    carrier: str = "plivo"

    # Plivo -- the carrier.
    plivo_auth_id: str | None = None
    plivo_auth_token: str | None = None
    plivo_from_number: str | None = None
    plivo_account_is_trial: bool = False

    # Telnyx as a VENDOR UNDER TEST. Not carrier config: the API
    # key exists so vendors/telnyx.py can verify the assistant's configuration
    # and produce its receipt. The number is optional -- the adapter discovers
    # it from the assistant's own TeXML app.
    telnyx_api_key: str | None = None
    telnyx_assistant_id: str | None = None
    telnyx_vendor_number: str | None = None

    # Public HTTPS base URL, no trailing slash. Plivo reaches every webhook
    # through it, and the V3 signature is validated against it.
    public_base_url: str | None = None

    # Vapi as a VENDOR UNDER TEST. Same rationale as Telnyx above: read-only
    # config verification and the receipt. Number optional -- the adapter
    # discovers it from the numbers whose assistantId is this assistant.
    vapi_api_key: str | None = None
    vapi_assistant_id: str | None = None
    vapi_phone_number: str | None = None

    # Retell AI as a VENDOR UNDER TEST. Two objects rather than one: the agent
    # holds voice and turn-taking, its response_engine points at a separate
    # retell-llm holding the prompt and greeting, so the adapter reads both.
    retell_api_key: str | None = None
    retell_agent_id: str | None = None
    retell_phone_number: str | None = None

    # Bland AI as a VENDOR UNDER TEST. No agent id: on this platform the inbound
    # NUMBER carries the prompt and greeting, so the number is what pins it.
    bland_api_key: str | None = None
    bland_phone_number: str | None = None

    # ElevenLabs as a VENDOR UNDER TEST. It sells no numbers,
    # so the number is always one imported from Twilio/Exotel/a SIP trunk.
    elevenlabs_api_key: str | None = None
    elevenlabs_agent_id: str | None = None
    elevenlabs_phone_number: str | None = None

    # Object storage for PUBLISHING recordings, which is a separate step from
    # measuring and is NOT part of the open runner: the bench never reads from
    # here and nothing on the measurement path writes to it. Your runs keep
    # their audio under runs/ regardless, so leaving all three unset is the
    # normal case and costs nothing. They exist because the published board
    # links each figure to the tape it came from, and that link has to be
    # anonymous-read for anyone to check it.
    azure_storage_connection_string: str | None = None
    azure_storage_container: str = "recordings"
    # Set only to publish URLs through a CDN or custom domain in front of the
    # account; empty means link the account's own primary blob endpoint. Never
    # point this at the RA-GRS `-secondary` host -- that replica is async and
    # can serve bytes older than the receipt claims.
    azure_public_base_url: str | None = None

    runs_dir: Path = _PKG_ROOT / "runs"
    log_level: str = "INFO"
    log_file: str = str(_PKG_ROOT / "logs" / "harness.log")
    max_calls_per_run: int = 300

    # Escape hatch for local tests only. Never set in a deployed environment.
    verify_webhook_signatures: bool = True

    def require_carrier(self) -> None:
        """Fail loudly before any call-placing code path runs half-configured."""
        if self.carrier != "plivo":
            raise RuntimeError(
                f"unknown CARRIER={self.carrier!r} -- the bench runs on plivo"
            )
        needed = (
            ("PLIVO_AUTH_ID", self.plivo_auth_id),
            ("PLIVO_AUTH_TOKEN", self.plivo_auth_token),
            ("PLIVO_FROM_NUMBER", self.plivo_from_number),
            ("PUBLIC_BASE_URL", self.public_base_url),
        )
        missing = [name for name, value in needed if not value]
        if missing:
            raise RuntimeError(
                f"missing in .env: {', '.join(missing)} -- copy .env.example and fill in"
            )


settings = Settings()
