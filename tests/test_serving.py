"""Bind + TLS config, and the guards that stop a silent failure.

A wrong bind does not raise anything. The calls go out, the carrier tries a URL
nobody is listening on, the answer webhook never arrives, and the caller side
looks exactly like a vendor that went quiet. Every check here exists so that
failure mode costs zero calls.
"""

from __future__ import annotations

import argparse
import os

import pytest

from harness.config import settings
from harness.serving import (
    add_server_args,
    describe,
    server_kwargs,
    server_problems,
)


def _args(**over):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    add_server_args(parser)
    args = parser.parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_defaults_are_todays_behaviour():
    """Loopback, no TLS -- exactly how every run so far was served."""
    kwargs = server_kwargs(_args())
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert "ssl_certfile" not in kwargs and "ssl_keyfile" not in kwargs


def test_loopback_without_tls_is_fine(monkeypatch):
    monkeypatch.setattr(settings, "public_base_url", "https://x.trycloudflare.com")
    assert server_problems(_args()) == []


def test_public_bind_without_tls_is_refused(monkeypatch):
    """Plaintext ws:// on a public interface would put the call audio in the
    clear, and Telnyx would be told to connect to it."""
    monkeypatch.setattr(settings, "public_base_url", "https://fde.example.dev")
    problems = server_problems(_args(host="0.0.0.0"))
    assert problems and "plaintext" in problems[0]


def test_half_a_tls_config_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "public_base_url", "https://fde.example.dev")
    cert = tmp_path / "fullchain.pem"
    cert.write_text("x")
    problems = server_problems(_args(ssl_certfile=str(cert)))
    assert any("together" in p for p in problems)


def test_missing_cert_is_caught_before_dialling(tmp_path, monkeypatch):
    """A skipped copy step must not surface as a failed call."""
    monkeypatch.setattr(settings, "public_base_url", "https://fde.example.dev:8443")
    problems = server_problems(_args(host="0.0.0.0", port=8443,
                                     ssl_certfile=str(tmp_path / "nope.pem"),
                                     ssl_keyfile=str(tmp_path / "nokey.pem")))
    assert len([p for p in problems if "does not exist" in p]) == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_root_owned_cert_is_reported_not_crashed(tmp_path, monkeypatch):
    """The Let's Encrypt layout, reproduced.

    `Path.is_file()` RAISES PermissionError when a parent directory is unreadable,
    so the first version of this guard crashed on exactly the mistake it existed
    to catch -- and only on a host where /etc/letsencrypt is root-only. It has to
    report, not raise.
    """
    private = tmp_path / "live"
    private.mkdir()
    cert = private / "fullchain.pem"
    cert.write_text("x")
    key = private / "privkey.pem"
    key.write_text("x")
    private.chmod(0o000)
    try:
        monkeypatch.setattr(settings, "public_base_url",
                            "https://fde.example.dev:8443")
        problems = server_problems(_args(host="0.0.0.0", port=8443,
                                         ssl_certfile=str(cert),
                                         ssl_keyfile=str(key)))
        assert problems, "an unreadable cert must be refused"
        assert all("not readable" in p or "does not exist" in p for p in problems)
    finally:
        private.chmod(0o755)


def test_port_mismatch_with_public_base_url_is_caught(tmp_path, monkeypatch):
    """The whole point: serving 8443 while telling Telnyx 443 loses every call."""
    cert = tmp_path / "c.pem"; cert.write_text("x")
    key = tmp_path / "k.pem"; key.write_text("x")
    monkeypatch.setattr(settings, "public_base_url", "https://fde.example.dev")
    problems = server_problems(_args(host="0.0.0.0", port=8443,
                                     ssl_certfile=str(cert), ssl_keyfile=str(key)))
    assert any("port 443" in p and "8443" in p for p in problems)
    assert any("https://fde.example.dev:8443" in p for p in problems)


def test_matching_port_passes(tmp_path, monkeypatch):
    cert = tmp_path / "c.pem"; cert.write_text("x")
    key = tmp_path / "k.pem"; key.write_text("x")
    monkeypatch.setattr(settings, "public_base_url", "https://fde.example.dev:8443")
    args = _args(host="0.0.0.0", port=8443,
                 ssl_certfile=str(cert), ssl_keyfile=str(key))
    assert server_problems(args) == []
    kwargs = server_kwargs(args)
    assert kwargs["ssl_certfile"] == str(cert)
    assert kwargs["ssl_keyfile"] == str(key)
    assert describe(args) == "serving https://0.0.0.0:8443"


def test_http_public_base_url_with_tls_is_refused(tmp_path, monkeypatch):
    cert = tmp_path / "c.pem"; cert.write_text("x")
    key = tmp_path / "k.pem"; key.write_text("x")
    monkeypatch.setattr(settings, "public_base_url", "http://fde.example.dev:8443")
    problems = server_problems(_args(host="0.0.0.0", port=8443,
                                     ssl_certfile=str(cert), ssl_keyfile=str(key)))
    assert any("must be https" in p for p in problems)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.5"])
def test_loopback_forms_are_all_recognised(host, monkeypatch):
    monkeypatch.setattr(settings, "public_base_url", "https://x.example.dev")
    assert server_problems(_args(host=host)) == []
