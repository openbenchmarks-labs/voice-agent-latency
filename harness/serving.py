"""How the harness exposes itself: bind address and its own TLS.

Until now the harness always bound loopback and something else -- cloudflared in
development -- carried the public HTTPS. That works, but the tunnel turned out to
be the reason the delay rig could not be used as a reference: the rig's replies
cross it too, and the resulting jitter (sigma ~850 ms) swamped the ~531 ms offset
the rig exists to measure.

Terminating TLS here removes the middle entirely. The harness listens on a real
port with a real certificate, so nothing sits between the carrier and us.

The functions below are shared by `harness.bench` and `harness.sweep` so the two
cannot drift into different flags or different guards -- every failure this
project has lost calls to was a config mismatch nobody was checking.
"""

from __future__ import annotations

import argparse
import ipaddress
from urllib.parse import urlparse

from .config import settings

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def add_server_args(parser: argparse.ArgumentParser) -> None:
    """Bind + TLS flags, identical in every entry point that serves webhooks."""
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address. Loopback (default) expects something "
                             "in front of us; a public address requires --ssl-* "
                             "so the media stream is not plaintext")
    parser.add_argument("--ssl-certfile", default=None,
                        help="PEM chain, to serve HTTPS ourselves instead of "
                             "sitting behind a proxy or tunnel")
    parser.add_argument("--ssl-keyfile", default=None,
                        help="private key for --ssl-certfile")


def _unreadable(path: str) -> str | None:
    """Why we cannot read `path`, or None if we can.

    Opens it rather than stat-ing it, for two reasons. Being able to read the file
    is what uvicorn actually needs, and `Path.is_file()` RAISES PermissionError
    when a parent directory is unreadable -- which is precisely the Let's Encrypt
    layout this check exists for. That crashed the guard on the exact mistake it
    was written to catch.
    """
    try:
        with open(path, "rb"):
            return None
    except FileNotFoundError:
        return "does not exist"
    except IsADirectoryError:
        return "is a directory, not a file"
    except PermissionError:
        return "is not readable by this user"
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"


def _is_loopback(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def server_problems(args) -> list[str]:
    """Reasons this bind + PUBLIC_BASE_URL combination would not work.

    Checked before any call is placed, because every one of these fails in a way
    that looks like something else: a wrong port means the answer webhook simply
    never arrives, and the caller side then looks like a vendor that went quiet.
    """
    problems: list[str] = []
    cert = getattr(args, "ssl_certfile", None)
    key = getattr(args, "ssl_keyfile", None)
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)

    if bool(cert) != bool(key):
        problems.append("--ssl-certfile and --ssl-keyfile must be given together")
    for label, path in (("--ssl-certfile", cert), ("--ssl-keyfile", key)):
        if path:
            reason = _unreadable(path)
            if reason:
                problems.append(f"{label} {path!r} {reason} (Let's Encrypt files "
                                f"are root-owned -- copy them somewhere this user "
                                f"can read)")

    serving_tls = bool(cert and key)
    if not _is_loopback(host) and not serving_tls:
        problems.append(
            f"--host {host} is public but no certificate was given. The media "
            f"stream would be plaintext ws://. Pass --ssl-certfile/--ssl-keyfile, "
            f"or bind loopback and put a proxy in front."
        )

    # PUBLIC_BASE_URL is what the carrier is told to call. If it disagrees with
    # what we are actually listening on, the calls die silently.
    base = settings.public_base_url
    if base and serving_tls:
        parsed = urlparse(base)
        if parsed.scheme != "https":
            problems.append(f"PUBLIC_BASE_URL is {parsed.scheme}:// but we are "
                            f"serving TLS -- it must be https://")
        stated = parsed.port or (443 if parsed.scheme == "https" else 80)
        if stated != port:
            problems.append(
                f"PUBLIC_BASE_URL points at port {stated} but we are listening on "
                f"{port}. The carrier would call the wrong port and the answer "
                f"webhook would never arrive. Set PUBLIC_BASE_URL to "
                f"https://{parsed.hostname}:{port}"
            )
    return problems


def server_kwargs(args) -> dict:
    """uvicorn.Config kwargs for this bind."""
    kwargs = {"host": getattr(args, "host", "127.0.0.1"),
              "port": getattr(args, "port", 8000),
              "log_level": "warning"}
    cert = getattr(args, "ssl_certfile", None)
    key = getattr(args, "ssl_keyfile", None)
    if cert and key:
        kwargs["ssl_certfile"] = cert
        kwargs["ssl_keyfile"] = key
    return kwargs


def describe(args) -> str:
    """One line for the run header, so the log records how we were exposed."""
    kwargs = server_kwargs(args)
    scheme = "https" if "ssl_certfile" in kwargs else "http"
    return f"serving {scheme}://{kwargs['host']}:{kwargs['port']}"
