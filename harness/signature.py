"""Plivo webhook authentication.

Two mechanisms, because Plivo offers two kinds of inbound connection:

- HTTP webhooks carry `X-Plivo-Signature-V3` / `X-Plivo-Signature-V3-Nonce`:
  HMAC-SHA256 over the exact URL Plivo called + sorted params + nonce, keyed with
  the account auth token. We delegate the math to the SDK's validator rather than
  reimplementing it. Accounts with several auth tokens send comma-separated
  signatures; the SDK handles matching any of them.

- The per-turn dialog callbacks cannot rely on a body signature alone, so each
  URL also embeds a per-call token from `secrets.token_urlsafe`. Unguessable URL
  == the auth. The registry only accepts tokens for calls this process placed.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from plivo.utils import validate_v3_signature

from .config import settings

log = logging.getLogger(__name__)


async def verify_plivo_webhook(request: Request) -> dict:
    """FastAPI dependency: authenticate a Plivo webhook and return its form params.

    Returns the parsed POST params on success so routes don't parse the body twice.
    Raises 403 on a bad or missing signature.
    """
    form = await request.form()
    params = {key: str(value) for key, value in form.items()}

    if not settings.verify_webhook_signatures:
        log.warning("webhook signature verification DISABLED by settings")
        return params

    if not settings.plivo_auth_token:
        raise HTTPException(status_code=500, detail="PLIVO_AUTH_TOKEN not configured")

    signature = request.headers.get("X-Plivo-Signature-V3", "")
    nonce = request.headers.get("X-Plivo-Signature-V3-Nonce", "")
    if not signature or not nonce:
        raise HTTPException(status_code=403, detail="missing signature headers")

    # Plivo signs the URL it was configured to call. Behind the tunnel/proxy the
    # app sees http://localhost; reconstruct the public URL from the configured
    # base plus the request path, which is exactly what we handed to Plivo.
    if settings.public_base_url:
        url = settings.public_base_url.rstrip("/") + request.url.path
    else:
        url = str(request.url)

    method = request.method
    valid = validate_v3_signature(
        method, url, nonce, settings.plivo_auth_token, signature,
        params if method == "POST" else None,
    )
    if not valid:
        log.warning("rejected webhook: bad V3 signature for %s", request.url.path)
        raise HTTPException(status_code=403, detail="invalid signature")
    return params
