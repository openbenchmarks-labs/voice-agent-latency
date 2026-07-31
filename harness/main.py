"""Health check + shared logging setup.

The bench (harness.bench) and the probe (harness.probe_dialog) each build
their own FastAPI app; this module exists for `_configure_logging` and a
standalone health endpoint:

    uvicorn harness.main:app --port 8000
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI

from .config import settings


def _configure_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    os.makedirs(os.path.dirname(settings.log_file) or ".", exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_file, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


_configure_logging()
log = logging.getLogger(__name__)

app = FastAPI(title="voice-bench harness")


@app.get("/healthz")
async def healthz() -> dict:
    configured = bool(settings.plivo_auth_id and settings.plivo_auth_token
                      and settings.plivo_from_number)
    return {
        "ok": True,
        "carrier": settings.carrier,
        "public_base_url": settings.public_base_url,
        "carrier_configured": configured,
    }
