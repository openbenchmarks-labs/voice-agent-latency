"""Fail early, and legibly, when a tool is run with the wrong interpreter.

Reached with an interpreter that lacks the analyzer's dependencies, the failure
surfaces as a ModuleNotFoundError several frames deep inside the analyzer, which
reads as a broken tool rather than a wrong `python`. Worse, the obvious response
-- pip installing the missing package into whatever venv is active -- produces a
SECOND, less obvious error (webrtcvad imports pkg_resources, which recent
Pythons no longer ship), sending the reader further from the actual problem.

This bites hardest where the runner sits inside a larger repo that has a
virtualenv of its own: the two are not interchangeable, and the wrong one is
already on PATH.

So: check up front, name the interpreter that works, and exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: What the analyzer needs to produce a number at all.
REQUIRED = ("numpy", "soundfile", "onnxruntime", "webrtcvad")

VOICE_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def require_voice_venv() -> None:
    """Exit with instructions unless the runner's dependencies are importable.

    Imports rather than checks for a spec: a package can be present but broken
    (an installed webrtcvad whose own import fails), and that must read the same
    as a missing one.
    """
    broken: list[str] = []
    for name in REQUIRED:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 -- any failure means unusable
            broken.append(f"{name} ({type(exc).__name__})")
    if not broken:
        return

    lines = [
        "wrong Python: this tool needs the voice runner's virtualenv.",
        f"  running:   {sys.executable}",
        f"  unusable:  {', '.join(broken)}",
    ]
    if VOICE_PYTHON.exists():
        lines += [
            "",
            "Run it with the runner's interpreter instead:",
            f"  cd {VOICE_PYTHON.parents[2]}",
            f"  .venv/bin/python tools/{Path(sys.argv[0]).name} "
            + " ".join(sys.argv[1:]),
        ]
    else:
        lines += [
            "",
            f"The runner's virtualenv is missing ({VOICE_PYTHON}). Create it:",
            "  python3 -m venv .venv",
            "  .venv/bin/pip install -r requirements.txt",
        ]
    lines.append(
        "\n(If another virtualenv is active, installing these into it is not the "
        "fix -- use the runner's own.)"
    )
    sys.exit("\n".join(lines))
