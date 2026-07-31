"""The Plivo XML the harness serves. Three elements, each load-bearing.

`<Record>` must come from the ANSWER XML, not from a later API call: API-started
recordings are mono, and only XML-started recordings are stereo (verified
against Plivo docs). Stereo -- our audio on one channel, the vendor's on the
other, one shared clock -- is what every reported number is measured from
. `recordSession="true"` records the whole call in the background
from answer to hangup, so recording is not an element that "runs" and blocks
the dialog. `playBeep="false"` because a beep would land in the recording next
to our speech and pollute the near-channel VAD.

`<GetInput inputType="speech">` is both our mouth and our turn-taking: the
nested `<Speak>` plays our line, then Plivo listens and POSTs the vendor's
transcript to `action`. Its endpointing decides when the CONVERSATION moves on
-- never when a measurement starts or ends. Both TTFAB endpoints are found in
the recording afterwards, so a slow or eager GetInput costs discard rate, never
a wrong number.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

# Plivo's documented ranges. An out-of-range value is rejected at call time --
# i.e. it presents as "the vendor never answered" -- so clamp instead.
EXECUTION_TIMEOUT_RANGE = (5, 60)
SPEECH_END_TIMEOUT_RANGE = (2, 10)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def build_record_element(*, callback_url: str, max_seconds: int = 300) -> str:
    """Background stereo recording for the whole call. The measurement tape."""
    return (
        "<Record "
        'recordSession="true" '
        'recordChannelType="stereo" '
        'fileFormat="wav" '
        'playBeep="false" '
        f'maxLength="{int(max_seconds)}" '
        f"callbackUrl={quoteattr(callback_url)} "
        'callbackMethod="POST" '
        'redirect="false"'
        "/>"
    )


def build_speak(text: str, *, voice: str, language: str) -> str:
    return (f"<Speak voice={quoteattr(voice)} language={quoteattr(language)}>"
            f"{escape(text)}</Speak>")


def build_getinput(
    *,
    action_url: str,
    prompt_text: str | None,
    voice: str,
    language: str,
    speech_end_timeout: int | str,
    execution_timeout: int,
) -> str:
    """Say our line (if any), then listen for the vendor and POST the transcript.

    `prompt_text=None` is the greeting listener: vendors greet on connect (the
    prototype confirmed Telnyx assistants do), so the first GetInput of a call
    speaks nothing and only listens. `speech_end_timeout` accepts "auto" --
    Plivo's own endpointing, which the prototype validated live -- or a fixed
    number of seconds.
    """
    if isinstance(speech_end_timeout, str):
        end_timeout: int | str = speech_end_timeout
    else:
        end_timeout = _clamp(speech_end_timeout, *SPEECH_END_TIMEOUT_RANGE)

    attrs = [
        f"action={quoteattr(action_url)}",
        'method="POST"',
        'inputType="speech"',
        f"language={quoteattr(language)}",
        f"speechEndTimeout={quoteattr(str(end_timeout))}",
        f'executionTimeout="{_clamp(execution_timeout, *EXECUTION_TIMEOUT_RANGE)}"',
    ]
    body = build_speak(prompt_text, voice=voice, language=language) if prompt_text else ""
    return f"<GetInput {' '.join(attrs)}>{body}</GetInput>"


def build_hangup() -> str:
    return "<Hangup/>"


def build_redirect(url: str) -> str:
    """Fetch the next XML from `url`. Elements after this are never reached.

    This is what a GetInput falls through to when its executionTimeout expires
    with NO speech at all. Plivo does not POST the action URL in that case --
    measured 2026-07-30, bench-telnyx-20260730-110251: five calls where the
    vendor produced no audio hung up at exactly executionTimeout with no action
    webhook, because the element after the GetInput was <Hangup/>. A silent
    vendor therefore cost the whole call instead of yielding the turns it
    failed to answer. Redirecting back into the dialog turns "call lost" into
    "these turns got no reply", which is a result rather than a gap.
    """
    return f"<Redirect>{escape(url)}</Redirect>"


def build_response(*elements: str) -> str:
    """Wrap elements in the document Plivo expects."""
    return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
            + "".join(e for e in elements if e) + "</Response>")
