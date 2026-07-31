"""The scripted dialog: script assembly, the state machine, and the receipt.

None of this measures anything -- the analyzer does that from the recording.
What these tests protect is that the conversation happens at all, in the right
order, and that the caller's exact configuration is recoverable afterwards.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from harness import dialog as D
from harness.answerxml import (
    EXECUTION_TIMEOUT_RANGE,
    build_getinput,
    build_record_element,
)
from harness.config import settings


@pytest.fixture(autouse=True)
def base_url(monkeypatch):
    monkeypatch.setattr(settings, "public_base_url", "https://bench.example:8443")


@pytest.fixture(scope="session")
def pinned_config(tmp_path_factory):
    """A dialog config the tests own.

    These tests describe how a script is ASSEMBLED, so they must not read the
    shipped config/dialog.yaml -- someone legitimately turning off the greeting
    turn there should not make the test suite fail, and a test that changes its
    verdict when an operator edits a config is not testing the code.
    """
    import yaml

    path = tmp_path_factory.mktemp("dialogcfg") / "dialog.yaml"
    path.write_text(yaml.safe_dump({
        "voice": "Polly.Joanna",
        "language": "en-US",
        "speech_end_timeout": "auto",
        "execution_timeout": 30,
        "greeting_timeout": 20,
        "include_greeting_turn": True,
        "include_goodbye_turn": True,
        "default_cases": ["price-basic"],
    }))
    return path


@pytest.fixture
def script(pinned_config):
    return D.build_script(["price-basic", "free-trial"], config_path=pinned_config)


@pytest.fixture
def call(tmp_path, script):
    return D.DialogSession(call_id="bench-x-001", out_dir=tmp_path, script=script)


def events(call) -> list[dict]:
    path = call.out_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def event_names(call) -> list[str]:
    return [e["event"] for e in events(call)]


# --------------------------------------------------------------------------- #
# Script assembly
# --------------------------------------------------------------------------- #


def test_the_script_is_t1_then_cases_then_t3(script):
    kinds = [t.kind for t in script.turns]
    assert kinds == ["t1_greeting", "case", "case", "t3_goodbye"]
    assert [t.index for t in script.turns] == [1, 2, 3, 4]
    assert script.turns[0].text == "Hi there."
    assert script.turns[-1].text.lower().startswith("that's all i needed")


def test_case_turns_carry_their_questions_and_expected_keywords(script):
    case = script.turns[1]
    assert case.case_id == "price-basic"
    assert "Basic plan" in case.text
    assert case.expect_keywords  # unique verifiable value from the scenarios file


def test_an_unknown_case_id_fails_before_any_call_is_placed():
    with pytest.raises(KeyError) as excinfo:
        D.build_script(["no-such-case"])
    assert "no-such-case" in str(excinfo.value)


def _script_with(tmp_path, pinned_config, cases=("price-basic",), **overrides):
    import yaml

    config = yaml.safe_load(pinned_config.read_text())
    config.update(overrides)
    path = tmp_path / "dialog.yaml"
    path.write_text(yaml.safe_dump(config))
    return D.build_script(list(cases) if cases else None, config_path=path)


def test_the_greeting_and_goodbye_turns_can_be_switched_off(tmp_path, pinned_config):
    """Both are optional, but the case questions are not."""
    bare = _script_with(tmp_path, pinned_config, include_greeting_turn=False,
                        include_goodbye_turn=False)
    assert [t.kind for t in bare.turns] == ["case"]
    assert bare.turns[0].index == 1

    no_greeting = _script_with(tmp_path, pinned_config, include_greeting_turn=False)
    assert [t.kind for t in no_greeting.turns] == ["case", "t3_goodbye"]
    assert [t.index for t in no_greeting.turns] == [1, 2]


def test_dropping_the_greeting_turn_is_a_different_instrument(tmp_path, pinned_config):
    """First-response cost does not vanish with the greeting turn -- it moves
    onto the first case question. Runs with and without must therefore not be
    poolable, and the receipt is what makes that visible."""
    with_greeting = _script_with(tmp_path, pinned_config, include_greeting_turn=True)
    without = _script_with(tmp_path, pinned_config, include_greeting_turn=False)
    assert with_greeting.receipt()["sha256"] != without.receipt()["sha256"]


def test_a_call_with_no_questions_is_refused(tmp_path, pinned_config):
    """Greeting and goodbye alone would place calls and measure nothing worth
    publishing."""
    with pytest.raises(KeyError) as excinfo:
        _script_with(tmp_path, pinned_config, cases=None, default_cases=[])
    assert "measures nothing" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The rotation
# --------------------------------------------------------------------------- #


def _rotating(tmp_path, pinned_config, n_calls, **overrides):
    import yaml

    config = yaml.safe_load(pinned_config.read_text())
    config.update(overrides)
    config.setdefault("case_rotation", [
        ["price-basic", "data-cap"],
        ["price-premium", "contract"],
        ["speed-basic", "free-trial"],
    ])
    path = tmp_path / "rot.yaml"
    path.write_text(yaml.safe_dump(config))
    return D.build_run_scripts(n_calls, config_path=path)


def test_consecutive_calls_ask_different_questions(tmp_path, pinned_config):
    """Three questions asked fifty times measures three questions."""
    scripts = _rotating(tmp_path, pinned_config, 3)
    asked = [[t.case_id for t in s.turns if t.case_id] for s in scripts]
    assert asked[0] != asked[1] != asked[2]
    assert asked[0] == ["price-basic", "data-cap"]
    assert asked[1] == ["price-premium", "contract"]


def test_the_rotation_wraps_when_there_are_more_calls_than_sets(tmp_path, pinned_config):
    scripts = _rotating(tmp_path, pinned_config, 7)
    asked = [[t.case_id for t in s.turns if t.case_id] for s in scripts]
    assert asked[3] == asked[0] and asked[4] == asked[1]
    assert len(scripts) == 7


def test_call_slot_i_is_identical_across_runs_and_vendors(tmp_path, pinned_config):
    """The property that makes two vendors comparable at all. Answers differ in
    length, and a platform that buffers its TTS starts speaking later on longer
    ones -- so a vendor that drew short questions would look faster for a
    reason that is not about the platform. Sampling would allow exactly that;
    rotating makes the questions a property of the call slot."""
    first = _rotating(tmp_path, pinned_config, 5)
    second = _rotating(tmp_path, pinned_config, 5)
    assert ([s.receipt()["sha256"] for s in first]
            == [s.receipt()["sha256"] for s in second])


def test_explicit_cases_collapse_the_rotation(tmp_path, pinned_config):
    """--cases is for debugging one question, so every call must ask it."""
    import yaml

    config = yaml.safe_load(pinned_config.read_text())
    config["case_rotation"] = [["price-basic"], ["contract"]]
    path = tmp_path / "rot.yaml"
    path.write_text(yaml.safe_dump(config))

    scripts = D.build_run_scripts(4, ["free-trial"], config_path=path)
    asked = {tuple(t.case_id for t in s.turns if t.case_id) for s in scripts}
    assert asked == {("free-trial",)}


def test_the_run_receipt_records_what_every_call_slot_asked(tmp_path, pinned_config):
    """One script's hash cannot describe a rotating run, so the run-level
    receipt records the plan and each call keeps its own hash."""
    scripts = _rotating(tmp_path, pinned_config, 5)
    plan = D.run_plan_receipt(scripts)

    assert plan["rotation_length"] == 3
    assert [c["call_index"] for c in plan["calls"]] == [0, 1, 2, 3, 4]
    assert plan["calls"][0]["cases"] == ["price-basic", "data-cap"]
    assert plan["calls"][3]["sha256"] == plan["calls"][0]["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", plan["sha256"])
    assert plan["voice"] == scripts[0].voice


def test_changing_the_rotation_changes_the_run_receipt(tmp_path, pinned_config):
    a = D.run_plan_receipt(_rotating(tmp_path, pinned_config, 3))
    b = D.run_plan_receipt(_rotating(
        tmp_path, pinned_config, 3,
        case_rotation=[["price-basic", "data-cap"], ["late-fee", "moving"]]))
    assert a["sha256"] != b["sha256"]


def test_a_rotation_naming_an_unknown_case_fails_before_dialling(tmp_path,
                                                                pinned_config):
    with pytest.raises(KeyError):
        _rotating(tmp_path, pinned_config, 2,
                  case_rotation=[["price-basic"], ["no-such-case"]])


def test_the_vendor_prompt_travels_with_the_script(script):
    """The questions and the prompt that can answer them are one artifact."""
    assert "Northwind" in script.system_prompt


def test_the_greeting_wait_is_not_shortened_by_the_vendors_idle_setting():
    """This used to return 60% of the vendor's idle-reprompt setting, which for
    a 10 s setting gave a 6 s window. The Telnyx assistant's greeting lands at
    9-10 s, so we began asking questions before it had greeted on SIX of SEVEN
    calls (bench-telnyx-20260730-{114637,115245}) and every one of those calls
    degraded. Talking over a vendor that has not greeted costs the call; the
    vendor's idle timer does not even start until it stops speaking."""
    assert D.greeting_timeout_for(10.0) == D.GREETING_TIMEOUT_S
    assert D.greeting_timeout_for(None) == D.GREETING_TIMEOUT_S
    # How LONG to wait is a tuning question settled in config/dialog.yaml and
    # still under investigation; that it does not depend on the vendor's idle
    # setting is the invariant, and the only thing pinned here.
    assert D.greeting_timeout_for(10.0) == D.greeting_timeout_for(2.0)


def test_the_greeting_wait_survives_a_silly_configured_value():
    assert D.greeting_timeout_for(None, ceiling=1.0) == D.MIN_GREETING_TIMEOUT_S


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #


def test_the_receipt_hashes_the_whole_caller_configuration(script):
    receipt = script.receipt()
    assert receipt["voice"] and receipt["language"]
    assert len(receipt["turns"]) == script.n_turns
    assert re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])


def test_the_same_script_hashes_the_same_and_a_changed_line_does_not(script, pinned_config):
    other = D.build_script(["price-basic", "free-trial"], config_path=pinned_config)
    assert other.receipt()["sha256"] == script.receipt()["sha256"]

    reworded = replace(script, turns=(replace(script.turns[0], text="Hello there."),)
                       + script.turns[1:])
    assert reworded.receipt()["sha256"] != script.receipt()["sha256"]


def test_a_different_voice_is_a_different_instrument(script):
    """Voice, language and endpointing are stimulus, not presentation: a run
    with a different voice is not comparable to one without it."""
    assert (replace(script, voice="Polly.Matthew").receipt()["sha256"]
            != script.receipt()["sha256"])
    assert (replace(script, speech_end_timeout=3).receipt()["sha256"]
            != script.receipt()["sha256"])


# --------------------------------------------------------------------------- #
# XML
# --------------------------------------------------------------------------- #


def test_the_answer_xml_arms_a_stereo_wav_recording_and_listens_first(call):
    """Stereo comes only from XML-started recordings, and the first GetInput
    must not speak: vendors greet the moment they connect."""
    xml = call.answer_xml({"CallUUID": "CALL-1"})
    assert 'recordSession="true"' in xml
    assert 'recordChannelType="stereo"' in xml
    assert 'fileFormat="wav"' in xml
    assert 'playBeep="false"' in xml
    assert "<Speak" not in xml
    assert "/webhooks/dialog/" in xml and "/greeting" in xml
    assert call.call_control_id == "CALL-1"
    assert call.state == D.STATE_AWAIT_GREETING


def test_getinput_timeouts_are_clamped_into_plivos_accepted_range():
    """Out-of-range values are rejected at call time and present as 'the vendor
    never answered', which is the hardest failure to diagnose."""
    xml = build_getinput(action_url="https://x/a", prompt_text="hi",
                         voice="Polly.Joanna", language="en-US",
                         speech_end_timeout=99, execution_timeout=999)
    assert f'executionTimeout="{EXECUTION_TIMEOUT_RANGE[1]}"' in xml
    assert 'speechEndTimeout="10"' in xml

    auto = build_getinput(action_url="https://x/a", prompt_text=None,
                          voice="Polly.Joanna", language="en-US",
                          speech_end_timeout="auto", execution_timeout=1)
    assert 'speechEndTimeout="auto"' in auto
    assert f'executionTimeout="{EXECUTION_TIMEOUT_RANGE[0]}"' in auto


def test_markup_in_a_question_cannot_break_out_of_the_speak_element():
    """Questions come from a JSON file, not from a Python literal -- a stray
    ampersand there would otherwise serve Plivo an unparseable document, which
    presents as a call that answers and immediately dies."""
    xml = build_getinput(action_url="https://x/a?a=1&b=2",
                         prompt_text="Fast & cheap? <script>",
                         voice="Polly.Joanna", language="en-US",
                         speech_end_timeout="auto", execution_timeout=30)
    assert "Fast &amp; cheap? &lt;script&gt;" in xml
    assert 'action="https://x/a?a=1&amp;b=2"' in xml


def test_the_recording_callback_url_is_ours():
    xml = build_record_element(callback_url="https://bench.example/webhooks/recording")
    assert 'callbackUrl="https://bench.example/webhooks/recording"' in xml


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #


def walk(call, transcripts):
    """Drive a whole call: greeting, then one reply per turn."""
    call.answer_xml({"CallUUID": "CALL-1"})
    xml = call.handle_action("greeting", {"Speech": transcripts[0]})
    for index in range(1, call.script.n_turns + 1):
        xml = call.handle_action(str(index), {"Speech": transcripts[index]})
    return xml


def test_a_whole_call_speaks_every_turn_in_order_then_hangs_up(call):
    final = walk(call, ["Thanks for calling Northwind Internet.",
                        "Sure, how can I help?",
                        "The Basic plan is $39 a month.",
                        "Yes, we offer a 30 day free trial.",
                        "Goodbye, and thanks for choosing Northwind Internet."])
    assert "<Hangup/>" in final
    assert call.state == D.STATE_COMPLETE
    assert call.dialog_done.is_set()
    assert call.turns_spoken == call.script.n_turns

    served = [e for e in events(call) if e["event"] == "turn_prompt_served"]
    assert [e["turn"] for e in served] == [1, 2, 3, 4]


def test_each_vendor_reply_is_recorded_against_the_turn_it_answers(call):
    walk(call, ["greeting", "hello", "The Basic plan is $39 a month.",
                "Yes, a 30 day free trial.", "Goodbye."])
    rows = {row["index"]: row for row in call.turn_metadata()}
    assert rows[2]["transcript"] == "The Basic plan is $39 a month."
    assert rows[2]["case_id"] == "price-basic"
    assert call.greeting_transcript == "greeting"


def test_answers_are_checked_against_the_cases_expected_keywords(call):
    walk(call, ["greeting", "hello", "The Basic plan is $39 a month.",
                "I am not sure about that.", "Goodbye."])
    rows = {row["index"]: row for row in call.turn_metadata()}
    assert rows[2]["answer_verified"] is True     # contains "39"
    assert rows[3]["answer_verified"] is False    # free-trial keyword missing
    # T1/T3 have nothing to verify -- absence of a claim, not a failed one.
    assert rows[1]["answer_verified"] is None
    assert rows[4]["answer_verified"] is None


def test_answer_matching_is_case_insensitive_and_substring_safe():
    assert D.answer_matches("It is THIRTY-NINE dollars", ("thirty-nine",)) is True
    assert D.answer_matches("no idea", ("39",)) is False
    assert D.answer_matches("anything", ()) is None


def test_a_timed_out_getinput_redirects_back_into_the_dialog(call):
    """Plivo does NOT post the action URL when a GetInput expires with no
    speech -- it runs the next element. With <Hangup/> there, five live calls
    against a silent vendor were lost outright instead of yielding the turns it
    failed to answer (bench-telnyx-20260730-110251). Every GetInput must be
    followed by a redirect back to its own action."""
    answer = call.answer_xml({"CallUUID": "CALL-1"})
    assert "<Redirect>" in answer and "<Hangup/>" not in answer
    assert f"/webhooks/dialog/{call.token}/greeting" in answer

    turn_xml = call.handle_action("greeting", {"Speech": "hi"})
    assert "<Redirect>" in turn_xml and "<Hangup/>" not in turn_xml
    assert f"/webhooks/dialog/{call.token}/1" in turn_xml


def test_a_silent_vendor_still_gets_asked_every_question(call):
    """The redirect path end to end: no speech anywhere, yet all four turns are
    spoken and the call ends cleanly. Those turns become no_response, which is
    a measurement; a dropped call is not."""
    call.answer_xml({"CallUUID": "CALL-1"})
    call.handle_action("greeting", {})            # timed out, no Speech
    for index in range(1, call.script.n_turns + 1):
        xml = call.handle_action(str(index), {})  # timed out again
    assert call.turns_spoken == call.script.n_turns
    assert "<Hangup/>" in xml
    assert all(row["reply_timed_out"] for row in call.turn_metadata()
               if row["spoken"])


def test_a_redirect_storm_cannot_keep_a_call_alive(call):
    """Every GetInput now redirects on timeout, so a delivery loop would burn
    money for as long as the carrier allowed."""
    call.answer_xml({"CallUUID": "CALL-1"})
    for _ in range(60):
        xml = call.handle_action("1", {})
    assert "<Hangup/>" in xml
    assert "action_storm" in event_names(call)
    assert call.dialog_done.is_set()


def test_a_step_we_never_issued_ends_the_call_instead_of_raising(call):
    """The dialog webhook is publicly reachable; a 500 here would strand a
    live call mid-conversation."""
    call.answer_xml({"CallUUID": "CALL-1"})
    assert "<Hangup/>" in call.handle_action("not-a-turn", {})
    assert "<Hangup/>" in call.handle_action("99", {})
    assert "action_unknown_step" in event_names(call)


def test_a_vendor_that_never_greets_still_gets_asked_the_questions(call):
    """A silent greeting window is a finding, not a reason to abandon a call
    that is already connected and already being recorded."""
    call.answer_xml({"CallUUID": "CALL-1"})
    xml = call.handle_action("greeting", {"Speech": ""})
    assert call.greeting_timed_out is True
    assert "greeting_timeout" in event_names(call)
    assert "<Speak" in xml and call.state == D.turn_state(1)


def test_a_turn_with_no_reply_advances_and_is_marked_timed_out(call):
    walk(call, ["greeting", "hello", "", "yes a free trial", "bye"])
    rows = {row["index"]: row for row in call.turn_metadata()}
    assert rows[2]["reply_timed_out"] is True
    assert rows[3]["transcript"] == "yes a free trial"   # later turns unaffected
    assert "reply_timeout" in event_names(call)


def test_a_replayed_webhook_repeats_its_xml_instead_of_skipping_a_turn(call):
    """Plivo retries. A duplicate delivery that advanced the script would skip
    a question and silently shorten the call."""
    call.answer_xml({"CallUUID": "CALL-1"})
    call.handle_action("greeting", {"Speech": "hi"})
    first = call.handle_action("1", {"Speech": "hello"})
    again = call.handle_action("1", {"Speech": "hello"})
    assert again == first
    assert "action_replayed" in event_names(call)
    assert call.turns_spoken == 2


def test_every_turn_is_listed_in_metadata_even_if_the_call_died_early(call):
    call.answer_xml({"CallUUID": "CALL-1"})
    call.handle_action("greeting", {"Speech": "hi"})
    call.handle_action("1", {"Speech": "hello"})
    rows = call.turn_metadata()
    assert [r["index"] for r in rows] == [1, 2, 3, 4]
    assert rows[1]["spoken"] is True     # turn 2 was asked
    assert rows[3]["spoken"] is False    # turn 4 never happened
    assert rows[3]["transcript"] == ""


def test_answering_is_signalled_separately_from_finishing(call):
    """A call nobody picks up has no conversation to wait out. Waiting the full
    conversation deadline for one cost three minutes per unanswered call
    (bench-telnyx-20260730-111843, call-004)."""
    assert not call.answered.is_set()
    call.answer_xml({"CallUUID": "CALL-1"})
    assert call.answered.is_set()
    assert not call.dialog_done.is_set()


def test_a_fixed_endpointing_value_reaches_the_getinput(call, pinned_config):
    """"auto" ended a real greeting after "thanks for calling" and we spoke
    over the rest. A fixed value demands real silence first."""
    script = D.build_script(["price-basic"], config_path=pinned_config)
    fixed = replace(script, speech_end_timeout=2)
    session = D.DialogSession(call_id="x", out_dir=call.out_dir, script=fixed)
    assert 'speechEndTimeout="2"' in session.answer_xml({"CallUUID": "C"})


def test_the_hangup_webhook_releases_the_bench_loop(call):
    call.note_hangup({"CallUUID": "CALL-1"})
    assert call.hangup_seen.is_set() and call.dialog_done.is_set()


def test_the_recording_callback_is_captured_as_the_fast_path(call):
    call.note_recording_callback({"RecordUrl": "https://media.plivo.com/r.wav",
                                  "RecordingID": "REC-1"})
    assert call.recording_url == "https://media.plivo.com/r.wav"


# --------------------------------------------------------------------------- #
# Channel map
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The webhook surface
# --------------------------------------------------------------------------- #


def test_the_bench_app_serves_a_whole_conversation(tmp_path, monkeypatch):
    """End to end over HTTP, because the state machine passing in isolation
    says nothing about whether Plivo can actually reach it: the route shapes,
    the token in the action URL and the XML content type all have to line up."""
    from fastapi.testclient import TestClient

    from harness.bench import Registry, build_bench_app

    monkeypatch.setattr(settings, "verify_webhook_signatures", False)
    monkeypatch.setattr(settings, "carrier", "plivo")

    registry = Registry()
    call = D.DialogSession(call_id="call-000", out_dir=tmp_path,
                           script=D.build_script(["price-basic"]))
    registry.current = call
    client = TestClient(build_bench_app(registry))

    answer = client.post("/webhooks/answer", data={"CallUUID": "CALL-1"})
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("application/xml")
    assert "recordChannelType=\"stereo\"" in answer.text
    assert call.call_control_id == "CALL-1"

    base = f"/webhooks/dialog/{call.token}"
    assert client.post(f"{base}/greeting",
                       data={"Speech": "Thanks for calling Northwind Internet."}
                       ).status_code == 200
    for index in range(1, call.script.n_turns + 1):
        reply = client.post(f"{base}/{index}", data={"Speech": "the Basic plan is $39"})
        assert reply.status_code == 200
    assert "<Hangup/>" in reply.text
    assert call.dialog_done.is_set()

    assert client.post("/webhooks/recording",
                       data={"RecordUrl": "https://media.plivo.com/r.wav"}
                       ).status_code == 200
    assert call.recording_url == "https://media.plivo.com/r.wav"

    assert client.post("/webhooks/hangup", data={"CallDuration": "31"}).status_code == 200
    assert call.hangup_seen.is_set()


def test_a_dialog_webhook_with_the_wrong_token_is_refused(tmp_path, monkeypatch):
    """The action URL is what puts words in our mouth and hangs us up; the
    per-call token is what stops anyone else driving the call."""
    from fastapi.testclient import TestClient

    from harness.bench import Registry, build_bench_app

    monkeypatch.setattr(settings, "verify_webhook_signatures", False)
    monkeypatch.setattr(settings, "carrier", "plivo")

    registry = Registry()
    registry.current = D.DialogSession(call_id="call-000", out_dir=tmp_path,
                                       script=D.build_script(["price-basic"]))
    client = TestClient(build_bench_app(registry))
    assert client.post("/webhooks/dialog/not-the-token/greeting",
                       data={"Speech": "hi"}).status_code == 409


def test_a_webhook_between_calls_is_refused_rather_than_misattributed(monkeypatch):
    """A late webhook from the previous call must not be recorded against the
    next one -- that is how a stale transcript ends up in someone's metadata."""
    from fastapi.testclient import TestClient

    from harness.bench import Registry, build_bench_app

    monkeypatch.setattr(settings, "verify_webhook_signatures", False)
    monkeypatch.setattr(settings, "carrier", "plivo")

    client = TestClient(build_bench_app(Registry()))
    assert client.post("/webhooks/answer", data={"CallUUID": "X"}).status_code == 409


# --------------------------------------------------------------------------- #
# Channel map
# --------------------------------------------------------------------------- #


def test_the_channel_map_declares_its_own_provenance():
    """A silently inverted channel map would swap every t1 and t2 and still
    produce plausible numbers, so the map always travels with its source."""
    mapping = D.channel_map()
    assert mapping["near"] != mapping["far"]
    assert mapping["source"]


def test_our_leg_is_channel_one_as_measured():
    """Pinned from probe-dialog-20260729-214809, and deliberately NOT the
    intuitive order: on that call our two spoken lines were on channel 1 and
    the callee's speech on channel 0, corroborated by channel 1 being
    digitally silent (-240 dBFS) between utterances while channel 0 carried
    line noise. Reverting this to the obvious-looking {near: 0, far: 1} would
    measure their speech-end against our reply and still look plausible."""
    assert D.PLIVO_CHANNEL_MAP == {"near": 1, "far": 0}
    assert not D.channel_map_is_provisional()
    assert "probe-dialog-" in D.CHANNEL_MAP_SOURCE
