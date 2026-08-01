"""Turn saved audio into one reported result.

Pure: no network, no credentials, no database. `python -m analyzer runs/<run_id>`
reproduces every published number from files on disk, which is what lets a vendor
disputing their row re-derive it themselves.

Two paths produce a TTFAB, and they are not equal in standing:

Every reported number comes from one place: the carrier's stereo recording,
both directions on one clock. t2 is VAD on the far channel. t1 is VAD plus an
energy refinement on the near channel in scripted-dialog mode, or correlation
against a known waveform in reference mode -- which is how the analyzer is
validated against a known answer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import onset as O
from .correlate import MIN_PSR, locate_t1
from .resample import samples_to_ms

# Bump on any change that can move a reported number. Stored in every result so a
# mixed-version run directory is detectable rather than silently averaged.
# 1.1.0 (2026-07-28): noise floor became a low percentile of frame energy rather
#   than the RMS of the first 300 ms, after the first Telnyx bench run showed half
#   of real recordings open mid-greeting (which inflated the floor and pushed
#   onsets ~10-40 ms late); TTFG is withheld on such recordings. Results from
#   1.0.0 are NOT comparable -- re-run the analyzer over old runs.
# 1.2.0 (2026-07-28): multi-turn. Every call now reports a `turns` list, with
#   discards at the turn level so one bad turn does not waste the rest. Top-level
#   fields mirror turn 1, so single-turn results are unchanged from 1.1.0.
# 1.3.0 (2026-07-28): per-turn confidence is re-scored in a two-sided window, so
#   an early turn is no longer judged against later turns' audio. The first live
#   4-turn run discarded turn 1 of every call at PSR ~2.08 for exactly that reason
#   while its position was correct; re-scored it reads ~4.1 and passes. Positions
#   and TTFAB values are unchanged -- only confidence, and therefore discard
#   verdicts, move.
# 2.0.0 (2026-07-30): scripted-dialog mode. Our side of the call is now a live
#   Plivo TTS dialog rather than committed audio, so there is no waveform to
#   correlate: t1 becomes the refined END of our own speech on the near channel
#   (analyzer/onset.py _refine_offset). On this path `psr`, `psr_global` and
#   `drift_ms` are always None, `unlocatable`/`drift`/`out_of_order` never fire,
#   and three discards replace them -- `double_talk`, `offset_disagree`,
#   `turn_count_mismatch` -- plus `channel_map_suspect` at call level. Numbers
#   from this path are NOT comparable with reference-mode numbers: t1 means a
#   different thing and carries VAD-class error rather than being sample-exact.
#   Reference mode is unchanged, so archived runs re-analyze identically apart
#   from this version stamp. `metadata.json` selects the path via `mode`.
# 2.1.0 (2026-07-30): `idle_filler` is no longer applied when the recording
#   opened mid-greeting. On such a tape the pre-stimulus count is a count of
#   greeting FRAGMENTS, so a vendor fast enough to greet before the recorder
#   armed had every call discarded whole -- observed on the first Vapi run
#   (bench-vapi-20260730-141015), where a flawless four-turn conversation was
#   thrown away with three clean per-turn measurements in it. Such calls now
#   carry `idle_filler_unassessable_clipped_greeting` and keep their turns. No
#   archived run has both conditions, so committed results are unchanged.
# 2.2.0 (2026-07-30): the onset cross-check now distinguishes a CONTRADICTING
#   second opinion from an ABSENT one. webrtcvad sometimes fails to find an
#   utterance at all (it never declares the preceding silence, so there is no
#   rising edge to match), and the nearest-edge distance then measures an
#   unrelated event. Such turns carry `vad_crosscheck_unavailable` and keep
#   their measurement; a genuine disagreement still discards. See
#   VAD_NO_OPINION_MS for the measured separation. No archived turn exceeds
#   100 ms, so committed results are unchanged.
# 2.3.0 (2026-07-30): the onset cross-check became ASYMMETRIC. Our onset landing
#   EARLIER than webrtcvad's is the failure it was built for (energy firing on
#   comfort noise, ~200 ms of false speed) and still discards at 40 ms; landing
#   LATER means webrtcvad tripped on a breath or codec artifact ahead of the
#   real speech, and is tolerated to MAX_VAD_ONSET_LATE_MS. 7 of 9 real
#   disagreements were the conservative direction. Turns now carry
#   `vad_disagreement_signed_ms` so the direction survives into the receipt.
#   Unlike 2.1.0 and 2.2.0 this CAN move an archived verdict: one committed turn
#   (bench-telnyx-20260730-111843 call-003 turn 3, 100 ms) sits in the affected
#   band and would be kept rather than discarded if its sign is positive. Its
#   audio is not committed, so re-deriving it needs the run directory.
# 2.4.0 (2026-07-31): in DIALOG mode `idle_filler` records a flag instead of
#   discarding the call, and barge detection measures against the LAST
#   pre-stimulus utterance rather than the first. Both follow from one fact: a
#   vendor may pause mid-greeting for longer than GREETING_MERGE_GAP_MS, and the
#   pre-stimulus region is then a split greeting rather than an idle re-prompt.
#   Measured on bench-vapi-20260731-152959: Vapi greets in two phrases with a
#   960-1224 ms pause (86 of 98 calls; no other vendor ever splits, 0/392), so
#   22 calls were discarded whole -- and WHICH 22 was decided by whether the
#   carrier's recorder happened to catch the greeting's first ~100 ms loudly
#   enough to trip CLIPPED_START_RMS and earn the 2.1.0 exemption. Recovering
#   the 78 turns in them moves that run's p50 by 1 ms (1563 -> 1564) and its p95
#   by 12 ms (2008.4 -> 2020.8): the discard was costing n, not protecting the
#   number. The other four vendors are untouched -- 0 calls, 0 turns, 0 ms.
#   Safe because dialog mode searches for a reply with first_after(our turn
#   START), so nothing before our first word can be selected as any turn's
#   reply -- the rule was guarding a path that cannot be reached. Deliberately
#   introduces NO new threshold: a millisecond value chosen against Vapi's pause
#   would simply relocate the failure to the next vendor. Reference mode is
#   UNCHANGED, so Gate A's `idle_filler` fixture and every archived
#   reference-mode result are unaffected; dialog-mode runs must be re-derived.
ANALYZER_VERSION = "2.4.0"

# Discard thresholds. These are the frozen discard rules; they belong
# in METHODOLOGY.md with a commit date before the first measurement call.
MAX_DRIFT_MS = 50.0
# The onset cross-check is ASYMMETRIC, because the two ways it can disagree are
# not equally dangerous.
#
# OURS EARLIER than webrtcvad is the failure this check exists for: the energy
# pass fires on comfort noise and the vendor looks ~200 ms faster than it is
# (see analyzer/onset.py). Kept tight.
#
# OURS LATER means webrtcvad tripped on a breath, a codec artifact or line noise
# ahead of the real speech and we did not -- it fires readily on telephone audio
# (18 rising edges in a window where Silero found 3 utterances). Being
# conservative is a different kind of mistake, and a symmetric 40 ms rule was
# discarding it as though it were the dangerous one: of 9 disagreements measured
# 2026-07-30 across 87 real turns, 7 were us being LATER by 44-104 ms, and their
# TTFAB values were indistinguishable from the turns that passed (Bland: kept
# median 1526 ms, discarded median 1532 ms). The late bound is set well above
# that observed spread and well below a genuinely missed onset, which lands the
# measurement on the wrong utterance entirely.
MAX_VAD_DISAGREEMENT_MS = O.DISAGREEMENT_FLAG_MS      # ours EARLIER: the risky side
MAX_VAD_ONSET_LATE_MS = 250.0                         # ours LATER: the safe side

# Beyond this, webrtcvad did not find our utterance AT ALL, and the "nearest
# edge" distance is measuring the gap to some unrelated event. That is an
# absence of a second opinion, not a contradicting one, and discarding on it
# throws away good measurements.
#
# The two cases are cleanly separable, measured 2026-07-30 over 73 turns:
# genuine disagreements top out at 100 ms (max 94 ms on Vapi, 100 ms on
# Telnyx), while the misses start at 1080 ms. Nothing lands in between. On the
# first multi-call Vapi run this rule discarded 7 turns whose TTFAB (median
# 1714 ms, range 1352-1820) sat squarely inside the kept distribution (median
# 1646 ms, range 368-2302) -- ordinary measurements refused by a referee that
# had failed to see the utterance.
#
# webrtcvad's aggressiveness is NOT the fix: 3 is already the best setting by a
# wide margin on this audio (median disagreement 12 ms, against 2172-2794 ms at
# 0/1/2), so the misses are its floor rather than a tuning error.
VAD_NO_OPINION_MS = 500.0
RESPONSE_TIMEOUT_MS = 15_000.0
GREETING_TIMEOUT_MS = 20_000.0
CONTENT_MIN_DURATION_MS = 400.0

# Detection tolerance when deciding whether we talked over the greeting. Onsets
# carry a few ms of error, so a hard comparison would flag clean calls.
BARGE_GREETING_MARGIN_MS = 50.0

# Segments closer together than this belong to the same utterance. Used only to
# decide how far the greeting extends and whether a separate idle prompt followed
# it -- see OnsetAnalysis.utterance_groups.
#
# Measured on the fixtures: pauses inside a greeting run 174-200 ms, while an idle
# "are you still there?" prompt arrives after ~1130 ms of silence. 700 ms sits
# cleanly between the two. Anything below ~250 ms would split a greeting and
# discard the call as an idle prompt; anything above ~1100 ms would absorb a real
# idle prompt into the greeting and miss it.
GREETING_MERGE_GAP_MS = 700.0

# Scripted-dialog mode: segments of OUR speech closer together than this belong
# to the same turn. Same reasoning as GREETING_MERGE_GAP_MS, applied to the near
# channel: TTS pauses at commas and full stops, which exceeds MIN_SILENCE_MS and
# splits one spoken line into several segments, while consecutive turns are
# separated by the vendor's whole reply -- seconds, not milliseconds.
TURN_MERGE_GAP_MS = 700.0

# Scripted-dialog mode: if the vendor is still speaking this close to the end of
# our turn, we were talking over each other and t1 is not a clean anchor.
DOUBLE_TALK_WINDOW_MS = 300.0

# Scripted-dialog mode: how far our own t1 may sit from webrtcvad's nearest
# falling edge before the turn is discarded.
#
# Much wider than the onset cross-check's 40 ms, and not by preference. Speech
# ends by decaying, so where it "stops" is genuinely ambiguous in a way an onset
# is not, and webrtcvad's 20 ms frames plus its own hangover put its falling
# edge systematically late: measured on the clean fixtures the two detectors sit
# ~66 ms apart while t1 itself is within ~2 ms of truth. At 40 ms this rule
# would discard every turn of a perfect call. What it is here to catch is t1
# landing in the wrong PLACE -- a mid-sentence pause, or the wrong utterance
# entirely -- which is a hundreds-of-ms error, not a tens-of-ms one.
MAX_OFFSET_DISAGREEMENT_MS = 200.0

# Scripted-dialog mode: our own audio leaking into the vendor's channel.
#
# A recording where the far channel carries an echo of us is not hypothetical,
# and it is the most dangerous artifact on this path: the leaked copy of our own
# speech starts BEFORE the vendor replies, so a detector that believes it
# reports a wildly early -- often negative -- TTFAB that still looks like a
# number. Leakage is heavily attenuated, so it is separable by level: a far
# segment overlapping our speech and this far below it is us, not them. Real
# double-talk arrives at comparable level and is caught by DOUBLE_TALK_WINDOW_MS
# instead.
BLEED_MIN_ATTENUATION_DB = 12.0

# If the far channel is already this loud in its opening frames, the recorder came
# up mid-utterance and the greeting on the tape is clipped.
#
# Why it matters: recording starts when our record_start takes effect, which is
# necessarily after the vendor began greeting. Measured on the first Telnyx bench
# run: 5/10 recordings began mid-speech. Consequences differ per metric --
#   - TTFAB is untouched (both endpoints are seconds later)
#   - TTFG becomes meaningless: `greeting_onset` is wherever the clip began, which
#     produced an apparent 8 s p95 that was pure artifact. So TTFG is withheld.
#   - the `idle_filler` rule loses its reference: with no greeting on the tape the
#     vendor's idle prompt becomes "the greeting" and a polluted turn passes. The
#     flag lets those calls be excluded downstream.
CLIPPED_START_RMS = 200.0
CLIPPED_START_WINDOW_MS = 120.0


def _barge_margin(rate: int) -> int:
    """Overlap tolerance before calling a turn an interruption.

    Onsets and segment ends both carry a few ms of detection error, and the reply
    end is the noisier of the two, so a hard comparison would flag clean turns.
    """
    return int(round(BARGE_GREETING_MARGIN_MS * rate / 1000.0))


def _starts_mid_speech(far: np.ndarray, rate: int) -> bool:
    n = max(1, int(round(CLIPPED_START_WINDOW_MS * rate / 1000.0)))
    head = far[:n].astype(np.float64)
    if head.size == 0:
        return False
    return float(np.sqrt(np.mean(head * head))) > CLIPPED_START_RMS


@dataclass
class TurnResult:
    """One measured turn. A call yields several of these.

    Discards live HERE rather than on the call: if turn 3 fails, turns 1, 2 and 4
    are still valid measurements, and throwing away a whole call for one bad turn
    would waste three good ones.
    """

    index: int = 0
    stimulus_start_ms: float | None = None
    t1_ms: float | None = None
    t2_ms: float | None = None
    ttfab_onset_ms: float | None = None
    ttfab_content_ms: float | None = None
    vendor_response_duration_ms: float | None = None
    psr: float | None = None
    psr_global: float | None = None
    drift_ms: float | None = None
    vad_disagreement_ms: float | None = None
    #: The same figure with its sign: positive means OUR onset is later than
    #: webrtcvad's. Kept because the direction decides whether a disagreement is
    #: the dangerous kind (we fired early on noise) or the conservative kind.
    vad_disagreement_signed_ms: float | None = None
    flags: list[str] = field(default_factory=list)
    discard_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.discard_reason is None


@dataclass
class Result:
    """One call's measurement. Serialised verbatim to result.json."""

    call_id: str = ""
    run_id: str = ""
    vendor: str = ""
    kind: str = ""
    stimulus_id: str | None = None
    known_delay_ms: float | None = None

    source: str = "recording"
    rate: int = 0

    greeting_onset_ms: float | None = None
    greeting_end_ms: float | None = None
    stimulus_start_ms: float | None = None
    t1_ms: float | None = None
    t2_ms: float | None = None

    ttfab_onset_ms: float | None = None
    ttfab_content_ms: float | None = None
    ttfg_ms: float | None = None
    vendor_response_duration_ms: float | None = None

    psr: float | None = None
    psr_global: float | None = None
    drift_ms: float | None = None
    vad_disagreement_ms: float | None = None
    noise_floor_dbfs: float | None = None

    flags: list[str] = field(default_factory=list)
    discard_reason: str | None = None
    analyzer_version: str = ANALYZER_VERSION

    # Multi-turn. Single-turn calls carry exactly one entry, and the top-level
    # fields above mirror turn 1 so every existing consumer keeps working.
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.discard_reason is None

    @property
    def usable_turns(self) -> list[TurnResult]:
        """Turns worth reporting. Empty if the call itself failed."""
        if self.discard_reason is not None:
            return []
        return [t for t in self.turns if t.usable]


def measure_recording(near: np.ndarray, far: np.ndarray, reference: np.ndarray,
                      rate: int, *, base: Result | None = None) -> Result:
    """Measure a single-turn stereo recording. The primary path.

    Thin wrapper over measure_turns: one turn is just the N=1 case.
    """
    return measure_turns(near, far, [reference], rate, base=base)


def measure_turns(near: np.ndarray, far: np.ndarray,
                  references: list[np.ndarray], rate: int, *,
                  base: Result | None = None) -> Result:
    """Measure every turn in a stereo recording.

    Each turn has its own known reference, so turn i's speech-end is found by the
    same matched filter with reference i -- four turns is four searches, no
    segmentation of our own audio and no guessing which utterance was which. Two
    things keep the searches honest:

      * each search starts AFTER the previous turn's t1, so a reference cannot
        match an earlier turn's audio (and the search shrinks as we go)
      * a monotonicity check: if a turn's t1 does not advance, the reference
        matched the wrong place and that turn is discarded rather than reported

    The greeting, TTFG and idle-prompt logic stay keyed to TURN 1's stimulus
    start, exactly as in the single-turn case.
    """
    r = base or Result()
    r.source = "recording"
    r.rate = rate
    r.analyzer_version = ANALYZER_VERSION

    ms = lambda n: samples_to_ms(n, rate)  # noqa: E731

    if near.size == 0 or far.size == 0 or not references:
        r.discard_reason = "audio_missing"
        return r

    far_analysis = O.analyze(far, rate)
    r.noise_floor_dbfs = (
        None if not np.isfinite(far_analysis.noise_floor_dbfs)
        else float(far_analysis.noise_floor_dbfs)
    )

    clipped = _starts_mid_speech(far, rate)
    if clipped:
        r.flags.append("recording_started_mid_speech")

    # --- locate each turn ---------------------------------------------------- #
    after = 0
    previous_t1: int | None = None
    estimates: list[tuple[TurnResult, object]] = []

    for index, reference in enumerate(references, start=1):
        turn = TurnResult(index=index)

        # Searching only the tail after the previous turn keeps references from
        # cross-matching and makes later searches cheaper.
        window = near[after:]
        if window.size <= reference.size:
            turn.discard_reason = "unlocatable"
            turn.flags.append("no_audio_left_to_search")
            estimates.append((turn, None))
            continue

        estimate = locate_t1(window, reference, rate)
        t1_abs = estimate.t1 + after
        start_abs = estimate.stimulus_start + after

        turn.stimulus_start_ms = ms(start_abs)
        turn.t1_ms = ms(t1_abs)
        turn.psr = estimate.psr
        turn.psr_global = estimate.psr_global
        turn.drift_ms = estimate.drift_ms

        if previous_t1 is not None and t1_abs <= previous_t1:
            # The reference matched at or before the previous turn: the pairing is
            # wrong, so any interval computed from it would be fiction.
            turn.discard_reason = "out_of_order"
            estimates.append((turn, None))
            continue

        previous_t1 = t1_abs
        after = t1_abs
        estimates.append((turn, (estimate, t1_abs, start_abs)))

    # --- second pass: re-score confidence in a two-sided window -------------- #
    # Pass 1 bounds each search below (after the previous turn) but not above, so
    # an early turn's match competes against every LATER turn's audio. With one
    # synthetic voice asking similar questions those rivals are strong: the first
    # live 4-turn run scored turn 1 at PSR 2.07-2.09 -- under the 3.0 floor, so it
    # was discarded -- while turns 2-4 scored 3.1-7.0 purely because their windows
    # were smaller. The positions were right; only the confidence was polluted.
    #
    # Competition from a DIFFERENT turn is not genuine ambiguity: we know the
    # script order. So re-locate each turn inside [previous t1, next stimulus
    # start], where the only thing it can match is itself.
    located_positions = [(i, loc) for i, (_, loc) in enumerate(estimates)
                         if loc is not None]
    for order, (position, loc) in enumerate(located_positions):
        turn, _ = estimates[position]
        reference = references[position]

        lo = located_positions[order - 1][1][1] if order > 0 else 0
        hi = (located_positions[order + 1][1][2]
              if order + 1 < len(located_positions) else near.size)
        if hi - lo <= reference.size:
            continue  # window too tight to re-score; keep the pass-1 estimate

        rescored = locate_t1(near[lo:hi], reference, rate)
        t1_abs = rescored.t1 + lo
        start_abs = rescored.stimulus_start + lo

        turn.stimulus_start_ms = ms(start_abs)
        turn.t1_ms = ms(t1_abs)
        turn.psr = rescored.psr
        turn.psr_global = rescored.psr_global
        turn.drift_ms = rescored.drift_ms
        estimates[position] = (turn, (rescored, t1_abs, start_abs))

    # --- greeting / TTFG / idle prompt, keyed to turn 1 ---------------------- #
    first_located = next((loc for _, loc in estimates if loc is not None), None)
    pre_stimulus: list = []
    if first_located is not None:
        _, _, first_start = first_located
        pre_stimulus = far_analysis.utterance_groups(
            GREETING_MERGE_GAP_MS, before=first_start
        )
        if pre_stimulus:
            greeting = pre_stimulus[0]
            r.greeting_onset_ms = ms(greeting.start)
            r.greeting_end_ms = ms(greeting.end)
            # Anchored to the start of the saved audio, not to carrier answer
            # -- and withheld when the tape opens mid-speech,
            # because then it measures where the clip began, not where the vendor
            # started talking.
            r.ttfg_ms = None if clipped else ms(greeting.start)
        if len(pre_stimulus) > 1:
            r.flags.append(f"pre_stimulus_utterances={len(pre_stimulus)}")

    # --- the vendor's reply to each turn ------------------------------------- #
    for position, (turn, located) in enumerate(estimates):
        if located is None:
            continue
        estimate, t1_abs, start_abs = located

        # The reply belongs to this turn only if it starts before the NEXT turn's
        # audio does; anything later is the next turn's business.
        next_start = None
        for _, later in estimates[position + 1:]:
            if later is not None:
                next_start = later[2]
                break

        response = far_analysis.first_after(start_abs)
        if response is not None and next_start is not None and response.start >= next_start:
            response = None

        if response is not None:
            turn.t2_ms = ms(response.start)
            turn.ttfab_onset_ms = ms(response.start - t1_abs)
            turn.vendor_response_duration_ms = ms(response.duration)
            turn.vad_disagreement_signed_ms = O.crosscheck_signed_ms(
                response.start, far, rate)
            turn.vad_disagreement_ms = (
                None if turn.vad_disagreement_signed_ms is None
                else abs(turn.vad_disagreement_signed_ms))
            content = far_analysis.content_after(t1_abs, CONTENT_MIN_DURATION_MS)
            if content is not None and (next_start is None or content.start < next_start):
                turn.ttfab_content_ms = ms(content.start - t1_abs)

            # Did the vendor speak more than once before our next turn? That is
            # either a two-part reply or an idle re-prompt. The interval we
            # measured is still correct (it ends at the FIRST onset), but the
            # conversation carries an extra vendor utterance, so later turns are
            # answering a slightly different history. Flagged, not discarded.
            if next_start is not None:
                between = far_analysis.utterance_groups(
                    GREETING_MERGE_GAP_MS, before=next_start
                )
                extra = [g for g in between if g.start > response.start]
                if extra:
                    turn.flags.append(f"vendor_spoke_again_before_next_turn={len(extra)}")

            # Did WE talk over the reply? Live VAD fired during a pause inside it,
            # so the next turn interrupts rather than follows. This is the
            # multi-turn analogue of `barged_greeting`, and the main risk the turn
            # loop introduces: replies are longer and pause more than greetings.
            reply_end = response.start + response.duration
            if next_start is not None and next_start < reply_end - _barge_margin(rate):
                next_turn = next(t for t, loc in estimates[position + 1:]
                                 if loc is not None)
                next_turn.flags.append(f"overlapped_reply_of_turn_{turn.index}")
                if next_turn.discard_reason is None:
                    next_turn.discard_reason = "barged_reply"

        _apply_turn_discard_rules(turn, estimate)

    r.turns = [turn for turn, _ in estimates]

    # --- mirror turn 1 onto the top-level fields ---------------------------- #
    # Everything downstream (reports, the HTML page, the sweep's regression) reads
    # these, and single-turn runs must stay byte-comparable with earlier data.
    first = r.turns[0]
    r.stimulus_start_ms = first.stimulus_start_ms
    r.t1_ms = first.t1_ms
    r.t2_ms = first.t2_ms
    r.ttfab_onset_ms = first.ttfab_onset_ms
    r.ttfab_content_ms = first.ttfab_content_ms
    r.vendor_response_duration_ms = first.vendor_response_duration_ms
    r.psr = first.psr
    r.psr_global = first.psr_global
    r.drift_ms = first.drift_ms
    r.vad_disagreement_ms = first.vad_disagreement_ms
    # Turn 1's diagnostic flags also surface at call level, where the old
    # single-turn consumers (report, HTML page) look for them.
    for flag in first.flags:
        if flag not in r.flags:
            r.flags.append(flag)

    _apply_discard_rules(r, far_analysis, pre_stimulus)
    return r


def _apply_turn_discard_rules(turn: TurnResult, estimate) -> None:
    """Per-turn rules. Same vocabulary as the call-level rules.

    Ordered most-structural first: a turn whose stimulus could not be located has
    no meaningful t1, so asking whether its reply arrived on time is moot.
    """
    if turn.discard_reason is not None:
        return

    if not estimate.confident:
        turn.flags.append(f"psr={estimate.psr:.2f}<{MIN_PSR}")
        turn.discard_reason = "unlocatable"
        return

    if abs(estimate.drift_ms) > MAX_DRIFT_MS:
        turn.discard_reason = "drift"
        return

    if turn.t2_ms is None:
        turn.discard_reason = "no_response"
        return

    if turn.ttfab_onset_ms is not None and turn.ttfab_onset_ms > RESPONSE_TIMEOUT_MS:
        turn.discard_reason = "no_response"
        return

    if _onset_crosscheck_fails(turn):
        turn.discard_reason = "vad_disagree"
        return


# --------------------------------------------------------------------------- #
# Scripted-dialog mode: no reference, t1 from our own speech offsets
# --------------------------------------------------------------------------- #


def measure_dialog(near: np.ndarray, far: np.ndarray, expected_turns: int,
                   rate: int, *, base: Result | None = None) -> Result:
    """Measure a call where OUR side was a live dialog, not a known waveform.

    The reference-mode trick -- matched-filter each turn's committed clip
    against the near channel -- is unavailable here: Plivo's TTS renders our
    lines afresh on every call, so there is no known waveform to correlate.
    Instead our own speech is segmented the same way the vendor's always was,
    and t1 becomes the refined END of each of our utterances.

    Two things follow, and both are stated in the report rather than papered
    over. t1 is no longer sample-exact, so `psr`, `psr_global` and `drift_ms`
    -- all correlation diagnostics -- are permanently None on this path. And
    turn identity is now structural: turn i is our i-th utterance, so the count
    has to be right for the pairing to mean anything, which is what the channel
    and turn-count checks below defend.
    """
    r = base or Result()
    r.source = "recording"
    r.rate = rate
    r.analyzer_version = ANALYZER_VERSION

    ms = lambda n: samples_to_ms(n, rate)  # noqa: E731

    if near.size == 0 or far.size == 0 or expected_turns <= 0:
        r.discard_reason = "audio_missing"
        return r

    near_analysis = O.analyze(near, rate, refine_offsets=True)
    far_analysis = O.analyze(far, rate, refine_offsets=True)
    r.noise_floor_dbfs = (
        None if not np.isfinite(far_analysis.noise_floor_dbfs)
        else float(far_analysis.noise_floor_dbfs)
    )

    if _starts_mid_speech(far, rate):
        r.flags.append("recording_started_mid_speech")

    our_turns = near_analysis.utterance_groups(TURN_MERGE_GAP_MS)
    r.flags.append(f"near_utterances={len(our_turns)}")

    # Nobody said anything, on either side. Distinct from turn_count_mismatch,
    # which means we could not pair what WAS said: this is a call that connected
    # and carried no speech at all, and reporting it as a pairing failure sends
    # the reader looking for a bug in the analyzer instead of at the call.
    if not our_turns and not far_analysis.segments:
        r.flags.append(f"duration_s={samples_to_ms(len(near), rate) / 1000:.1f}")
        r.discard_reason = "dead_air"
        return r

    # Is the channel map right? Plivo does not document which stereo channel
    # carries which party, and an inversion is invisible in the numbers: it
    # would measure the vendor's speech-end against our own reply and produce
    # plausible nonsense. We know exactly how many times WE spoke, so count.
    if len(our_turns) != expected_turns:
        swapped = len(far_analysis.utterance_groups(TURN_MERGE_GAP_MS))
        if swapped == expected_turns:
            r.flags.append(f"other_channel_has_{swapped}_utterances")
            r.discard_reason = "channel_map_suspect"
        else:
            r.flags.append(f"expected_turns={expected_turns}")
            r.discard_reason = "turn_count_mismatch"
        return r

    far_analysis, bleed = _without_bleed(near, far, our_turns, far_analysis, rate)
    if bleed:
        r.flags.append(f"far_channel_bleed={bleed}")

    # --- greeting, before our first word ------------------------------------- #
    first_start = our_turns[0].start
    pre_stimulus = far_analysis.utterance_groups(GREETING_MERGE_GAP_MS,
                                                 before=first_start)
    if pre_stimulus:
        greeting = pre_stimulus[0]
        r.greeting_onset_ms = ms(greeting.start)
        r.greeting_end_ms = ms(greeting.end)
        r.ttfg_ms = (None if "recording_started_mid_speech" in r.flags
                     else ms(greeting.start))
    if len(pre_stimulus) > 1:
        r.flags.append(f"pre_stimulus_utterances={len(pre_stimulus)}")
        # The measured pauses, so a reader can tell a split greeting from an idle
        # re-prompt without re-opening the WAV. Descriptive only -- nothing
        # compares against it, because a threshold here would be fitted to
        # whichever vendor was measured last.
        pauses = [ms(b.start - a.end)
                  for a, b in zip(pre_stimulus, pre_stimulus[1:])]
        r.flags.append("greeting_pause_ms="
                       + ",".join(f"{p:.0f}" for p in pauses))

    # --- one measurement per thing we said ----------------------------------- #
    turns: list[TurnResult] = []
    for index, ours in enumerate(our_turns, start=1):
        turn = TurnResult(index=index)
        turn.stimulus_start_ms = ms(ours.start)
        turn.t1_ms = ms(ours.end)
        turns.append(turn)

        next_start = (our_turns[index].start if index < len(our_turns) else None)

        # Was the vendor still talking as we finished? Then "the end of the
        # caller's speech" is not a clean anchor -- we spoke over each other and
        # whatever comes next answers an interruption. Distinct from
        # barged_reply, which is about the NEXT turn overlapping THIS reply.
        overlap = far_analysis.first_between(
            max(0, ours.end - ms_window(DOUBLE_TALK_WINDOW_MS, rate)), ours.end)
        double_talk = overlap is not None
        if double_talk:
            turn.flags.append("vendor_speaking_at_our_speech_end")

        response = far_analysis.first_after(ours.start)
        if response is not None and next_start is not None and response.start >= next_start:
            response = None

        if response is not None:
            turn.t2_ms = ms(response.start)
            turn.ttfab_onset_ms = ms(response.start - ours.end)
            turn.vendor_response_duration_ms = ms(response.duration)
            turn.vad_disagreement_signed_ms = O.crosscheck_signed_ms(
                response.start, far, rate)
            turn.vad_disagreement_ms = (
                None if turn.vad_disagreement_signed_ms is None
                else abs(turn.vad_disagreement_signed_ms))
            content = far_analysis.content_after(ours.end, CONTENT_MIN_DURATION_MS)
            if content is not None and (next_start is None or content.start < next_start):
                turn.ttfab_content_ms = ms(content.start - ours.end)

            if next_start is not None:
                between = far_analysis.utterance_groups(GREETING_MERGE_GAP_MS,
                                                        before=next_start)
                extra = [g for g in between if g.start > response.start]
                if extra:
                    turn.flags.append(
                        f"vendor_spoke_again_before_next_turn={len(extra)}")

            # Did we talk over the reply? Plivo's endpointing decided the vendor
            # had finished when it had not, so the next turn interrupts.
            reply_end = response.start + response.duration
            if next_start is not None and next_start < reply_end - _barge_margin(rate):
                turn.flags.append(f"our_turn_{index + 1}_overlapped_this_reply")
                _mark_next_barged(turns, index)

        # Second opinion on t1 itself. The onset cross-check could never guard
        # t1 while it came from correlation; here it is the same class of
        # estimate as t2 and gets the same scrutiny.
        offset_disagreement = O.crosscheck_offset_disagreement_ms(ours.end, near, rate)
        if offset_disagreement is not None:
            turn.flags.append(f"t1_crosscheck={offset_disagreement:.1f}ms")

        _apply_dialog_turn_discard_rules(turn, double_talk=double_talk,
                                         offset_disagreement_ms=offset_disagreement)

    r.turns = turns
    _mirror_first_turn(r)
    _apply_discard_rules(
        r, far_analysis, pre_stimulus, dialog=True,
        greeting_span_end_ms=(ms(pre_stimulus[-1].end) if pre_stimulus else None))
    return r


def ms_window(ms_value: float, rate: int) -> int:
    return int(round(ms_value * rate / 1000.0))


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    x = audio.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _without_bleed(near: np.ndarray, far: np.ndarray,
                   our_turns: list[O.Segment], far_analysis: O.OnsetAnalysis,
                   rate: int) -> tuple[O.OnsetAnalysis, int]:
    """Drop far-channel segments that are just an echo of our own speech.

    Judged by level against the near channel over the same samples, because
    that is what separates leakage from the vendor genuinely talking over us:
    an echo is heavily attenuated, real double-talk is not. Returns a filtered
    analysis plus how many segments were dropped, so the call can say so.
    """
    ratio = 10.0 ** (-BLEED_MIN_ATTENUATION_DB / 20.0)
    kept: list[O.Segment] = []
    dropped = 0
    for segment in far_analysis.segments:
        overlaps_us = any(segment.start < t.end and t.start < segment.end
                          for t in our_turns)
        if overlaps_us:
            near_level = _rms(near[segment.start:segment.end])
            far_level = _rms(far[segment.start:segment.end])
            if near_level > 0 and far_level < near_level * ratio:
                dropped += 1
                continue
        kept.append(segment)
    if not dropped:
        return far_analysis, 0
    return (
        O.OnsetAnalysis(segments=kept,
                        coarse_starts=list(far_analysis.coarse_starts),
                        noise_floor_dbfs=far_analysis.noise_floor_dbfs,
                        rate=far_analysis.rate),
        dropped,
    )


def _mark_next_barged(turns: list[TurnResult], index: int) -> None:
    """The overlap is recorded against the turn that did the interrupting."""
    if index < len(turns):
        nxt = turns[index]
        nxt.flags.append(f"overlapped_reply_of_turn_{index}")
        if nxt.discard_reason is None:
            nxt.discard_reason = "barged_reply"


def _apply_dialog_turn_discard_rules(turn: TurnResult, *, double_talk: bool,
                                     offset_disagreement_ms: float | None) -> None:
    """Per-turn rules for the dialog path, most structural first.

    `unlocatable` and `drift` cannot fire here -- both are correlation verdicts
    about a reference that does not exist. They are replaced by the two ways a
    VAD-derived t1 can be wrong: someone else was talking at it (double_talk),
    or the two detectors disagree about where our speech ended
    (offset_disagree).
    """
    if turn.discard_reason is not None:
        return

    if double_talk:
        turn.discard_reason = "double_talk"
        return

    if (offset_disagreement_ms is not None
            and offset_disagreement_ms > MAX_OFFSET_DISAGREEMENT_MS):
        turn.discard_reason = "offset_disagree"
        return

    if turn.t2_ms is None:
        turn.discard_reason = "no_response"
        return

    if turn.ttfab_onset_ms is not None and turn.ttfab_onset_ms > RESPONSE_TIMEOUT_MS:
        turn.discard_reason = "no_response"
        return

    if _onset_crosscheck_fails(turn):
        turn.discard_reason = "vad_disagree"
        return


def _onset_crosscheck_fails(turn: TurnResult) -> bool:
    """Did the second detector CONTRADICT this onset?

    Three outcomes, and only the middle one is a failure:
      agrees        within MAX_VAD_DISAGREEMENT_MS -- nothing to say
      contradicts   between that and VAD_NO_OPINION_MS -- the detectors saw the
                    same utterance and disagree about where it starts, which is
                    exactly what this check exists to catch
      no opinion    beyond VAD_NO_OPINION_MS -- webrtcvad never found the
                    utterance, so the distance is to some unrelated event

    Both non-agreeing cases are recorded on the turn; only a contradiction
    discards it. Treating silence from the referee as a verdict against the
    measurement is what cost 7 good turns on bench-vapi-20260730-143407.
    """
    signed = turn.vad_disagreement_signed_ms
    disagreement = turn.vad_disagreement_ms
    if disagreement is None or disagreement <= MAX_VAD_DISAGREEMENT_MS:
        return False
    if disagreement > VAD_NO_OPINION_MS:
        turn.flags.append(
            f"vad_crosscheck_unavailable={disagreement:.0f}ms")
        return False

    # Ours LATER: webrtcvad saw something before the speech we found. Tolerated
    # up to MAX_VAD_ONSET_LATE_MS -- that is the conservative direction, not the
    # one that flatters a vendor.
    if signed is not None and signed > 0:
        if signed <= MAX_VAD_ONSET_LATE_MS:
            turn.flags.append(f"vad_onset_conservative=+{signed:.0f}ms")
            return False
        turn.flags.append(f"vad_onset_late=+{signed:.0f}ms")
        return True

    turn.flags.append(f"vad_disagreement={disagreement:.1f}ms")
    return True


def _mirror_first_turn(r: Result) -> None:
    """Top-level fields mirror turn 1 -- every existing consumer reads these."""
    if not r.turns:
        return
    first = r.turns[0]
    r.stimulus_start_ms = first.stimulus_start_ms
    r.t1_ms = first.t1_ms
    r.t2_ms = first.t2_ms
    r.ttfab_onset_ms = first.ttfab_onset_ms
    r.ttfab_content_ms = first.ttfab_content_ms
    r.vendor_response_duration_ms = first.vendor_response_duration_ms
    r.psr = first.psr
    r.psr_global = first.psr_global
    r.drift_ms = first.drift_ms
    r.vad_disagreement_ms = first.vad_disagreement_ms
    for flag in first.flags:
        if flag not in r.flags:
            r.flags.append(flag)


def _apply_discard_rules(r: Result, far: O.OnsetAnalysis,
                         pre_stimulus: list[O.Segment], *,
                         dialog: bool = False,
                         greeting_span_end_ms: float | None = None) -> None:
    """Apply the frozen discard rules, most structural first.

    Order matters: a call with no locatable stimulus has no meaningful t1, so
    there is no point asking whether its response arrived on time.

    Which rules live here vs on the turn:

      * genuinely call-level -- no greeting at all, we talked over the greeting.
        These invalidate the whole conversation, so every turn in it is suspect.
      * turn-level (unlocatable / drift / no_response / vad_disagree) are recorded
        on the turn by _apply_turn_discard_rules. On a MULTI-turn call the call
        survives them, so one bad turn does not waste the others.
      * on a SINGLE-turn call they are mirrored up to the call, in the original
        rule order, so results recorded before multi-turn existed re-analyze
        identically.

    `dialog` and `greeting_span_end_ms` are supplied only by measure_dialog.
    Reference mode calls this with neither, and gets 2.3.0 behaviour exactly --
    which is what keeps archived reference-mode runs re-derivable and Gate A's
    `idle_filler` fixture meaningful. See the 2.4.0 note at the top of the file
    for why the two paths are allowed to differ here.
    """
    single_turn = len(r.turns) == 1
    turn = r.turns[0] if r.turns else None

    if not far.segments:
        r.discard_reason = "no_greeting"
        return

    if r.greeting_onset_ms is not None and r.greeting_onset_ms > GREETING_TIMEOUT_MS:
        r.discard_reason = "no_greeting"
        return

    if single_turn and turn and turn.discard_reason in _STRUCTURAL_TURN_REASONS:
        r.discard_reason = turn.discard_reason
        return

    # Did we talk over the greeting? Live VAD fired early, so whatever came back
    # is a reaction to an interruption rather than to a complete turn.
    #
    # Measured against the END of the pre-stimulus region when the caller supplies
    # it, not against `greeting_end_ms` (the first utterance's end). A vendor that
    # pauses mid-greeting for longer than the merge gap has a greeting_end_ms that
    # stops at phrase one, which understates the extent we could have talked over
    # -- so the vendor most likely to be barged is the one this rule would have
    # protected least. Conservative in the safe direction: it can only make the
    # rule fire more. Verified on bench-*-20260731-152959: the tightest margin
    # across 489 calls is 2192 ms against a 50 ms threshold, so no real call
    # changes verdict.
    barge_extent_ms = (greeting_span_end_ms if greeting_span_end_ms is not None
                       else r.greeting_end_ms)
    if (
        barge_extent_ms is not None
        and r.stimulus_start_ms is not None
        and r.stimulus_start_ms < barge_extent_ms - BARGE_GREETING_MARGIN_MS
    ):
        r.discard_reason = "barged_greeting"
        return

    # The "are you still there?" prompt. Without this rule it silently becomes the
    # measured response and produces a fast, wrong number.
    #
    # Detected as a second utterance group before our stimulus, rather than as any
    # speech inside a time window -- a greeting normally arrives as several
    # segments, so a window test flags every two-part greeting as an idle prompt.
    #
    # Not applied when the tape opened mid-greeting: the count is then a count of
    # FRAGMENTS, not of utterances. A vendor that greets the instant it answers
    # beats the recorder to the first syllable, so its greeting arrives already
    # split and every such call reads as an idle prompt. Measured 2026-07-30 on
    # bench-vapi-20260730-141015: a flawless four-turn call -- every question
    # answered correctly -- was discarded whole while three of its four turns had
    # clean measurements. TTFG is already withheld on these recordings for the
    # same reason (the greeting's start is not on the tape), so nothing that
    # depends on the greeting's extent is reported either way; TTFAB does not,
    # and is kept.
    if len(pre_stimulus) > 1:
        if dialog:
            # Dialog mode cannot be corrupted by this, so it is recorded rather
            # than judged. The reply search for turn i is
            # first_after(our turn i START) -- see measure_dialog -- so every
            # candidate reply lies after our first word, and the whole
            # pre-stimulus region is outside every search window. Discarding the
            # call therefore protected no measurement while throwing away the
            # turns that WERE measured.
            #
            # It stays a flag rather than becoming nothing, because "the vendor
            # spoke before we did" is real information: it is either a split
            # greeting or a genuine 'are you still there?', and the count plus
            # greeting_pause_ms let a reader tell which without the audio.
            r.flags.append(f"vendor_spoke_before_first_turn={len(pre_stimulus) - 1}")
        elif "recording_started_mid_speech" not in r.flags:
            r.discard_reason = "idle_filler"
            return
        else:
            r.flags.append("idle_filler_unassessable_clipped_greeting")

    if single_turn and turn and turn.discard_reason:
        r.discard_reason = turn.discard_reason
        return


# --------------------------------------------------------------------------- #
# Loading from a run directory
# --------------------------------------------------------------------------- #

#: Turn-level failures that, on a SINGLE-turn call, are the call's failure.
#  Split at the same point the original rule order did, so single-turn verdicts
#  are byte-identical to pre-multi-turn behaviour.
_STRUCTURAL_TURN_REASONS = ("unlocatable", "drift", "out_of_order")


RECORDING_WAV = "recording.wav"
REFERENCE_WAV = "our_audio.wav"
METADATA_JSON = "metadata.json"
RESULT_JSON = "result.json"

#: metadata.json `mode` that selects measure_dialog. Anything else (including
#: absent) is reference mode.
DIALOG_MODE = "scripted_dialog"


def _base_from_metadata(meta: dict) -> Result:
    return Result(
        call_id=meta.get("call_id", ""),
        run_id=meta.get("run_id", ""),
        vendor=meta.get("vendor", ""),
        kind=meta.get("kind", ""),
        stimulus_id=meta.get("stimulus_id"),
        known_delay_ms=meta.get("known_delay_ms"),
    )


def measure_call(call_dir: Path) -> Result:
    """Measure one call directory. Never raises on missing audio -- it discards."""
    call_dir = Path(call_dir)
    meta_path = call_dir / METADATA_JSON
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    base = _base_from_metadata(meta)

    # Only the recording is required up front. Which reference files are needed
    # depends on the turn script, so that is checked where they are resolved below
    # -- requiring `our_audio.wav` unconditionally would reject a perfectly good
    # multi-turn call directory that only has per-turn references.
    rec = call_dir / RECORDING_WAV
    if not rec.exists():
        base.discard_reason = "audio_missing"
        base.analyzer_version = ANALYZER_VERSION
        return base

    audio, rate = sf.read(rec, dtype="int16", always_2d=True)

    channels = meta.get("channel_map") or {"near": 0, "far": 1}
    if audio.shape[1] < 2:
        base.discard_reason = "audio_missing"
        base.flags.append("recording_not_stereo")
        base.analyzer_version = ANALYZER_VERSION
        return base

    near = np.ascontiguousarray(audio[:, channels.get("near", 0)])
    far = np.ascontiguousarray(audio[:, channels.get("far", 1)])

    # Which measurement path? The mode is stamped by whatever placed the call.
    # Absent means a run recorded before modes existed, i.e. reference mode --
    # so archived directories keep re-analyzing exactly as they did.
    if meta.get("mode") == DIALOG_MODE:
        result = measure_dialog(near, far, _expected_turns(meta), rate, base=base)
        _attach_transcript_flags(result, meta)
        return result

    # Reference mode: metadata lists one reference per turn. Single-turn runs
    # (and every run recorded before turns existed) fall back to our_audio.wav.
    from .resample import resample_int16

    reference_files = [t.get("reference_wav") for t in (meta.get("turns") or [])
                       if t.get("reference_wav")]
    if not reference_files:
        reference_files = [REFERENCE_WAV]

    references: list[np.ndarray] = []
    for name in reference_files:
        path = call_dir / name
        if not path.exists():
            base.discard_reason = "audio_missing"
            base.flags.append(f"missing_reference:{name}")
            base.analyzer_version = ANALYZER_VERSION
            return base
        data, data_rate = sf.read(path, dtype="int16")
        if data_rate != rate:
            data = resample_int16(data, data_rate, rate)
        references.append(data)

    return measure_turns(near, far, references, rate, base=base)


def _expected_turns(meta: dict) -> int:
    """How many times we actually spoke. The pairing depends on getting this
    right, so prefer what the dialog recorded over what it intended."""
    turns = meta.get("turns") or []
    spoken = [t for t in turns if t.get("spoken")]
    if spoken:
        return len(spoken)
    played = meta.get("turns_played")
    return int(played) if played else len(turns)


def _attach_transcript_flags(result: Result, meta: dict) -> None:
    """Carry the answer check onto the turn it belongs to.

    Accuracy never touches a latency figure -- a wrong answer is a different
    failure from a slow one, and merging them would hide both. It rides along
    as a flag so the report can show them side by side.
    """
    by_index = {t.get("index"): t for t in (meta.get("turns") or [])}
    for turn in result.turns:
        row = by_index.get(turn.index)
        if not row:
            continue
        if row.get("case_id"):
            turn.flags.append(f"case={row['case_id']}")
        verified = row.get("answer_verified")
        if verified is True:
            turn.flags.append("answer_verified")
        elif verified is False:
            turn.flags.append("answer_not_verified")


def analyze_run(run_dir: Path, write: bool = True) -> list[Result]:
    """Measure every call in a run directory, in a stable order.

    Sorted by call id so repeated runs produce byte-identical output -- exit
    criterion 4 is a comparison of files, and directory iteration order is not
    guaranteed.
    """
    run_dir = Path(run_dir)
    results: list[Result] = []
    for call_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        result = measure_call(call_dir)
        if write:
            (call_dir / RESULT_JSON).write_text(
                json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
            )
        results.append(result)
    return results
