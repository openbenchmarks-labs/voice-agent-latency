"""Find when the vendor starts speaking (t2), in two stages.

Neither stage works alone.

An energy threshold is precise but trivially fooled: a vendor that streams
continuous comfort noise instead of silence trips it on the first frame and
appears roughly 200 ms faster than it is. That is the single easiest way for a
latency benchmark to be wrong in a vendor's favour.

Silero is immune to that -- it classifies speech, not loudness -- but it decides
in 32 ms windows, which is coarse next to the differences we are trying to
resolve.

So: Silero decides *whether* a region is speech, then a fine pass decides
*exactly where* it starts, searching only inside the window Silero already
accepted. webrtcvad runs alongside as an independent cross-check; disagreement is
flagged rather than averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .codec import int16_to_float
from .resample import ms_to_samples, resample_int16, samples_to_ms

MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"

# Silero v5 takes a fixed window -- 512 samples at 16 kHz, 256 at 8 kHz, both
# 32 ms -- AND a mandatory context prefix of window/8 samples carried over from
# the previous chunk. The onnx graph accepts any input length, so omitting the
# context does not error; it silently produces garbage:
#
#   rate    window  context   max prob   frac speech
#   16000   512     0         0.004      0.000       <- fails outright
#   16000   512     64        1.000      0.883
#    8000   256     0         0.980      0.270       <- silently degraded
#    8000   256     32        1.000      0.883
#
# Measured on analyzer/fixtures/sources/greeting.wav. With the context both rates
# agree exactly, which is what confirms the contract. The 8 kHz no-context case is
# the dangerous one: it yields plausible-looking segments that are badly wrong, and
# every t2 would have inherited the error.
SILERO_WINDOWS = {8000: 256, 16000: 512}
SILERO_CONTEXT = {8000: 32, 16000: 64}

SPEECH_THRESHOLD = 0.5

# Enter speech only after this many consecutive positive windows, and attribute
# the onset to the FIRST of them -- not the one that tripped the counter, which
# would bias every onset late by one window.
ONSET_HYSTERESIS_WINDOWS = 2

# A gap shorter than this does not end a segment. Set above a typical
# stop-consonant closure so "Saturday" is one segment, not three.
MIN_SILENCE_MS = 200.0

# Segments shorter than this are dropped as spurious.
MIN_SPEECH_MS = 60.0

# Stage two searches this far either side of the coarse onset.
#
# Sized at four 32 ms windows rather than two. Silero is stateful, so identical
# response audio can be classified a couple of windows later depending on what
# preceded it -- the idle-filler fixture measures a coarse estimate 63 ms late for
# exactly that reason. A radius of 64 ms only just reaches the true onset in that
# case, and would miss it entirely if the coarse estimate slipped one more window.
#
# A wide radius is only safe because the search is clamped to the end of the
# previous speech segment (see `earliest` below), so it can never reach back and
# lock onto earlier speech.
REFINE_RADIUS_MS = 128.0

# Fine-pass framing. 2 ms hop is well below the precision we claim, so the
# refinement is not itself the limiting factor.
FINE_HOP_MS = 2.0
FINE_WINDOW_MS = 8.0

# How far above the measured noise floor a frame must sit to count as the onset.
# 10 dB is enough to clear comfort noise while still catching a breathy attack.
FINE_MARGIN_DB = 10.0

# The lowest noise floor we will believe. A channel whose head is exact digital
# silence measures -inf dBFS, which would make `floor + margin` accept every frame
# and collapse the refinement to "start of search window" -- an onset roughly one
# search radius early, on every call. Real telephony never delivers true silence,
# but fixtures do, and mu-law's smallest step is around -72 dBFS anyway, so
# anything below this is quantisation rather than signal.
FLOOR_FLOOR_DBFS = -75.0

# The onset must stay above threshold this long, so a single noisy frame does not
# become the answer.
FINE_MIN_DURATION_MS = 10.0

# Fallback window for signals too short to frame (see noise_floor_dbfs).
NOISE_WINDOW_MS = 300.0

# The noise floor is the Nth percentile of frame energy across the channel.
# Low enough to land in genuine background on a recording that is mostly silence
# between turns; high enough not to be dragged to the single quietest frame.
NOISE_PERCENTILE = 10.0

# Above this, stage one and the cross-check are considered to disagree.
DISAGREEMENT_FLAG_MS = 40.0


@dataclass(frozen=True)
class Segment:
    """A run of speech, in samples. `start` is inclusive, `end` exclusive."""

    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class OnsetAnalysis:
    segments: list[Segment] = field(default_factory=list)
    coarse_starts: list[int] = field(default_factory=list)
    noise_floor_dbfs: float = -np.inf
    rate: int = 8000

    def first_after(self, sample: int) -> Segment | None:
        """First speech segment starting at or after `sample`."""
        for seg in self.segments:
            if seg.start >= sample:
                return seg
        return None

    def first_between(self, lo: int, hi: int) -> Segment | None:
        """First speech segment starting inside [lo, hi)."""
        for seg in self.segments:
            if lo <= seg.start < hi:
                return seg
        return None

    def utterance_groups(self, merge_gap_ms: float,
                         before: int | None = None) -> list[Segment]:
        """Merge segments separated by less than `merge_gap_ms` into utterances.

        Needed because a greeting is usually more than one segment. "Hi, thanks for
        calling. How can I help you today?" has intra-sentence pauses of roughly
        180 ms, which is above MIN_SILENCE_MS, so it arrives as two or three
        segments -- more of them under background noise, which lengthens the
        pauses that clear the threshold.

        Treating `segments[0]` as the whole greeting would put the greeting's own
        later fragments inside the idle-prompt window and discard perfectly good
        calls. Any vendor with a two-part greeting would fail.

        A single silence threshold cannot separate both cases: intra-greeting
        pauses (~180 ms) sit too close to a filler-to-answer gap (~300 ms). So
        segments stay fine-grained for the filler/content distinction, and grouping
        with a wider gap is applied only where utterance extent is what matters.
        """
        segs = [s for s in self.segments if before is None or s.start < before]
        if not segs:
            return []

        gap = ms_to_samples(merge_gap_ms, self.rate)
        groups: list[list[Segment]] = [[segs[0]]]
        for seg in segs[1:]:
            if seg.start - groups[-1][-1].end < gap:
                groups[-1].append(seg)
            else:
                groups.append([seg])
        return [Segment(start=g[0].start, end=g[-1].end) for g in groups]

    def content_after(self, sample: int, min_duration_ms: float) -> Segment | None:
        """First *substantive* segment after `sample`.

        Substantive means long enough to be an answer rather than a filler token.
        A short segment followed by a gap is treated as filler and skipped, which
        is what separates TTFAB-content from TTFAB-onset. We do not filter filler
        out of the onset number and we do not penalise it -- filler-first genuinely
        improves how responsive an agent feels. We just report both.
        """
        need = ms_to_samples(min_duration_ms, self.rate)
        for seg in self.segments:
            if seg.start >= sample and seg.duration >= need:
                return seg
        return None


class _Silero:
    """Lazily-loaded onnx session. Loading costs ~50 ms, so it is cached."""

    _session = None

    @classmethod
    def session(cls):
        if cls._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            # Single-threaded and deterministic: the analyzer must produce
            # identical results on re-runs, and thread
            # scheduling is a cheap way to lose that.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            cls._session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return cls._session

    @classmethod
    def probabilities(cls, audio: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
        """Speech probability per window. Returns (probs, window_size)."""
        if rate not in SILERO_WINDOWS:
            raise ValueError(f"Silero supports {sorted(SILERO_WINDOWS)}, got {rate}")

        window = SILERO_WINDOWS[rate]
        n_ctx = SILERO_CONTEXT[rate]
        x = int16_to_float(audio).astype(np.float32)

        n_windows = len(x) // window
        if n_windows == 0:
            return np.zeros(0, dtype=np.float32), window

        sess = cls.session()
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(rate, dtype=np.int64)

        # Context starts as zeros and is then the tail of the previous window.
        # Mandatory -- see SILERO_WINDOWS above.
        ctx = np.zeros(n_ctx, dtype=np.float32)

        probs = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            chunk = x[i * window : (i + 1) * window]
            inp = np.concatenate([ctx, chunk])[None, :]
            out, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
            probs[i] = out[0, 0]
            ctx = chunk[-n_ctx:]
        return probs, window


def noise_floor_dbfs(audio: np.ndarray, rate: int,
                     window_ms: float = NOISE_WINDOW_MS) -> float:
    """The channel's noise floor in dBFS, as a low percentile of frame energy.

    This is the reference the fine pass measures against, so it has to come from
    a genuinely quiet region.

    It used to be the RMS of the first `window_ms`, which assumed the recording
    opens in silence. That holds for fixtures and fails on real calls: the
    recorder comes up after the vendor has started greeting, so 5/10 calls in the
    first Telnyx bench run opened mid-speech. Estimating the floor from speech
    inflated it by tens of dB and pushed response onsets ~10 ms late.

    A low percentile finds the quiet stretches wherever they happen to be, so a
    clipped head costs nothing. Recordings are mostly silence between turns, so
    the 10th percentile lands in real background.
    """
    if len(audio) == 0:
        return -np.inf

    hop = max(1, ms_to_samples(20.0, rate))
    win = max(2, ms_to_samples(20.0, rate))
    energies = _frame_energies(audio, rate, hop, win)

    if energies.size == 0:
        # Signal shorter than one frame: fall back to a plain RMS of what exists.
        x = int16_to_float(audio[: ms_to_samples(window_ms, rate)])
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        return 20.0 * np.log10(rms) if rms > 0 else -np.inf

    return float(np.percentile(energies, NOISE_PERCENTILE))


def _frame_energies(audio: np.ndarray, rate: int, hop: int, win: int) -> np.ndarray:
    """Per-frame RMS in dBFS."""
    if len(audio) < win:
        return np.zeros(0)
    n = 1 + (len(audio) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = int16_to_float(audio[idx])
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12))


def _refine_onset(audio: np.ndarray, rate: int, coarse: int, floor_db: float,
                  earliest: int = 0) -> int:
    """Locate the exact onset near `coarse`.

    Searches a window around the coarse estimate for the first frame that clears
    the noise floor by a margin and stays clear. Falls back to the coarse estimate
    if nothing qualifies, so a failed refinement degrades to 32 ms resolution
    rather than to nonsense.

    `earliest` bounds how far back the search may reach -- normally the end of the
    previous speech segment. Without it, a wide radius could walk backwards into
    the tail of the preceding utterance and report that as the onset.
    """
    radius = ms_to_samples(REFINE_RADIUS_MS, rate)
    lo = max(0, earliest, coarse - radius)
    hi = min(len(audio), coarse + radius)
    if hi - lo <= 0:
        return coarse

    hop = max(1, ms_to_samples(FINE_HOP_MS, rate))
    win = max(2, ms_to_samples(FINE_WINDOW_MS, rate))

    energies = _frame_energies(audio[lo:hi], rate, hop, win)
    if energies.size == 0:
        return coarse

    threshold = max(floor_db, FLOOR_FLOOR_DBFS) + FINE_MARGIN_DB
    need = max(1, int(round(FINE_MIN_DURATION_MS / FINE_HOP_MS)))

    above = energies > threshold
    # First index where `need` consecutive frames are above threshold.
    if need > 1:
        kernel = np.ones(need, dtype=int)
        runs = np.convolve(above.astype(int), kernel, mode="valid")
        candidates = np.nonzero(runs >= need)[0]
    else:
        candidates = np.nonzero(above)[0]

    if candidates.size == 0:
        return coarse

    # Attribute the onset to the END of the first qualifying window, not its start.
    # The threshold sits far below speech level, so a window trips as soon as
    # speech *touches* it -- meaning the first qualifying window began roughly one
    # window-width before the true onset. Measured across all fixtures:
    #
    #   attribution   mean error   max |error|
    #   window start    -6.79 ms      7.25 ms
    #   window centre   -2.79 ms      3.25 ms
    #   window end      +1.21 ms      3.00 ms
    #
    # Start-attribution alone would put every TTFAB ~7 ms low, and because t1 is
    # sample-exact from correlation the bias would not cancel -- it would land
    # directly in the reported number.
    return lo + int(candidates[0]) * hop + win


def _refine_offset(audio: np.ndarray, rate: int, coarse: int, floor_db: float,
                   earliest: int = 0, latest: int | None = None) -> int:
    """Locate the exact END of speech near `coarse`. The mirror of _refine_onset.

    Only needed since t1 stopped coming from correlation. When our own speech
    was a known waveform, the end of it was sample-exact; in a live dialog it is
    whatever the speech detector says, and Silero resolves only to its 32 ms
    window. That error does not cancel -- it lands directly in every TTFAB --
    so the coarse end is refined the same way onsets are, by walking frame
    energies at a 2 ms hop.

    Searches for the LAST run of frames clearing the noise floor by a margin.
    Attribution mirrors _refine_onset: the run's start index plus one window
    width, which lands on the END of that run's first window. Measured on the
    clean dialog fixture (analyzer/fixtures/dialog.py attribution_table(),
    recomputed on every Gate A run rather than remembered):

        attribution     mean t1 error   max |error|
        run start          -9.9 ms         11.0 ms
        run start + win    -1.9 ms          3.0 ms

    Start-attribution alone puts every t1 ~10 ms early, and since TTFAB is
    t2 - t1 the bias lands directly in the reported number with nothing to
    cancel it.

    `earliest` and `latest` bound the search -- normally this segment's start
    and the next segment's start -- so a wide radius cannot report the previous
    utterance's tail or the next one's head as this segment's end. Falls back to
    the coarse value if nothing qualifies, degrading to 32 ms resolution rather
    than to nonsense.
    """
    radius = ms_to_samples(REFINE_RADIUS_MS, rate)
    hard_latest = len(audio) if latest is None else min(len(audio), latest)
    lo = max(0, earliest, coarse - radius)
    hi = min(hard_latest, coarse + radius)
    if hi - lo <= 0:
        return coarse

    hop = max(1, ms_to_samples(FINE_HOP_MS, rate))
    win = max(2, ms_to_samples(FINE_WINDOW_MS, rate))

    energies = _frame_energies(audio[lo:hi], rate, hop, win)
    if energies.size == 0:
        return coarse

    threshold = max(floor_db, FLOOR_FLOOR_DBFS) + FINE_MARGIN_DB
    need = max(1, int(round(FINE_MIN_DURATION_MS / FINE_HOP_MS)))

    above = energies > threshold
    if need > 1:
        kernel = np.ones(need, dtype=int)
        runs = np.convolve(above.astype(int), kernel, mode="valid")
        candidates = np.nonzero(runs >= need)[0]
    else:
        candidates = np.nonzero(above)[0]

    if candidates.size == 0:
        return coarse

    return lo + int(candidates[-1]) * hop + win


def _segments_from_probs(probs: np.ndarray, window: int, rate: int) -> list[tuple[int, int]]:
    """Coarse speech segments from per-window probabilities, with hysteresis."""
    speech = probs >= SPEECH_THRESHOLD

    min_silence = max(1, int(round(MIN_SILENCE_MS / samples_to_ms(window, rate))))
    on_run = ONSET_HYSTERESIS_WINDOWS

    segments: list[tuple[int, int]] = []
    i = 0
    n = len(speech)
    while i < n:
        if not speech[i]:
            i += 1
            continue

        # Require `on_run` consecutive positives to enter, then attribute the
        # onset to the first of them.
        if on_run > 1 and not speech[i : i + on_run].all():
            i += 1
            continue

        start = i
        i += on_run
        silence = 0
        end = start + on_run
        while i < n:
            if speech[i]:
                silence = 0
                end = i + 1
            else:
                silence += 1
                if silence >= min_silence:
                    break
            i += 1

        segments.append((start * window, end * window))

    return segments


def analyze(audio: np.ndarray, rate: int, *,
            refine_offsets: bool = False) -> OnsetAnalysis:
    """Full two-stage analysis of one channel.

    `refine_offsets` additionally refines each segment's END. Off by default:
    in reference mode only onsets are measured from (t1 comes from correlation),
    and leaving the default alone keeps archived runs re-analysing identically.
    Scripted-dialog mode turns it on for the near channel, where the end of our
    own speech IS t1.
    """
    floor_db = noise_floor_dbfs(audio, rate)
    probs, window = _Silero.probabilities(audio, rate)

    coarse = [(s, e) for s, e in _segments_from_probs(probs, window, rate)
              if e - s >= ms_to_samples(MIN_SPEECH_MS, rate)]

    segments: list[Segment] = []
    coarse_starts: list[int] = []
    prev_end = 0
    for i, (c_start, c_end) in enumerate(coarse):
        refined = _refine_onset(audio, rate, c_start, floor_db, earliest=prev_end)
        # Refinement may only move the onset inside the searched window; it must
        # never cross past the segment's end.
        refined = min(refined, max(c_start, c_end - 1))

        end = c_end
        if refine_offsets:
            next_start = coarse[i + 1][0] if i + 1 < len(coarse) else None
            end = _refine_offset(audio, rate, c_end, floor_db,
                                 earliest=refined + 1, latest=next_start)
            end = max(end, refined + 1)
        segments.append(Segment(start=refined, end=end))
        coarse_starts.append(c_start)
        prev_end = c_end

    return OnsetAnalysis(
        segments=segments,
        coarse_starts=coarse_starts,
        noise_floor_dbfs=floor_db,
        rate=rate,
    )


def webrtcvad_onsets(audio: np.ndarray, rate: int, aggressiveness: int = 3,
                     frame_ms: int = 20) -> list[int]:
    """Independent cross-check.

    webrtcvad only accepts 8/16/32/48 kHz and 10/20/30 ms frames, so anything else
    is resampled to 16 kHz first and the result mapped back.
    """
    import webrtcvad

    work_rate = rate if rate in (8000, 16000, 32000, 48000) else 16000
    work = audio if work_rate == rate else resample_int16(audio, rate, work_rate)

    vad = webrtcvad.Vad(aggressiveness)
    n = ms_to_samples(frame_ms, work_rate)
    if n <= 0 or len(work) < n:
        return []

    flags = []
    for i in range(len(work) // n):
        frame = np.ascontiguousarray(work[i * n : (i + 1) * n], dtype=np.int16)
        flags.append(vad.is_speech(frame.tobytes(), work_rate))

    onsets = []
    prev = False
    for i, f in enumerate(flags):
        if f and not prev:
            onsets.append(int(round(i * n * rate / work_rate)))
        prev = f
    return onsets


def crosscheck_signed_ms(onset: int, audio: np.ndarray, rate: int) -> float | None:
    """Distance from `onset` to the nearest webrtcvad rising edge, SIGNED.

    Positive means OUR onset is later than webrtcvad's; negative means earlier.
    The direction is what makes the number actionable, because the two errors
    are not equally dangerous:

      ours EARLIER   the failure this check exists for -- the energy pass fired
                     on comfort noise and the vendor looks ~200 ms faster than
                     it is
      ours LATER     webrtcvad tripped on a breath, a codec artifact or line
                     noise ahead of the real speech, and we did not. Being
                     conservative is not the same kind of mistake.
    """
    others = webrtcvad_onsets(audio, rate)
    if not others:
        return None
    nearest = min(others, key=lambda o: abs(o - onset))
    return samples_to_ms(onset - nearest, rate)


def crosscheck_disagreement_ms(onset: int, audio: np.ndarray, rate: int) -> float | None:
    """How far `onset` sits from the nearest webrtcvad onset, in ms.

    Returned rather than acted on. Magnitude only -- see crosscheck_signed_ms
    when the direction matters, which for the discard rule it does.
    """
    signed = crosscheck_signed_ms(onset, audio, rate)
    return None if signed is None else abs(signed)


def webrtcvad_offsets(audio: np.ndarray, rate: int, aggressiveness: int = 3,
                      frame_ms: int = 20) -> list[int]:
    """Falling edges, for cross-checking a speech END.

    The onset cross-check only ever looked for rising edges, which was enough
    while t1 came from correlation. In scripted-dialog mode t1 IS a falling
    edge, so it needs a second opinion of its own.
    """
    import webrtcvad

    work_rate = rate if rate in (8000, 16000, 32000, 48000) else 16000
    work = audio if work_rate == rate else resample_int16(audio, rate, work_rate)

    vad = webrtcvad.Vad(aggressiveness)
    n = ms_to_samples(frame_ms, work_rate)
    if n <= 0 or len(work) < n:
        return []

    flags = []
    for i in range(len(work) // n):
        frame = np.ascontiguousarray(work[i * n : (i + 1) * n], dtype=np.int16)
        flags.append(vad.is_speech(frame.tobytes(), work_rate))

    offsets = []
    for i, f in enumerate(flags):
        if not f and i and flags[i - 1]:
            # The offset is the boundary between the last speech frame and this
            # one, i.e. the start of this frame.
            offsets.append(int(round(i * n * rate / work_rate)))
    if flags and flags[-1]:
        offsets.append(int(round(len(flags) * n * rate / work_rate)))
    return offsets


def crosscheck_offset_disagreement_ms(offset: int, audio: np.ndarray,
                                      rate: int) -> float | None:
    """Distance from `offset` to the nearest webrtcvad falling edge, in ms."""
    return _nearest_ms(offset, webrtcvad_offsets(audio, rate), rate)


def _nearest_ms(sample: int, others: list[int], rate: int) -> float | None:
    if not others:
        return None
    nearest = min(others, key=lambda o: abs(o - sample))
    return samples_to_ms(abs(nearest - sample), rate)
