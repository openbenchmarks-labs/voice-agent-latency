"""Locate our own speech end (t1) by matched filtering.

REFERENCE MODE ONLY. This path applies when our side of the call is committed
audio we know sample-for-sample, so t1 can be found by matched filtering rather
than trusted from when we wrote it to the socket -- write time is not transmit
time, and the carrier exposes no acknowledgement that would tell us the
difference.

Every published number on this board comes from the OTHER path. In
scripted-dialog mode the caller is the carrier's own TTS reading a script, so
there is no known waveform to match and t1 is the refined end of our speech on
the near channel (analyzer/onset.py `_refine_offset`). This module is retained
because reference mode is how the analyzer is validated against a known answer,
which is what Gate A does.

Three specifics, each of which is worth roughly tens of milliseconds:

1. Match the LAST 500 ms of reference speech, not the whole clip. Matching the
   whole thing and adding its length assumes the audio arrived as one rigid
   block; a 40 ms gap mid-stimulus would push t1 out by 40 ms. Matching the tail
   locates the end directly.

2. Use GCC-PHAT rather than plain cross-correlation. Speech is quasi-periodic, so
   plain correlation produces sidelobes at the pitch period (5-10 ms) -- exactly
   the precision band we care about. Phase-transform whitening flattens the
   spectrum and sharpens the peak.

3. Compare the tail result against a whole-reference match. Their difference is a
   direct measurement of stretch or loss inside the stimulus, and it is the
   `drift` discard rule.

Confidence comes out as peak-to-sidelobe ratio. Below threshold the stimulus is
not locatable and the call is discarded rather than reported with a bad t1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len, irfft, rfft

# Partial phase whitening. beta=0 is plain cross-correlation, 1.0 is pure
# GCC-PHAT. 0.75 is chosen from measurement on the fixtures, not by convention:
#
#   beta   t1 error   tail PSR   global PSR   40 ms drift detected?
#   0.00   exact      3.03       2.45         no  -- reports 0 ms, silently wrong
#   0.50   exact      6.48       10.2         yes
#   0.75   exact      7.51       35.3         yes
#   1.00   exact      6.39       21530        yes
#
# Two things decide it. Plain correlation cannot see the drift at all, which
# disables the `drift` discard rule. And at beta=1 the surface degenerates to a
# delta function against an exact reference, so global PSR runs to five figures
# and stops being a usable confidence number. 0.75 keeps drift detection, gives
# the widest tail-PSR margin, and leaves PSR interpretable.
DEFAULT_BETA = 0.75

# Sidelobes are measured outside this radius of the peak, so the mainlobe is not
# counted against itself. 20 ms is wide enough to exclude the pitch-period
# structure of the mainlobe on adult speech.
SIDELOBE_EXCLUSION_MS = 20.0

# Below this, treat the stimulus as unlocatable (the `unlocatable` discard).
#
# Margin note for Phase C: on fixtures the clean cases sit at ~7.5, but the
# barge-in case measures 3.68 because the vendor's reply overlaps the template
# window and adds uncorrelated energy exactly where we are matching. That is only
# a 20% margin, and real telephony audio will be noisier than a fixture. This
# threshold must be re-validated against the delay-sweep recordings; if genuine
# barge-in calls start being discarded as `unlocatable`, the discard-rate
# breakdown in the report is what will show it.
MIN_PSR = 3.0

TAIL_TEMPLATE_MS = 500.0


@dataclass(frozen=True)
class Match:
    """Where a reference was found inside a signal."""

    lag: int          # index in the signal where the reference begins
    end: int          # one past the reference's last sample -> t1
    peak: float       # correlation peak height
    psr: float        # peak-to-sidelobe ratio
    confident: bool


def _prep(x: np.ndarray) -> np.ndarray:
    """To float64 with DC removed.

    DC offset is common on telephony captures and would put a large spike at zero
    lag that swamps the real peak.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0:
        return y
    return y - y.mean()


def gcc_phat(signal: np.ndarray, reference: np.ndarray, beta: float = DEFAULT_BETA,
             eps: float = 1e-12) -> np.ndarray:
    """Cross-correlate `reference` against `signal` with phase whitening.

    Returns the correlation surface indexed by lag: element `k` is the score for
    the reference starting at sample `k` of the signal.
    """
    sig = _prep(signal)
    ref = _prep(reference)
    if ref.size == 0 or sig.size == 0:
        return np.zeros(0)

    nfft = next_fast_len(sig.size + ref.size)
    SIG = rfft(sig, nfft)
    REF = rfft(ref, nfft)

    cross = SIG * np.conj(REF)
    if beta > 0.0:
        # beta=1 divides out magnitude entirely, leaving phase. Partial values
        # trade peak sharpness against noise robustness.
        cross /= np.power(np.abs(cross), beta) + eps

    cc = irfft(cross, nfft)
    # Only non-negative lags are meaningful: the reference cannot start before
    # the recording does.
    return cc[: max(1, sig.size - ref.size + 1)]


def _psr(cc: np.ndarray, peak_idx: int, rate: int) -> float:
    """Peak height over the largest sidelobe outside the mainlobe."""
    excl = max(1, int(round(SIDELOBE_EXCLUSION_MS * rate / 1000.0)))
    lo, hi = max(0, peak_idx - excl), min(len(cc), peak_idx + excl + 1)

    masked = np.concatenate([cc[:lo], cc[hi:]])
    if masked.size == 0:
        return float("inf")

    sidelobe = np.abs(masked).max()
    if sidelobe <= 0:
        return float("inf")
    return float(abs(cc[peak_idx]) / sidelobe)


def find(signal: np.ndarray, reference: np.ndarray, rate: int,
         beta: float = DEFAULT_BETA, min_psr: float = MIN_PSR) -> Match:
    """Locate `reference` inside `signal`."""
    cc = gcc_phat(signal, reference, beta=beta)
    if cc.size == 0:
        return Match(lag=0, end=0, peak=0.0, psr=0.0, confident=False)

    peak_idx = int(np.argmax(cc))
    psr = _psr(cc, peak_idx, rate)
    return Match(
        lag=peak_idx,
        end=peak_idx + len(reference),
        peak=float(cc[peak_idx]),
        psr=psr,
        confident=psr >= min_psr,
    )


def tail_template(reference: np.ndarray, rate: int,
                  window_ms: float = TAIL_TEMPLATE_MS) -> np.ndarray:
    """The last `window_ms` of the reference.

    The reference must already end at its last speech sample -- trailing digital
    silence would put the template on silence and destroy the peak. Stimulus prep
    guarantees this (see fixtures/generate.py).
    """
    n = int(round(window_ms * rate / 1000.0))
    if n <= 0 or n >= len(reference):
        return np.asarray(reference)
    return np.asarray(reference)[-n:]


@dataclass(frozen=True)
class T1Estimate:
    """t1 from both the tail and the whole reference, plus their disagreement."""

    t1: int                 # from the tail match -- this is the reported value
    t1_global: int          # from the whole-reference match
    drift_samples: int      # t1 - t1_global
    drift_ms: float
    psr: float
    psr_global: float
    stimulus_start: int     # where the stimulus begins, from the global match
    confident: bool


def locate_t1(signal: np.ndarray, reference: np.ndarray, rate: int,
              beta: float = DEFAULT_BETA, min_psr: float = MIN_PSR) -> T1Estimate:
    """Find the end of our stimulus in `signal`, both ways.

    `t1` is the tail estimate and is what gets reported. `drift_ms` is the
    diagnostic: large values mean the stimulus was stretched or lost samples
    mid-flight, so neither estimate should be trusted.
    """
    template = tail_template(reference, rate)

    tail = find(signal, template, rate, beta=beta, min_psr=min_psr)
    whole = find(signal, reference, rate, beta=beta, min_psr=min_psr)

    drift = tail.end - whole.end
    return T1Estimate(
        t1=tail.end,
        t1_global=whole.end,
        drift_samples=drift,
        drift_ms=drift * 1000.0 / rate,
        psr=tail.psr,
        psr_global=whole.psr,
        stimulus_start=whole.lag,
        confident=tail.confident,
    )
