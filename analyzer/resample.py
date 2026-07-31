"""Rational resampling.

Two rates matter in this pipeline:

  16000 Hz  Silero's native rate, and the rate fixture sources are authored at
   8000 Hz  the carrier's stereo recording, which every reported number comes from

Sources are authored at 16 kHz and downsampled to 8 kHz to match the tape.
`resample_poly` is used rather than FFT resampling because it is phase-linear and
introduces a *known, constant* group delay that we compensate exactly -- an
uncompensated filter delay would shift t1 and t2 by the same amount, which
cancels in TTFAB, but would corrupt the absolute onsets we also report.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_int16(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 audio, returning int16.

    Operates in float64 internally, then clips rather than wraps, so a resampler
    overshoot near full scale cannot flip sign.
    """
    x = np.asarray(x)
    if src_rate == dst_rate:
        return x.astype(np.int16, copy=True)

    g = gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g

    y = resample_poly(x.astype(np.float64), up, down)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse a (frames, channels) array to mono by averaging."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return np.rint(x.astype(np.float64).mean(axis=1)).astype(x.dtype)


def ms_to_samples(ms: float, rate: int) -> int:
    """Milliseconds to whole samples, rounded to nearest."""
    return int(round(ms * rate / 1000.0))


def samples_to_ms(n: int, rate: int) -> float:
    """Samples to milliseconds."""
    return n * 1000.0 / rate
