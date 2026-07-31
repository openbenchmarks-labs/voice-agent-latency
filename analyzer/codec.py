"""G.711 mu-law encode/decode, vectorised in numpy.

Python 3.13 removed the stdlib `audioop` module, which is what everyone used for
this. Rather than depend on the `audioop-lts` backport we implement G.711
directly -- we need vectorised operations over whole recordings anyway, and the
codec is the reference path for correlation, so it is worth
having it be explicit and tested rather than borrowed.

Correctness is pinned two ways in tests: against libsndfile's own mu-law
implementation (via soundfile), and against the hand-checked G.711 identities
for zero and full scale.
"""

from __future__ import annotations

import numpy as np

# G.711 constants. BIAS is added before the exponent search so that small
# magnitudes land in the first segment; CLIP is the largest magnitude the 8-bit
# code can represent.
_BIAS = 0x84
_CLIP = 32635

# exp_lut[i] = position of the highest set bit of i, i.e. 0 for i in {0,1},
# 1 for {2,3}, 2 for {4..7} ... 7 for {128..255}.
_EXP_LUT = np.zeros(256, dtype=np.uint8)
for _i in range(1, 256):
    _EXP_LUT[_i] = int(_i).bit_length() - 1


def linear_to_mulaw(pcm: np.ndarray) -> np.ndarray:
    """int16 linear PCM -> uint8 mu-law."""
    x = np.asarray(pcm, dtype=np.int32)

    sign = np.where(x < 0, 0x80, 0x00).astype(np.int32)
    mag = np.abs(x)
    np.minimum(mag, _CLIP, out=mag)
    mag += _BIAS

    exponent = _EXP_LUT[(mag >> 7) & 0xFF].astype(np.int32)
    mantissa = (mag >> (exponent + 3)) & 0x0F

    code = ~(sign | (exponent << 4) | mantissa)
    return (code & 0xFF).astype(np.uint8)


def mulaw_to_linear(mu: np.ndarray) -> np.ndarray:
    """uint8 mu-law -> int16 linear PCM."""
    u = (~np.asarray(mu, dtype=np.int32)) & 0xFF

    mantissa = u & 0x0F
    exponent = (u & 0x70) >> 4
    magnitude = ((mantissa << 3) + _BIAS) << exponent

    out = np.where(u & 0x80, _BIAS - magnitude, magnitude - _BIAS)
    return out.astype(np.int16)


def mulaw_roundtrip(pcm: np.ndarray) -> np.ndarray:
    """Push int16 PCM through a mu-law encode/decode cycle.

    This is how the correlation reference is prepared for the recording path:
    the recording has been companded by the telephony leg, so the reference must
    be companded identically or the matched filter loses peak sharpness for no
    reason.
    """
    return mulaw_to_linear(linear_to_mulaw(pcm))


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Float audio in [-1, 1] -> int16, clipped rather than wrapped."""
    y = np.asarray(x, dtype=np.float64) * 32767.0
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    """int16 -> float64 in [-1, 1)."""
    return np.asarray(x, dtype=np.float64) / 32768.0
