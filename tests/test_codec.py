"""Pin the mu-law codec against an independent implementation.

`audioop` is gone in 3.13, so the reference here is libsndfile (via soundfile),
which has its own G.711 implementation we did not write.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from analyzer.codec import (
    float_to_int16,
    int16_to_float,
    linear_to_mulaw,
    mulaw_to_linear,
    mulaw_roundtrip,
)


def _libsndfile_mulaw_bytes(pcm: np.ndarray) -> np.ndarray:
    """Encode int16 PCM to mu-law using libsndfile, return the raw code bytes."""
    buf = io.BytesIO()
    sf.write(buf, pcm, 8000, format="RAW", subtype="ULAW")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def test_matches_libsndfile_across_full_int16_range():
    # Every distinct int16 value, so there is no sampling luck involved.
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    ours = linear_to_mulaw(pcm)
    theirs = _libsndfile_mulaw_bytes(pcm)
    assert ours.shape == theirs.shape
    mismatches = int(np.count_nonzero(ours != theirs))
    assert mismatches == 0, f"{mismatches} codes differ from libsndfile"


def test_decode_matches_libsndfile_for_all_256_codes():
    codes = np.arange(256, dtype=np.uint8)
    ours = mulaw_to_linear(codes)

    buf = io.BytesIO(codes.tobytes())
    theirs, _ = sf.read(
        buf, samplerate=8000, channels=1, format="RAW", subtype="ULAW", dtype="int16"
    )
    np.testing.assert_array_equal(ours, theirs)


def test_decode_encode_is_identity_except_for_the_dual_zero_code():
    # G.711 has two codes that decode to zero: 127 and 255. Re-encoding zero
    # canonicalises to 255, so 127 is the single legitimate mismatch. Every other
    # code must survive untouched -- if a second mismatch appears, a segment
    # boundary has drifted.
    codes = np.arange(256, dtype=np.uint8)
    back = linear_to_mulaw(mulaw_to_linear(codes))
    mismatches = np.nonzero(back != codes)[0]
    assert mismatches.tolist() == [127]
    assert back[127] == 255
    assert mulaw_to_linear(np.uint8([127]))[0] == 0


def test_roundtrip_error_stays_inside_the_companding_envelope():
    pcm = np.arange(-32768, 32768, dtype=np.int16)  # every value, no sampling luck
    out = mulaw_roundtrip(pcm)

    err = np.abs(out.astype(np.int64) - pcm.astype(np.int64))
    mag = np.abs(pcm.astype(np.int64))

    # mu-law is logarithmic, so the step size doubles per segment and the error
    # bound has to be relative. Measured worst case is ~6.5% in the 64-256 band
    # and ~3.2% above it; near-silence is bounded absolutely by the first-segment
    # step. Exact correctness is pinned by the libsndfile tests above -- this test
    # documents the companding envelope and catches gross regressions.
    near_zero = mag < 64
    assert err[near_zero].max() <= 4

    rel = err[~near_zero] / mag[~near_zero]
    assert rel.max() <= 0.07


def test_digital_silence_roundtrips_to_exact_zero():
    # Zero maps to code 255 and back to exactly zero. The fixtures depend on this:
    # the 500 ms gap between stimulus and response must stay exactly 500 ms after
    # the reference is companded, or Gate A measures the codec instead of the
    # analyzer.
    pcm = np.zeros(800, dtype=np.int16)
    assert np.array_equal(mulaw_roundtrip(pcm), pcm)


def test_sign_symmetry():
    pcm = np.array([-20000, -1000, -1, 1, 1000, 20000], dtype=np.int16)
    out = mulaw_roundtrip(pcm)
    # Companding is symmetric about zero, so +x and -x must map to the same
    # magnitude. An off-by-one in the sign bit shows up here.
    assert np.array_equal(np.abs(out[:3][::-1]), np.abs(out[3:]))


def test_clipping_is_saturating_not_wrapping():
    pcm = np.array([-32768, 32767], dtype=np.int16)
    out = mulaw_roundtrip(pcm)
    assert out[0] < -30000, "negative full scale wrapped"
    assert out[1] > 30000, "positive full scale wrapped"


@pytest.mark.parametrize("shape", [(0,), (1,), (3, 5)])
def test_shape_and_dtype_are_preserved(shape):
    pcm = np.zeros(shape, dtype=np.int16)
    mu = linear_to_mulaw(pcm)
    assert mu.shape == shape and mu.dtype == np.uint8
    assert mulaw_to_linear(mu).shape == shape


def test_float_int16_conversions_are_stable():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 0.999], dtype=np.float64)
    back = int16_to_float(float_to_int16(x))
    np.testing.assert_allclose(back, x, atol=1e-4)


def test_float_conversion_clips_out_of_range_input():
    x = np.array([-4.0, 4.0], dtype=np.float64)
    out = float_to_int16(x)
    assert out[0] == -32768 and out[1] == 32767
