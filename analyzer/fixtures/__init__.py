"""Synthetic fixtures with ground-truth onsets baked in.

Gate A is the only check in the whole pipeline that compares
the analyzer against a truth we constructed rather than against itself. Every
other validation is self-consistency, which a pipeline with a steady bias passes
perfectly.

Fixtures are assembled deterministically from the committed speech clips in
`sources/` -- see tools/make_fixture_sources.py for how those were produced.
Nothing here calls out to the network or to platform tools, so the fixtures build
identically on macOS and on the Linux VPS.

Sample-index convention, which is load-bearing and easy to get wrong:

    t1  index one PAST the last speech sample of our stimulus (exclusive end)
    t2  index OF the first speech sample of the vendor response (inclusive start)

so TTFAB == t2 - t1 with no off-by-one. Both the fixtures and the analyzer use
this convention; if they ever disagree, every reported number is biased by a
sample.
"""

from .generate import FIXTURES, Fixture, build, build_all

__all__ = ["FIXTURES", "Fixture", "build", "build_all"]
