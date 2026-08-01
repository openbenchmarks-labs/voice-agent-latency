# OpenBenchmarks Voice Agent Latency Benchmark

How long a caller waits before a voice AI agent starts speaking, measured from real phone calls.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/voice-agent-latency

This repo is the open data + code mirror of that page — the caller harness, the
offline audio analyzer, and every measured turn with the recording it was
measured from.

The runner itself is barebones and local: it places calls, measures them, and
writes everything under `runs/<run_id>/`. It has no database, posts nothing
anywhere, and needs no infrastructure beyond a carrier account.

**TTFAB (Time To First Audio Byte)** is the gap between the moment the caller stops speaking and the moment the agent's audio starts — the silence a real caller sits through on every turn. It is measured from a saved dual-channel recording of the actual phone call, never from a platform's own timestamps. Lower is better.

Both endpoints are found in the recording: `t1`, the end of our caller's speech, and `t2`, the start of the agent's reply. TTFAB is `t2 − t1`, per turn. Turns that fail a quality gate — the two sides talking over each other, our two detectors disagreeing, no reply at all — are discarded, and the discards are published per reason.

## Endpoints

- **Live benchmark UI** — https://openbenchmarks.com/voice-agent-latency
- **JSON API** — https://openbenchmarks.com/api/benchmarks/voice-agent-latency
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json
- **MCP server discovery** — https://openbenchmarks.com/.well-known/mcp.json

## Current leaderboard

Generated 2026-08-01 · analyzer `2.4.0`

| Platform | TTFAB p50 | TTFAB p95 | p95/p50 | Cost/min | Usable turns |
|---|---|---|---|---|---|
| Telnyx | **1,296 ms** | 1,856 ms | 1.43× | $0.0500 | 419/432 |
| ElevenLabs | **1,424 ms** | 1,768 ms | 1.24× | $0.0794 | 429/432 |
| Bland AI | 1,520 ms | 2,248 ms | 1.48× | $0.1408 | 429/432 |
| Vapi | 1,558 ms | 2,008 ms | 1.29× | $0.0836 | 382/432 |
| Retell AI | 1,740 ms | 2,259 ms | 1.30× | $0.1341 | 419/430 |

**2,078 usable turns across 550 calls**, from 2,158 measured — the other 80 were
measured and then discarded by a quality gate. Most of that is one 500-call
synchronised run where every platform was dialled with the *same question set at
the same instant*; see [Synchronised runs](#synchronised-runs). All of it
re-derives from the published audio; see
[Reproducing a number](#reproducing-a-number).

**Read the two columns together.** Telnyx has the lowest median but only the
third-tightest tail (1.43×). ElevenLabs is the most consistent platform on the
board at 1.24× — 128 ms slower at the median, but 88 ms *faster* at p95. The
widest tail belongs to Bland, at 1.48×.

That tail number matters more than it looks. Past roughly two seconds of silence
a caller assumes the line dropped and starts talking again, which collides with
the agent's reply and derails the turn. **Three platforms cross two seconds at
p95 — Bland, Retell and Vapi. Telnyx and ElevenLabs do not.** If your calls are
ordinary back-and-forth, the medians are what you feel. If a derailed turn is
expensive, the tail is.

`Cost/min` is what the platform's own billing API charged, over the seconds it
actually invoiced — not a rate card. Two caveats travel with it: Telnyx bills a
60-second minimum, so on ~44-second calls its invoiced rate is well below its
cost per minute of conversation ($0.0722); and the ElevenLabs account is on a
creator tier, so its figure is the list-price equivalent rather than money
charged. Both are recorded per call in `cost_notes`.

Read that literally: the 38 ms between Bland and Vapi is **inside the error
budget** and those two rows are not separable. The 128 ms between Telnyx and
ElevenLabs, the 96 ms between ElevenLabs and Bland, and the 182 ms between Vapi
and Retell are larger than it. What the data supports is the shape — the top four
sit within 262 ms of each other, Retell is 182 ms behind the next slowest, and
the tail spread (491 ms) slightly exceeds the median spread (444 ms).

## What's in this repo

| path | purpose |
|---|---|
| `harness/` | Places calls and steers the conversation. Never produces a number. |
| `harness/sync_bench.py` | Dials every platform at once, one question set at a time. See [Synchronised runs](#synchronised-runs). |
| `analyzer/` | The measurement. Pure: reads saved audio, no network, no credentials. |
| `analyzer/models/silero_vad.onnx` | The speech detector, vendored so the analyzer runs offline. |
| `analyzer/fixtures/` | Synthetic calls with known answers, for validating the analyzer. |
| `vendors/` | One read-only adapter per platform. Reads live config to build the receipt; never writes. |
| `carriers/` | The CPaaS that originates calls. Plivo only — see Methodology. |
| `config/vendors.yaml` | What each agent under test is configured to be. |
| `config/dialog.yaml` | The caller's script: voice, endpointing, and the 50 pinned question sets. |
| `tools/verify_run.py` | Re-derives published numbers from published audio. |
| `tools/blind_test.py` | Measures a synthetic call with a hidden gap, then reveals the answer. |
| `tools/setup_*_agent.py` | Provisions the agent under test on each platform, idempotently. |
| `tools/export_snapshot.py` | Turns a local run into the publishable per-call artifacts. |
| `tools/build_manifest.py` | Rebuilds `manifest.json`; `--check` says whether it is current. |
| `tools/reanalyze_run.py` | Re-measures your own saved runs. Costs no phone calls. |
| `tools/backfill_costs.py` | Asks each platform's billing API what your calls cost. |
| `tests/` | 397 tests. `pytest` needs no credentials and makes no network calls. |
| `runs/` | Where your own runs land — **gitignored**. See [Where results go](#where-results-go). |
| `data/voice-runs/` | Per-call artifacts: every turn's timings, flags, discards, cost, recording URL. Two generations — see below. |
| `data/voice-runs/README.md` | The artifact format, and what is redacted. |
| `data/latest-voice.json` | The snapshot the live page ingests. |
| `manifest.json` | Flat index of every run and call with recording URLs. Easy to ingest programmatically. |

## Reproducing a number

Every published turn is backed by a public recording and a deterministic analyzer, so you can re-derive any figure without an account on any platform:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/verify_run.py bench-telnyx-20260731-152959
```

That downloads each call's carrier tape from the URL in its artifact, **checks
the `sha256` and refuses on mismatch**, resamples it to the analyzer's 8 kHz
exactly as the harness did, re-measures every turn, and diffs its own answer
against the published one. `--all` does the whole board; the exit code is
non-zero if anything disagrees.


The pieces, if you would rather do it by hand: `data/voice-runs/<run_id>/call-NNN.json` carries `recording.url` and `recording.sha256` for the audio, `metadata` for the channel map and the script that was read, and `result.turns[]` for the per-turn `t1_ms`, `t2_ms` and `ttfab_onset_ms` being claimed.

## Running the benchmark yourself

The first three commands need no credentials and make no network calls:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                            # 397 tests
.venv/bin/python -m analyzer --gate-a                    # validate the analyzer against known answers
.venv/bin/python tools/blind_test.py                     # measure a synthetic call with a hidden gap

cp .env.example .env && $EDITOR .env                     # your Plivo + platform credentials
.venv/bin/python tools/setup_vapi_agent.py --dry-run     # provision the agent under test
.venv/bin/python -m harness.bench --vendor vapi --calls 10
```

Placing calls costs money on both legs — the carrier's and the platform's. The
bench refuses to run if the agent's live configuration disagrees with
`config/vendors.yaml`, because a published receipt has to describe the agent that
actually answered.

You will need your own Plivo account and your own agent on whichever platform you are measuring; the ids in `config/vendors.yaml` are ours and are inert without our keys.

### Synchronised runs

`harness.bench` measures one platform at a time, which leaves an hour between the
first vendor's call and the last one's. Anything that drifts in that hour —
platform load, model routing, carrier weather — lands on one column and not the
others, and it is indistinguishable from a real difference.

`harness.sync_bench` removes that confound. It dials every platform with the
**same question set at the same instant**, waits for the whole set to finish,
then moves to the next:

```bash
.venv/bin/python -m harness.sync_bench --calls 100
.venv/bin/python -m harness.sync_bench --calls 10 --vendors telnyx,vapi
```

The synchronisation point is the *dial*, not the setup. Each call runs on its own
thread and every thread waits on a `threading.Barrier` immediately before placing
the call, so the uneven cost of resolving a target and building a receipt is
absorbed before the clock matters. Without the barrier the spread between
placements is the setup cost, which is seconds.

Every set records how well that worked, in `runs/sync-<stamp>/sets.jsonl`:

```json
{"set_index": 0, "cases": ["price-basic", "data-cap", "free-trial"],
 "placement_spread_s": 0.077,
 "vendors": [{"vendor": "telnyx", "outcome": "usable",
              "placed_at": "2026-07-31T22:29:59.412Z", "ttfab_onset_ms": 1684.0}]}
```

**`placement_spread_s` is the honesty column.** It is the gap between the first
and last placement in the set — tens of milliseconds when the mode is working. If
it climbs into seconds, the calls were not simultaneous and the comparison is not
clean, and this is where that shows rather than being assumed.

The published run averaged **20–140 ms** of spread per set.

### Where results go

A run writes `runs/<run_id>/` and nothing else:

```
runs/bench-vapi-20260731-140212/
├── bench.json                  the report — percentiles, discards, receipts
├── report.html                 the same, rendered
├── applied_config.json         what the agent was actually running
├── caller_config.json          what our caller said, and how
└── call-000/
    ├── recording_raw.wav       the carrier's stereo tape
    ├── recording.wav           that tape at 8 kHz — what the analyzer reads
    ├── result.json             per-turn t1, t2, TTFAB, flags, discards
    ├── metadata.json           the script, the channel map, the receipts
    └── events.jsonl            every webhook and state change, timestamped
```

`runs/` is **gitignored, deliberately.** Those WAVs are 1–3 MB per call and are
regenerated every run — a 10-call run is 15–25 MB, and committing them would
grow the repo permanently for bytes nobody else needs. Git also cannot diff
them, so each re-run would add a fresh copy rather than a delta.

What gets published instead is the slim JSON under `data/voice-runs/`: every
turn's timings, flags and discards, with the recording referenced by URL and
checksum. That is roughly 4.5 MB for all 550 of our calls, it diffs cleanly, and
it is enough to re-derive every number.

`tools/reanalyze_run.py` re-measures your own runs in place — useful after an
analyzer change, since it costs no phone calls. Publishing audio somewhere
public is our own step and is not in this repo; if you want your runs shareable,
upload `recording_raw.wav` wherever you like and put the URL and `sha256` into
each artifact's `recording` block.

## Contributing a platform

1. Add `vendors/<platform>.py` implementing the four methods in `vendors/base.py`: `verify_agent`, `dial_target`, `applied_config`, `call_costs`. Adapters are **read-only by contract** — `tests/test_vendors.py` enforces it structurally, because a bench that can rewrite the thing it measures cannot publish a trustworthy receipt.
2. Add a block to `config/vendors.yaml` and register the slug in `vendors/registry.py`.
3. Add `tools/setup_<platform>_agent.py` so the agent under test can be provisioned from the committed config rather than clicked together.
4. Add `tests/test_vendor_<platform>.py`, mocking the transport. Then open a PR.

Corrections are welcome, including to numbers. If you think a figure is wrong,
`tools/verify_run.py` re-derives it from the published audio — if it disagrees
with what we published, that is a bug and an issue with the output attached is
the fastest way to get it fixed.

## Methodology

- **TTFAB, and only TTFAB.** Not measured: answer quality, voice quality, interruption handling, or platform features. Treat those as vendor claims until someone measures them.
- **Measured from audio, never from a timestamp.** A platform measures from where it stands, and the caller is not standing there. Its own recording of a call reads roughly 550 ms earlier than ours, and the latency it reports for itself runs roughly 490 ms below what we measure from that call's audio. The two agree, which is the point — both describe when a reply was produced, not when a caller heard it.
- **The carrier is never a platform under test.** Calls originate from Plivo, which is not on this board. Running a platform's agent over that platform's own network would invite a bias objection we could not answer with data. The leg that *answers* still belongs to whoever ships the number, and that is stated per platform on the live page.
- **Defaults are mostly the product, and we made three overrides.** Each agent runs the model, speech recognition and voice the platform gives a new signup, and the receipt records what it chose — pinning a stack would measure a platform you would have to configure to match. Three things we did change, all of them visible in `vendor_defaults_used`:
  - **Endpointing, where the platform exposes it as a fixed wait.** It is a timer sitting inside the number being measured. Vapi shipped 0.4 s (1.5 s after speech ending without punctuation) and now runs 0.1 s; Telnyx runs 0.1 s. Retell's value lives in `custom_stt_config.endpointing_ms` (1000 → 100 ms), which its adapter does **not** publish — so Retell's receipt does not contain the evidence for its own endpointing, and you should treat that one as our claim rather than a verified fact.
  - **ElevenLabs' stall phrase, disarmed** (`filler_armed: false`, `filler_timeout_seconds: -1.0`). Left on, the agent says "Hhmmmm...yeah." when the model is slow, and TTFAB would time the filler instead of the answer. Necessary for the metric to mean anything, and still an override on the row that wins the tail.
  - Nothing else. `tts_optimize_streaming_latency: 3` on ElevenLabs is in the receipt but was **not** set by our setup tool; we cannot show it is the platform default, so read it as unverified provenance.
- **"Same endpointing" does not mean "same turn-taking."** Telnyx pairs its 0.1 s timers with a semantic end-of-turn model (`eot_threshold: 0.8`, `eager_eot_threshold: 0.8`, `interrupt_prediction_threshold: 0.55`). ElevenLabs uses a learned turn model (`turn_model: turn_v3`, `turn_eagerness: normal`) with no fixed wait at all, and Bland exposes `interruption_threshold: 500`. So every platform here makes its own turn-taking decisions and the 0.1 s pin equalises one knob, not the behaviour. Earlier wording claimed Bland and ElevenLabs "expose no equivalent knob" — their own receipts show otherwise.
- **Half of Vapi's endpointing receipt is documentation, not observation.** `endpointing_source` records `start_speaking_plan: "api"` but `stop_speaking_plan: "vapi-documented-default"` — Vapi returns null for unset plans while still applying documented values server-side, so the adapter records the documented value and says so. It is honest, but it is not a live read, and the rest of this repo's receipts are.
- **Recording-path overhead is inside every figure.** We have not characterised the current path against a known-delay reference, so we quote no overhead figure and subtract none. Every number here is therefore an upper bound on the platform's own contribution.
- **Vapi's denominator is lower for a real reason.** 345 usable of 391, against 389/392 for the platforms either side of it. 45 of those turns were discarded because our two detectors disagreed about where its reply began by more than tolerance — a soft onset is a property of the platform, not of the instrument, and it is published as a discard rather than guessed at.
- **A discard is a turn we refuse to time, not a platform failure.** Discards are published per reason, because a platform that never answers is worse than one that answers slowly.
- **Two detectors, and they have to agree.** `t1` and `t2` come from Silero VAD with an energy refinement, cross-checked by an independent detector. Where the two disagree beyond tolerance the turn is discarded rather than averaged.
- **The analyzer is pure.** No credentials, no network, no clock. Given the same audio it returns the same number, which is what makes `verify_run.py` meaningful. `ANALYZER_VERSION` pins what "the same" means.

## License

MIT — see [LICENSE](LICENSE).
