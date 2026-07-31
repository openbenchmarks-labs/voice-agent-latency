"""Render a bench run as a self-contained HTML page.

Styling is lifted verbatim from ../simulator/html_report.py so the two reports
read as one family: near-black background, dashed hairline borders, warm tan
accent, monospace labels, sparing green/red, light-theme toggle. Inline CSS only
-- opens offline from file://.

The data is entirely different (latency distributions, not funnels), but the
vocabulary is the same: `.card` for grouped panels, `.step` blocks for headline
numbers, `.citation-table` for tabular detail, `.foot` for the pass/fail band,
`.badge`/`.verdict` for per-call verdicts.

Deliberate choice: the "not yet publishable" block renders as a `.foot bad`
band -- the loudest element on the page. A latency report that hides its own
limitations is the thing this project exists not to be.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from vendors.base import stack_summary

_CSS = """
:root{
  --bg:#0a0a0a; --panel:#0c0c0c; --text:#ededed; --dim:#8a8a8a; --faint:#5a5a5a;
  --line:#1d1d1d; --line2:#2a2a2a; --accent:#e8c397; --ok:#84c08a; --bad:#d97a7a;
  --nav-bg:rgba(13,13,13,.72);
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;
}
body.light{
  --bg:#f7f4ef; --panel:#fffaf3; --text:#171717; --dim:#606060; --faint:#8a8176;
  --line:#ded6ca; --line2:#cfc4b5; --accent:#8f5b24; --ok:#2f7d52; --bad:#b64848;
  --nav-bg:rgba(255,250,243,.88);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
  font-feature-settings:"ss01","cv11";-webkit-font-smoothing:antialiased;line-height:1.5}
.bg{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.bg-grid{position:absolute;inset:-1px;
  background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
  background-size:64px 64px;
  -webkit-mask-image:radial-gradient(circle at 50% 0%,rgba(0,0,0,.9),transparent 70%);
  mask-image:radial-gradient(circle at 50% 0%,rgba(0,0,0,.9),transparent 70%)}
.bg-orb{position:absolute;top:-160px;right:-120px;width:540px;height:540px;border-radius:999px;
  display:none}
.wrap{position:relative;z-index:1;max-width:1180px;margin:0 auto;padding:24px 28px 80px}
.mono{font-family:var(--mono)}

/* nav */
.nav{display:flex;align-items:center;justify-content:space-between;gap:20px;
  padding:10px 16px;border:1px dashed var(--line2);background:var(--nav-bg);margin-bottom:40px}
.nav-meta{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.nav-actions{display:flex;align-items:center;gap:10px}
.jump-links{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.jump-link{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--dim);text-decoration:none;border:1px dashed var(--line2);padding:5px 8px}
.jump-link:hover{color:var(--accent);border-color:var(--accent)}
.theme-toggle{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--text);background:transparent;border:1px dashed var(--line2);padding:5px 9px;cursor:pointer}
.theme-toggle:hover{border-color:var(--accent);color:var(--accent)}

/* hero */
.tag-line{display:inline-flex;align-items:center;gap:10px;padding:6px 12px;border:1px dashed var(--line2);
  font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--dim);width:max-content;margin-bottom:20px}
.tag-dot{width:6px;height:6px;border-radius:999px;background:var(--accent);box-shadow:0 0 6px rgba(232,195,151,.55)}
h1{margin:0;font-size:clamp(2.4rem,5vw,3.4rem);line-height:.96;letter-spacing:-.04em;font-weight:520;color:var(--text)}
.sub{margin:14px 0 36px;font-family:var(--mono);font-size:12.5px;color:var(--dim)}

/* group card */
.card{position:relative;border:1px dashed var(--line2);background:transparent;padding:24px 24px 22px;margin-bottom:24px}
.card-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:22px}
.tag{font-family:var(--mono);font-size:11px;letter-spacing:.04em;padding:4px 9px;border:1px dashed var(--line2);
  color:var(--dim);text-transform:lowercase}
.tag-case{color:var(--text);border-color:var(--line2)}
.tag-kind{color:var(--accent);border-color:rgba(232,195,151,.3)}
.tag-drop{color:var(--bad);border-color:rgba(217,122,122,.35)}

/* headline stat blocks (funnel .step vocabulary) */
.funnel{display:flex;flex-wrap:wrap;align-items:stretch;gap:0}
.step{border:1px dashed var(--line2);padding:14px 16px;min-width:160px;margin:0 14px 14px 0}
.step-name{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.step-pct{font-size:30px;font-weight:560;line-height:1.1;margin:6px 0 4px;font-feature-settings:"tnum";letter-spacing:-.01em}
.step-unit{font-size:15px;color:var(--faint);font-weight:400;margin-left:3px}
.step-det{font-family:var(--mono);font-size:11.5px;color:var(--dim)}
.step-ci{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:2px}
.step-nr{display:inline-block;margin-top:8px;padding:2px 7px;border:1px dashed var(--line2);
  font-family:var(--mono);font-size:10px;letter-spacing:.04em;color:var(--dim)}

/* rollups */
.rollups{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:2px 0 18px}
.rollup{border:1px dashed var(--line);padding:10px 12px;background:rgba(255,255,255,.012)}
.rollup-title{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:8px}
.rollup-list{display:flex;flex-wrap:wrap;gap:7px}
.rollitem{font-family:var(--mono);font-size:11px;color:var(--dim);border:1px dashed var(--line2);padding:3px 7px;max-width:100%;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rollitem strong{color:var(--text);font-weight:600}

/* tables */
.citation-card{border:1px dashed var(--line2);padding:16px 18px;margin:0 0 24px;background:rgba(255,255,255,.012)}
.citation-title{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-bottom:5px}
.citation-sub{font-family:var(--mono);font-size:11px;color:var(--dim);margin:0 0 12px}
.citation-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
.citation-table th,.citation-table td{padding:8px 10px;border-top:1px dashed var(--line);text-align:left}
.citation-table th{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.citation-table td.num{text-align:right;color:var(--text);font-variant-numeric:tabular-nums}
/* a header over a numeric column has to be right-aligned too, or it floats away
   from the figures it labels */
.citation-table th.num{text-align:right}
.citation-table tr.dropped td{color:var(--faint)}

/* distribution strip: one tick per call, positioned by value */
.dist-strip{position:relative;height:56px;border:1px dashed var(--line);margin:4px 0 8px;
  background:rgba(255,255,255,.012)}
.tick{position:absolute;top:8px;width:1px;height:26px;background:var(--accent);opacity:.75}
.tick.dropped{background:var(--bad);opacity:.5;height:14px;top:20px}
.dist-axis{position:absolute;left:0;right:0;bottom:5px;height:14px}
.dist-axis span{position:absolute;transform:translateX(-50%);font-family:var(--mono);
  font-size:9.5px;color:var(--faint);white-space:nowrap}
.dist-median{position:absolute;top:4px;width:1px;height:34px;background:var(--text);opacity:.5}
.dist-legend{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.04em;margin-bottom:14px}

/* turn curve: median TTFAB by conversation position */
.turn-curve{display:flex;align-items:flex-end;gap:14px;height:190px;
  border:1px dashed var(--line);padding:14px 16px 0;background:rgba(255,255,255,.012)}
.tc-col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;
  align-items:center;height:100%}
.tc-bar{width:100%;max-width:78px;background:rgba(232,195,151,.18);
  border:1px dashed rgba(232,195,151,.5);border-bottom:none;min-height:2px}
.tc-val{font-family:var(--mono);font-size:11.5px;color:var(--text);
  font-variant-numeric:tabular-nums;margin-bottom:5px}
.tc-name{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint);margin-top:7px}
.tc-n{font-family:var(--mono);font-size:9.5px;color:var(--faint);margin-bottom:12px}

/* footer bands */
.foot{border:1px dashed var(--line2);padding:14px 18px;margin-top:8px;font-family:var(--mono);font-size:12px;letter-spacing:.02em}
.foot.bad{color:var(--bad);border-color:rgba(217,122,122,.35);background:rgba(217,122,122,.04)}
.foot.ok{color:var(--ok);border-color:rgba(132,192,138,.28);background:rgba(132,192,138,.04)}
.foot ul{margin:10px 0 0;padding-left:18px;line-height:1.7}
.foot li{margin:4px 0}
.note{font-family:var(--mono);font-size:11px;line-height:1.6;color:var(--faint);margin-top:16px}
.empty{font-family:var(--mono);color:var(--dim)}

/* per-call detail */
.section-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin:40px 0 16px}
.cell{border:1px dashed var(--line2);padding:20px 22px;margin-bottom:18px}
.cell.pass{border-color:rgba(132,192,138,.5)}
.cell.fail{border-color:rgba(217,122,122,.5)}
.cell-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:18px}
.cell-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;
  border:1px dashed var(--line);padding:10px 12px;margin:-4px 0 4px;background:rgba(255,255,255,.018)}
.sum-item{font-family:var(--mono);font-size:11.5px;color:var(--dim)}
.sum-item span{display:block;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:2px}
.sum-item strong{font-weight:600;color:var(--text)}
.badge{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border:1px dashed var(--line2)}
.badge.ok{color:var(--ok);border-color:rgba(132,192,138,.35)}
.badge.bad{color:var(--bad);border-color:rgba(217,122,122,.35)}
.badge.faint{color:var(--faint)}
.verdict{font-family:var(--mono);font-size:11px;letter-spacing:.08em;padding:3px 10px;border:1px dashed}
.verdict.ok{color:var(--ok);border-color:rgba(132,192,138,.4)}
.verdict.bad{color:var(--bad);border-color:rgba(217,122,122,.4)}
.verdict.faint{color:var(--faint);border-color:var(--line2)}
.cell-contam{display:inline-flex;gap:6px;margin-left:auto}
.cell-notes{border-top:1px dashed var(--line);margin-top:16px;padding-top:14px}
.cell-notes ul{margin:8px 0 0;padding-left:18px;font-family:var(--mono);font-size:11.5px;line-height:1.6;color:var(--dim)}

/* call timeline: the measured interval, drawn to scale */
.timeline{border:1px dashed var(--line);padding:12px 14px;margin-top:14px;background:rgba(255,255,255,.012)}
.tl-label{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-bottom:10px}
.tl-track{position:relative;height:26px;border-bottom:1px dashed var(--line)}
.tl-seg{position:absolute;top:4px;height:14px;border:1px dashed var(--line2)}
.tl-seg.greeting{background:rgba(132,192,138,.14);border-color:rgba(132,192,138,.4)}
.tl-seg.ours{background:rgba(232,195,151,.16);border-color:rgba(232,195,151,.45)}
.tl-seg.response{background:rgba(132,192,138,.2);border-color:rgba(132,192,138,.55)}
.tl-gap{position:absolute;top:9px;height:1px;background:var(--accent)}
.tl-gap-label{position:absolute;top:-6px;transform:translateX(-50%);font-family:var(--mono);
  font-size:10px;color:var(--accent);white-space:nowrap;background:var(--bg);padding:0 4px}
.tl-seg.dropped{opacity:.35}
.tl-gap.dropped{background:var(--bad)}
.tl-marks{position:relative;height:15px;margin-top:3px}
.tl-marks span{position:absolute;transform:translateX(-50%);font-family:var(--mono);
  font-size:9.5px;color:var(--accent);white-space:nowrap}
.tl-key{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-family:var(--mono);font-size:10px;color:var(--faint)}
.tl-key i{display:inline-block;width:9px;height:9px;margin-right:5px;vertical-align:-1px;border:1px dashed}
.tl-key i.greeting{background:rgba(132,192,138,.14);border-color:rgba(132,192,138,.4)}
.tl-key i.ours{background:rgba(232,195,151,.16);border-color:rgba(232,195,151,.45)}
.tl-key i.response{background:rgba(132,192,138,.2);border-color:rgba(132,192,138,.55)}
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt_ms(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _discard_color(usable: int, attempts: int) -> str:
    """Discard rate colouring. A vendor that fails to answer is worse than a
    slightly slower one, so this is scored, not decoration."""
    if not attempts:
        return "var(--faint)"
    rate = usable / attempts
    if rate >= 0.9:
        return "var(--ok)"
    if rate >= 0.7:
        return "var(--accent)"
    return "var(--bad)"


def _step(name: str, value: str, detail: str = "", *, color: str = "var(--text)",
          unit: str = "", ci: str = "", note: str = "") -> str:
    body = (f'<div class="step-pct" style="color:{color}">{value}'
            f'{f"<span class=chr(34)step-unitchr(34)>{unit}</span>" if unit else ""}</div>')
    # (built without f-string quote gymnastics below)
    unit_html = f'<span class="step-unit">{_esc(unit)}</span>' if unit else ""
    body = f'<div class="step-pct" style="color:{color}">{_esc(value)}{unit_html}</div>'
    if detail:
        body += f'<div class="step-det">{_esc(detail)}</div>'
    if ci:
        body += f'<div class="step-ci">{_esc(ci)}</div>'
    if note:
        body += f'<div class="step-nr">{_esc(note)}</div>'
    return f'<div class="step"><div class="step-name">{_esc(name)}</div>{body}</div>'


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #


def _headline(report: dict) -> str:
    onset = report.get("ttfab_onset_ms") or {}
    content = report.get("ttfab_content_ms") or {}
    ttfg = report.get("ttfg_from_recording_start_ms") or {}
    attempts = report.get("attempts", 0)
    usable = report.get("usable", 0)
    n = report.get("ttfab_onset_n", 0)

    multi = report.get("turns_per_call", 1) > 1
    turn_attempts = report.get("turn_attempts", 0)
    turns_usable = report.get("turns_usable", 0)

    steps = [
        _step("TTFAB p50", _fmt_ms(onset.get("p50")),
              f"n={n} turns" if multi else f"n={n}", unit=" ms",
              color="var(--accent)",
              ci=f'p90 {_fmt_ms(onset.get("p90"))} · p95 {_fmt_ms(onset.get("p95"))}'
                 f' · p99 {_fmt_ms(onset.get("p99"))}' if onset else ""),
        _step("TTFAB content p50", _fmt_ms(content.get("p50")),
              f'p95 {_fmt_ms(content.get("p95"))}' if content else "", unit=" ms"),
    ]

    if multi:
        # Both denominators matter: a call can survive while some of its turns do
        # not, so one number cannot stand for the other.
        steps.append(_step(
            "usable turns", f"{turns_usable}/{turn_attempts}",
            f"across {usable}/{attempts} calls",
            color=_discard_color(turns_usable, turn_attempts),
            note=f"{report['turns_per_call']} turns per call"))
    else:
        steps.append(_step("usable calls", f"{usable}/{attempts}",
                           "discards are a result, not noise",
                           color=_discard_color(usable, attempts)))

    consistency = report.get("consistency_p95_over_p50")
    if consistency is not None:
        beyond = report.get("beyond_2x_median_fraction")
        steps.append(_step("consistency", f"{consistency:.2f}", "p95 / p50",
                           color="var(--dim)",
                           ci=f"{beyond:.0%} beyond 2x median" if beyond is not None else ""))

    negatives = report.get("negative_ttfab_count") or 0
    if negatives:
        steps.append(_step("barge-in", str(negatives),
                           "vendor spoke before we finished", color="var(--bad)"))

    return f'<div class="funnel">{"".join(steps)}</div>'


def _measurements(results: list[dict]) -> tuple[list[float], list[float]]:
    """Every measurement in the run, as (kept, dropped).

    One entry per TURN, not per call -- a 4-turn call contributes four points, and
    plotting only the first would hide three quarters of the data.
    """
    kept: list[float] = []
    dropped: list[float] = []
    for result in results:
        call_failed = bool(result.get("discard_reason"))
        turns = result.get("turns") or [result]
        for turn in turns:
            value = turn.get("ttfab_onset_ms")
            if value is None:
                continue
            if call_failed or turn.get("discard_reason"):
                dropped.append(value)
            else:
                kept.append(value)
    return kept, dropped


def _distribution(results: list[dict]) -> str:
    """One tick per measured turn, positioned by TTFAB. Shows the shape percentiles
    hide."""
    values, dropped = _measurements(results)
    if not values:
        return ""

    lo = min(values + dropped)
    hi = max(values + dropped)
    span = max(1.0, hi - lo)
    pad = span * 0.06
    lo, hi = lo - pad, hi + pad
    span = hi - lo

    def pos(v: float) -> float:
        return 100.0 * (v - lo) / span

    ticks = "".join(f'<i class="tick" style="left:{pos(v):.2f}%"></i>' for v in values)
    ticks += "".join(f'<i class="tick dropped" style="left:{pos(v):.2f}%"></i>'
                     for v in dropped)

    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    ticks += f'<i class="dist-median" style="left:{pos(median):.2f}%"></i>'

    # De-duplicate the axis labels: with a tight spread the median can coincide
    # with an end, and two labels on the same pixel read as a rendering fault.
    marks, seen = [], set()
    for value in (lo + pad, median, hi - pad):
        key = round(pos(value), 1)
        if key in seen:
            continue
        seen.add(key)
        marks.append(f'<span style="left:{pos(value):.2f}%">{_fmt_ms(value)}</span>')
    axis = "".join(marks)

    return (
        '<div class="dist-legend">every measured turn, placed by TTFAB · tan = kept, '
        'red = discarded (shown for shape only, excluded from all figures) · '
        'pale line = median</div>'
        f'<div class="dist-strip">{ticks}<div class="dist-axis">{axis}</div></div>'
    )


def _turn_curve(report: dict) -> str:
    """Does the agent slow down as conversation context grows?

    The finding this whole multi-turn path exists for. Pooled medians hide a
    reply that gets slower with every turn of context.
    """
    per_turn = report.get("per_turn") or {}
    if len(per_turn) < 2:
        return ""

    indices = sorted(per_turn, key=int)
    p50s = [per_turn[i].get("p50") for i in indices]
    known = [v for v in p50s if v is not None]
    if not known:
        return ""

    top = max(known) * 1.15
    bars = ""
    for i, value in zip(indices, p50s):
        height = 0.0 if value is None else 100.0 * value / top
        label = "—" if value is None else f"{value:,.0f}"
        bars += (
            '<div class="tc-col">'
            f'<div class="tc-val">{label}</div>'
            f'<div class="tc-bar" style="height:{height:.1f}%"></div>'
            f'<div class="tc-name">turn {_esc(i)}</div>'
            f'<div class="tc-n">n={per_turn[i].get("n", 0)}</div>'
            "</div>"
        )

    rows = ""
    for key, label in (("p50", "p50"), ("p90", "p90"), ("p95", "p95")):
        cells = "".join(
            f'<td class="num">{per_turn[i][key]:,.0f}</td>'
            if per_turn[i].get(key) is not None else '<td class="num">—</td>'
            for i in indices
        )
        rows += f"<tr><td>{label}</td>{cells}</tr>"
    usable = "".join(
        f'<td class="num">{per_turn[i]["usable"]}/{per_turn[i]["attempts"]}</td>'
        for i in indices
    )
    rows += f"<tr><td>usable turns</td>{usable}</tr>"
    heads = "".join(f'<th class="num">turn {_esc(i)}</th>' for i in indices)

    first, last = known[0], known[-1]
    delta = last - first
    direction = ("rises" if delta > 0 else "falls" if delta < 0 else "is flat")
    colour = "var(--bad)" if delta > 0 else "var(--ok)"

    return f"""<section class="card" id="turns">
  <div class="card-head">
    <span class="tag tag-case">turn curve</span>
    <span class="tag tag-kind">does context growth cost latency?</span>
  </div>
  <div class="turn-curve">{bars}</div>
  <p class="citation-sub" style="margin-top:14px">Median TTFAB by position in the
  conversation. Turn 1 is a fresh context; each later turn carries everything
  before it. Here the median <strong style="color:{colour}">{direction}</strong>
  by {abs(delta):,.0f} ms from turn {indices[0]} to turn {indices[-1]}.</p>
  <table class="citation-table"><thead><tr><th></th>{heads}</tr></thead>
  <tbody>{rows}</tbody></table>
</section>"""


def _vendor_panel(report: dict, applied: dict) -> str:
    used = report.get("vendor_defaults_used") or applied.get("defaults_used") or {}
    unsupported = report.get("vendor_unsupported") or applied.get("unsupported") or []
    # Each vendor's receipt names what its own platform exposes; resolving the
    # aliases here keeps the panel from printing a dash for a value the receipt
    # actually carries. See vendors.base.stack_summary.
    summary = stack_summary(used)

    def item(label: str, value) -> str:
        return (f'<span class="rollitem">{_esc(label)} <strong>'
                f'{_esc(value if value not in (None, "") else "—")}</strong></span>')

    stack = "".join([
        item("llm", summary["model"]),
        item("stt", summary["stt"]),
        item("tts", summary["voice"]),
    ])
    turn = "".join(
        [item(key.replace("_", " "), value)
         for key, value in summary["endpointing"].items()]
        + ([item("idle", summary["idle"])] if summary["idle"] else [])
    ) or item("endpointing", None)
    provenance = "".join([
        item("config sha256", str(report.get("vendor_config_sha256") or "")[:16] + "…"),
        item("version", summary["version"]),
        item("tools", ", ".join(summary["tools"]) or "none"),
    ])
    cannot = "".join(f'<span class="rollitem" style="color:var(--bad)">{_esc(k)}</span>'
                     for k in unsupported) or '<span class="rollitem">—</span>'

    return f"""<section class="card" id="config">
  <div class="card-head">
    <span class="tag tag-case">{_esc(report.get("vendor", "vendor"))}</span>
    <span class="tag tag-kind">closed division · defaults as shipped</span>
  </div>
  <div class="rollups">
    <div class="rollup"><div class="rollup-title">stack the vendor chose</div>
      <div class="rollup-list">{stack}</div></div>
    <div class="rollup"><div class="rollup-title">turn-taking knobs (these set TTFAB)</div>
      <div class="rollup-list">{turn}</div></div>
    <div class="rollup"><div class="rollup-title">provenance</div>
      <div class="rollup-list">{provenance}</div></div>
    <div class="rollup"><div class="rollup-title">cannot be pinned on this vendor</div>
      <div class="rollup-list">{cannot}</div></div>
  </div>
  <p class="note">Model, voice and STT were left unset, so these are the platform's own
  defaults — the product as a new signup receives it. Anything listed as unpinnable is
  recorded rather than footnoted: it is the reason this vendor cannot be compared on
  that axis at all.</p>
</section>"""


def _discard_table(report: dict) -> str:
    discards = report.get("discards") or {}
    multi = report.get("turns_per_call", 1) > 1
    # Discards are counted per TURN on a multi-turn run, so the denominator has to
    # be turn attempts. Calling them "calls" would understate the rate several-fold.
    unit = "turns" if multi else "calls"
    attempts = (report.get("turn_attempts") or 0) if multi else report.get("attempts", 0)
    attempts = attempts or report.get("attempts", 0)

    if not discards:
        return ('<section class="citation-card" id="discards">'
                f'<div class="citation-title">Discarded {unit}</div>'
                f'<p class="citation-sub">None. All {attempts} {unit} produced a '
                'usable measurement.</p></section>')

    rows = "".join(
        '<tr class="dropped">'
        f'<td>{_esc(reason)}</td>'
        f'<td class="num">{count}</td>'
        f'<td class="num">{count / attempts:.0%}</td>'
        f'<td>{_esc(_DISCARD_MEANING.get(reason, ""))}</td>'
        '</tr>'
        for reason, count in sorted(discards.items(), key=lambda kv: -kv[1])
    )
    total = sum(discards.values())
    note = ("" if not multi else
            " A whole-call failure (a poisoned greeting, say) invalidates every turn "
            "in that conversation; a turn-level failure costs only that turn.")
    return f"""<section class="citation-card" id="discards">
  <div class="citation-title">Discarded {unit}</div>
  <p class="citation-sub">{total} of {attempts} {unit} were refused rather than measured.
  A platform that fails to respond is worse than one that is slightly slower, so this
  is reported, never folded into a denominator.{note}</p>
  <table class="citation-table"><thead><tr><th>Reason</th><th class="num">{unit.title()}</th>
  <th class="num">Of attempts</th><th>What it means</th></tr></thead>
  <tbody>{rows}</tbody></table>
</section>"""


_DISCARD_MEANING = {
    "idle_filler": "the agent spoke during our silence; that prompt would otherwise "
                   "have been recorded as its answer",
    "idle_filler_unassessable_clipped_greeting":
        "the recording opened mid-greeting, so we cannot tell a second greeting "
        "fragment from an idle prompt — the check was skipped, not passed",
    "no_response": "no reply within the listening window",
    "barged_greeting": "we started talking before the greeting finished",
    "barged_reply": "our next question interrupted this reply",
    "vad_disagree": "two independent speech detectors disagreed on the onset",
    "no_greeting": "the agent never spoke",
    "audio_missing": "recording absent or too short",
    # Scripted-dialog mode: the ways a detected speech-end can be untrustworthy.
    "double_talk": "the agent was still speaking as our question ended, so the "
                   "end of our speech is not a clean starting point",
    "offset_disagree": "two independent speech detectors disagreed about where "
                       "our own question ended",
    "dead_air": "the call connected but carried no speech at all — neither "
                "side said anything",
    # Synthesised by harness/bench.py when a turn is unusable but carries no
    # reason. Should never appear; if it does, the analyzer failed to say why.
    "unknown": "unusable for a reason the analyzer did not record — a bug, "
               "not a property of the platform",
    "turn_count_mismatch": "the recording does not contain as many of our "
                           "questions as we asked, so replies cannot be paired "
                           "to them with confidence",
    "channel_map_suspect": "the two sides of the call appear to be on the "
                           "opposite stereo channels from the ones assumed",
    # Reference mode (archived runs measured against known audio).
    "unlocatable": "our own audio could not be found in the recording (legacy "
                   "reference mode)",
    "drift": "audio was stretched or lost in transit; timing untrustworthy "
             "(legacy reference mode)",
    "out_of_order": "a question matched the wrong place in the recording "
                    "(legacy reference mode)",
}


def _method_note(report: dict) -> str:
    """How the two endpoints were found. Different sentence per mode, because
    they are genuinely different measurements and the reader deserves to know
    which one produced the numbers above."""
    if report.get("mode") == "scripted_dialog":
        how_ours = ("our own speech is located by a two-stage speech detector on "
                    "our side of the recording, and t1 is where that speech ends")
    else:
        how_ours = ("our own speech is located by matched filtering against the "
                    "exact bytes we played")
    return (f'<p class="note">Every figure is derived by re-reading saved audio: '
            f'{how_ours}, the agent\'s by the same detector on its side. No floor '
            f'has been subtracted from any number. Discarded calls are excluded '
            f'from every figure but always shown.</p>')


def _buffering_panel(report: dict) -> str:
    r = report.get("buffering_correlation_r")
    if r is None:
        return ""
    n = report.get("buffering_n", 0)
    unit = "turns" if report.get("turns_per_call", 1) > 1 else "calls"
    buffered = r > 0.5
    verdict = ("reply length predicts when speech starts — the TTS appears to wait for "
               "the whole answer before speaking"
               if buffered else
               "reply length does not predict when speech starts — the TTS appears to "
               "stream as tokens arrive")
    colour = "var(--bad)" if buffered else "var(--ok)"
    return f"""<section class="citation-card" id="buffering">
  <div class="citation-title">Does the TTS wait for the full reply?</div>
  <p class="citation-sub">Correlation between response duration and TTFAB across
  {n} {unit}. Reported, never corrected for.</p>
  <div class="funnel">{_step("correlation r", f"{r:+.3f}", verdict, color=colour)}</div>
</section>"""


def _timeline(result: dict) -> str:
    """The whole conversation drawn to scale, one exchange per turn.

    Every turn's question, reply and measured interval, so a 4-turn call shows
    four gaps rather than only the first.
    """
    turns = result.get("turns") or [result]
    drawable = [t for t in turns
                if t.get("t1_ms") is not None and t.get("t2_ms") is not None]
    if not drawable:
        return ""

    greet_on = result.get("greeting_onset_ms")
    greet_end = result.get("greeting_end_ms")

    ends = [t["t2_ms"] + (t.get("vendor_response_duration_ms") or 400.0)
            for t in drawable]
    hi = max(ends) * 1.02 or 1.0
    span = max(1.0, hi)

    def left(v: float) -> float:
        return max(0.0, 100.0 * v / span)

    def width(a: float, b: float) -> float:
        return max(0.4, 100.0 * (b - a) / span)

    segs = ""
    if greet_on is not None and greet_end is not None:
        segs += (f'<div class="tl-seg greeting" style="left:{left(greet_on):.2f}%;'
                 f'width:{width(greet_on, greet_end):.2f}%"></div>')

    gaps = ""
    for turn in drawable:
        t1, t2 = turn["t1_ms"], turn["t2_ms"]
        duration = turn.get("vendor_response_duration_ms") or 400.0
        start = turn.get("stimulus_start_ms")
        dropped = " dropped" if turn.get("discard_reason") else ""

        if start is not None:
            segs += (f'<div class="tl-seg ours{dropped}" '
                     f'style="left:{left(start):.2f}%;'
                     f'width:{width(start, t1):.2f}%"></div>')
        segs += (f'<div class="tl-seg response{dropped}" style="left:{left(t2):.2f}%;'
                 f'width:{width(t2, t2 + duration):.2f}%"></div>')
        gaps += (f'<div class="tl-gap{dropped}" style="left:{left(t1):.2f}%;'
                 f'width:{width(t1, t2):.2f}%"></div>')
        if len(drawable) <= 2:
            gaps += (f'<div class="tl-gap-label" '
                     f'style="left:{left((t1 + t2) / 2):.2f}%">'
                     f'{_fmt_ms(t2 - t1)} ms</div>')

    # With four turns the labels collide, so number the turns underneath instead.
    marks = ""
    if len(drawable) > 2:
        marks = '<div class="tl-marks">' + "".join(
            f'<span style="left:{left((t["t1_ms"] + t["t2_ms"]) / 2):.2f}%">'
            f'{_esc(t.get("index", "?"))}·{_fmt_ms(t["t2_ms"] - t["t1_ms"])}</span>'
            for t in drawable
        ) + "</div>"

    return f"""<div class="timeline">
  <div class="tl-label">call timeline · to scale · {len(drawable)} turn(s)</div>
  <div class="tl-track">{segs}{gaps}</div>
  {marks}
  <div class="tl-key">
    <span><i class="greeting"></i>their greeting</span>
    <span><i class="ours"></i>our question</span>
    <span><i class="response"></i>their reply</span>
    <span style="color:var(--accent)">— measured interval (TTFAB), labelled turn·ms</span>
  </div>
</div>"""


def _call_cell(result: dict) -> str:
    reason = result.get("discard_reason")
    turns = result.get("turns") or []
    multi = len(turns) > 1
    kept = [t for t in turns if not t.get("discard_reason")] if multi else []

    ok = reason is None
    cls = "pass" if ok and (not multi or kept) else "fail"
    if reason is not None:
        verdict = f'<span class="verdict bad">{_esc(reason)}</span>'
    elif multi:
        cls = "pass" if len(kept) == len(turns) else "fail"
        style = "ok" if len(kept) == len(turns) else "bad"
        verdict = (f'<span class="verdict {style}">'
                   f'{len(kept)}/{len(turns)} turns measured</span>')
    else:
        verdict = '<span class="verdict ok">measured</span>'

    flags = "".join(f'<span class="badge faint">{_esc(f)}</span>'
                    for f in (result.get("flags") or []))

    if multi:
        # Summarise the CALL. The old layout showed turn 1's figures, which is
        # actively misleading when turn 1 is the turn that was discarded.
        values = sorted(t["ttfab_onset_ms"] for t in kept
                        if t.get("ttfab_onset_ms") is not None)
        median = values[len(values) // 2] if values else None
        replies = [t["vendor_response_duration_ms"] for t in kept
                   if t.get("vendor_response_duration_ms") is not None]
        items = "".join([
            f'<div class="sum-item"><span>median ttfab</span><strong>'
            f'{_fmt_ms(median, 1)} ms</strong></div>',
            f'<div class="sum-item"><span>range</span><strong>'
            + (f'{_fmt_ms(values[0])}–{_fmt_ms(values[-1])} ms' if values else "—")
            + '</strong></div>',
            f'<div class="sum-item"><span>turns measured</span><strong>'
            f'{len(kept)}/{len(turns)}</strong></div>',
            f'<div class="sum-item"><span>ttfg</span><strong>'
            f'{_fmt_ms(result.get("ttfg_ms"), 1)}{" ms" if result.get("ttfg_ms") else ""}'
            '</strong></div>',
            f'<div class="sum-item"><span>mean reply</span><strong>'
            + (f'{_fmt_ms(sum(replies) / len(replies))} ms' if replies else "—")
            + '</strong></div>',
        ])
    else:
        items = "".join([
            f'<div class="sum-item"><span>ttfab</span><strong>'
            f'{_fmt_ms(result.get("ttfab_onset_ms"), 1)} ms</strong></div>',
            f'<div class="sum-item"><span>content</span><strong>'
            f'{_fmt_ms(result.get("ttfab_content_ms"), 1)} ms</strong></div>',
            f'<div class="sum-item"><span>ttfg</span><strong>'
            f'{_fmt_ms(result.get("ttfg_ms"), 1)}{" ms" if result.get("ttfg_ms") else ""}'
            '</strong></div>',
            f'<div class="sum-item"><span>reply length</span><strong>'
            f'{_fmt_ms(result.get("vendor_response_duration_ms"))} ms</strong></div>',
            f'<div class="sum-item"><span>match confidence</span><strong>'
            f'{result["psr"]:.2f}</strong></div>'
            if result.get("psr") is not None else
            '<div class="sum-item"><span>match confidence</span><strong>—</strong></div>',
            f'<div class="sum-item"><span>drift</span><strong>'
            f'{_fmt_ms(result.get("drift_ms"), 1)} ms</strong></div>',
        ])

    turn_rows = ""
    if multi:
        rows = "".join(
            f'<tr{" class=\"dropped\"" if t.get("discard_reason") else ""}>'
            f'<td>turn {_esc(t.get("index"))}</td>'
            f'<td class="num">{_fmt_ms(t.get("ttfab_onset_ms"), 1)}</td>'
            f'<td class="num">{_fmt_ms(t.get("vendor_response_duration_ms"))}</td>'
            f'<td class="num">'
            + (f'{t["psr"]:.2f}' if t.get("psr") is not None else "—")
            + "</td>"
            f'<td class="num">{_fmt_ms(t.get("drift_ms"), 1)}</td>'
            f'<td>{_esc(t.get("discard_reason") or "measured")}'
            + ("".join(f' <span class="badge faint">{_esc(f)}</span>'
                       for f in (t.get("flags") or [])))
            + "</td></tr>"
            for t in turns
        )
        turn_rows = (
            '<table class="citation-table" style="margin-top:14px">'
            '<thead><tr><th>turn</th><th class="num">TTFAB ms</th>'
            '<th class="num">reply ms</th><th class="num">match</th>'
            '<th class="num">drift ms</th><th>verdict</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )

    return f"""<section class="cell {cls}">
  <div class="cell-head">
    <span class="tag tag-case">{_esc(result.get("call_id", "?"))}</span>
    {flags}
    <span class="cell-contam">{verdict}</span>
  </div>
  <div class="cell-summary">{items}</div>
  {turn_rows}
  {_timeline(result)}
</section>"""


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #


def render_html(report: dict, results: list[dict], *, applied: dict | None = None,
                title: str = "Voice Agent Latency", subtitle: str = "") -> str:
    applied = applied or {}
    vendor = report.get("vendor", "vendor")
    attempts = report.get("attempts", 0)
    usable = report.get("usable", 0)
    sub = subtitle or f"{vendor} · {usable}/{attempts} usable · {report.get('run_id', '')}"

    turns_per_call = report.get("turns_per_call", 1)
    scope = (f"turns 1–{turns_per_call}" if turns_per_call > 1 else "turn 1")

    cells = "".join(_call_cell(r) for r in results)
    call_section = (f'<section id="calls"><div class="section-label">Every call · '
                    f'measured and discarded</div>{cells}</section>' if cells else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} · {_esc(vendor)}</title><style>{_CSS}</style></head>
<body>
<div class="bg"><div class="bg-grid"></div><div class="bg-orb"></div></div>
<div class="wrap">
  <header class="nav">
    <span class="nav-meta">voice bench · {_esc(sub)}</span>
    <span class="nav-actions"><span class="jump-links">
      <a class="jump-link" href="#headline">headline</a>
      <a class="jump-link" href="#turns">turns</a>
      <a class="jump-link" href="#config">config</a>
      <a class="jump-link" href="#discards">discards</a>
      <a class="jump-link" href="#buffering">tts</a>
      <a class="jump-link" href="#calls">calls</a>
      <a class="jump-link" href="#limits">limits</a>
    </span><button class="theme-toggle" type="button"
      onclick="document.body.classList.toggle('light');localStorage.setItem('ob-theme',document.body.classList.contains('light')?'light':'dark')">theme</button></span>
  </header>
  <span class="tag-line"><span class="tag-dot"></span> time to first audio byte · inbound · {_esc(scope)}</span>
  <h1>{_esc(vendor)}</h1>
  <p class="sub">{_esc(sub)}</p>

  <section class="card" id="headline">
    <div class="card-head">
      <span class="tag tag-case">TTFAB</span>
      <span class="tag tag-kind">caller speech end → agent speech start</span>
      <span class="tag">measured from the recording, never from API timestamps</span>
    </div>
    {_headline(report)}
    {_distribution(results)}
  </section>

  {_turn_curve(report)}
  {_vendor_panel(report, applied)}
  {_discard_table(report)}
  {_buffering_panel(report)}
  {call_section}
  {_method_note(report)}
</div>
<script>if(localStorage.getItem('ob-theme')==='light')document.body.classList.add('light');</script>
</body></html>"""


def write_html(report: dict, results: list[dict], run_dir: Path, *,
               applied: dict | None = None) -> Path:
    """Render to <run_dir>/report.html and return the path."""
    run_dir = Path(run_dir)
    if applied is None:
        applied_path = run_dir / "applied_config.json"
        if applied_path.exists():
            applied = json.loads(applied_path.read_text())
    out = run_dir / "report.html"
    out.write_text(render_html(report, results, applied=applied))
    return out
