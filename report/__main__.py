"""Regenerate an HTML report from a saved run.

    python3 -m report runs/bench-telnyx-20260728-170635

Reads bench.json + each call's result.json. Credential-free, like the analyzer --
anyone holding the run directory can rebuild the page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .html_report import write_html


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    bench_path = run_dir / "bench.json"
    if not bench_path.exists():
        raise FileNotFoundError(
            f"no bench.json in {run_dir} -- is this a bench run directory?"
        )
    report = json.loads(bench_path.read_text())

    results = []
    for call_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        result_path = call_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        result.setdefault("call_id", call_dir.name)
        results.append(result)
    return report, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m report", description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"not a directory: {args.run_dir}", file=sys.stderr)
        return 2
    try:
        report, results = load_run(args.run_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    out = write_html(report, results, args.run_dir)
    print(f"wrote {out}  ({len(results)} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
