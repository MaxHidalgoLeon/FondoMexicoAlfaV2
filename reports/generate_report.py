#!/usr/bin/env python
"""
Self-contained strategy report generator.

Usage:
    python reports/generate_report.py
    python reports/generate_report.py --source yahoo --start 2020-01-01
    python reports/generate_report.py --source mock --hedge
    python reports/generate_report.py --source bloomberg --hedge --out reports/output/my_report.html
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

from src.pipeline import run_pipeline
from reports.charts import build_dashboard_html


def parse_args():
    p = argparse.ArgumentParser(description="Generate Fondo Mexico strategy HTML report.")
    p.add_argument("--source", choices=["mock", "yahoo", "bloomberg", "lseg"], default="mock")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--hedge", action="store_true", help="Include Layer 2 hedge overlay.")
    p.add_argument("--optimizer", choices=["mv", "cvar", "robust", "both"], default="mv",
                   help="Portfolio optimizer (default: mv). Use 'both' to compare MV vs min-CVaR.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or str(Path(__file__).parent / "output" / "strategy_report.html")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Running pipeline  source={args.source}  {args.start} to {args.end} ...")
    results = run_pipeline(
        hedge_mode=args.hedge,
        data_source=args.source,
        start_date=args.start,
        end_date=args.end,
        optimizer=args.optimizer,
    )

    print("[2/3] Building dashboard ...")
    html = build_dashboard_html(results, hedge_mode=args.hedge, data_source=args.source)

    print(f"[3/3] Saving -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone. Open in browser:\n  file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
