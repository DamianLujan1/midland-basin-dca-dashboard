"""EUR sensitivity to the forecast assumptions (prints a Markdown table).

The fits do not change between cases; only the forecast tail does.

Run:  python scripts/sensitivity.py   (from the repo root)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (DMIN_ANNUAL, ECON_LIMIT_BPM, MAX_LIFE_MONTHS,  # noqa: E402
                          run)

CASES = [
    ("Base case", DMIN_ANNUAL, ECON_LIMIT_BPM, MAX_LIFE_MONTHS),
    ("Optimistic tail (Dmin 5%)", 0.05, ECON_LIMIT_BPM, MAX_LIFE_MONTHS),
    ("Conservative tail (Dmin 10%)", 0.10, ECON_LIMIT_BPM, MAX_LIFE_MONTHS),
    ("Higher economic limit (250 bbl/mo)", DMIN_ANNUAL, 250.0, MAX_LIFE_MONTHS),
    ("30-year well life", DMIN_ANNUAL, ECON_LIMIT_BPM, 360),
    ("Source dataset's QC assumptions", 0.096, 250.0, 360),
]


def main() -> None:
    base = None
    print("| Case | Dmin (nom./yr) | Econ. limit (bbl/mo) | Max life (yr) "
          "| Mean EUR (bbl) | Median EUR (bbl) | vs base |")
    print("|---|---|---|---|---|---|---|")
    for name, dmin, econ, life in CASES:
        ok = run(econ_limit_bpm=econ, dmin_annual=dmin, max_life_months=life, write=False)
        ok = ok[ok["status"] == "ok"]
        mean = ok["eur_bbl"].mean()
        base = base or mean
        print(f"| {name} | {dmin:.1%} | {econ:,.0f} | {life // 12} | {mean:,.0f} "
              f"| {ok['eur_bbl'].median():,.0f} | {mean / base - 1:+.1%} |")


if __name__ == "__main__":
    main()
