"""EUR and PV10 sensitivity to the forecast and economic assumptions and the b
cap (prints a Markdown table).

Prices, costs, Dmin and the well life change only the forecast tail and value;
the b cap changes the fits of the wells it binds. P90/P10 ranges are skipped
(ranges=False) to keep this quick.

Run:  python scripts/sensitivity.py   (from the repo root)
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DMIN_ANNUAL, ECONOMICS, MAX_LIFE_MONTHS, run  # noqa: E402

E = ECONOMICS
CASES = [
    # name, Dmin, life (months), economics, fixed econ limit, b cap
    ("Base case", DMIN_ANNUAL, MAX_LIFE_MONTHS, E, None, None),
    ("Oil $50/bbl", DMIN_ANNUAL, MAX_LIFE_MONTHS, replace(E, oil_price=50.0), None, None),
    ("Oil $80/bbl", DMIN_ANNUAL, MAX_LIFE_MONTHS, replace(E, oil_price=80.0), None, None),
    ("Gas $0/mcf", DMIN_ANNUAL, MAX_LIFE_MONTHS, replace(E, gas_price=0.0), None, None),
    ("Fixed LOE +50% ($9,000/mo)", DMIN_ANNUAL, MAX_LIFE_MONTHS, replace(E, loe_fixed=9000.0),
     None, None),
    ("Flatter tail (Dmin 5%)", 0.05, MAX_LIFE_MONTHS, E, None, None),
    ("Steeper tail (Dmin 10%)", 0.10, MAX_LIFE_MONTHS, E, None, None),
    ("30-year well life", DMIN_ANNUAL, 360, E, None, None),
    ("Source dataset's QC assumptions (fixed 250 bbl/mo)", 0.096, 360, E, 250.0, None),
    ("b cap 1.0", DMIN_ANNUAL, MAX_LIFE_MONTHS, E, None, 1.0),
    ("b cap 1.5 (previous)", DMIN_ANNUAL, MAX_LIFE_MONTHS, E, None, 1.5),
    ("b cap 2.0 (first version's bound)", DMIN_ANNUAL, MAX_LIFE_MONTHS, E, None, 2.0),
    ("No b cap (b ≤ 5)", DMIN_ANNUAL, MAX_LIFE_MONTHS, E, None, 5.0),
]


def main() -> None:
    base = None
    print("| Case | Median econ. limit (bbl/mo) | Mean EUR (bbl) | Median EUR (bbl) "
          "| EUR vs base | Total PV10 ($MM) | PV10 vs base |")
    print("|---|---|---|---|---|---|---|")
    for name, dmin, life, econ, limit, b_cap in CASES:
        kw = {} if b_cap is None else {"b_cap": b_cap}
        ok = run(dmin_annual=dmin, max_life_months=life, economics=econ,
                 econ_limit_bpm=limit, ranges=False, write=False, **kw)
        ok = ok[ok["status"] == "ok"]
        mean, pv = ok["eur_bbl"].mean(), ok["pv10_usd"].sum()
        base = base or (mean, pv)
        print(f"| {name} | {ok['econ_limit_bpm'].median():,.0f} | {round(mean, -3):,.0f} "
              f"| {round(ok['eur_bbl'].median(), -3):,.0f} | {mean / base[0] - 1:+.1%} "
              f"| {pv / 1e6:,.1f} | {pv / base[1] - 1:+.1%} |")


if __name__ == "__main__":
    main()
