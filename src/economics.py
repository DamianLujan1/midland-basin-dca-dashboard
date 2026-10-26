"""Per-well PDP economics: monthly net cash flow, PV, economic limit, shut-in price.

Model, per forecast month with oil rate q (bbl) and gas g = GOR * q (mcf):

    net revenue  = NRI * (1 - ad valorem) * [q * P_oil * (1 - sev_oil) + g * P_gas * (1 - sev_gas)]
    LOE          = fixed + variable * q          (100% working interest)
    cash flow    = net revenue - LOE

Royalty owners bear their share of production and ad valorem taxes, so taxes
are applied to the net revenue interest. Cash flow is positive while
q >= fixed / (net revenue per bbl - variable), which is the economic limit: the
forecast stops there. PV discounts each month at mid-month (PV10 at the base
case's 10%/yr). Capex, workovers and plugging and abandonment (P&A) are
excluded. At ~$150k a well, discounted from each well's economic limit, P&A is
immaterial for the cohort (about 0.5% of total PV10) but not for the
shortest-life wells: about 13% to ~20% of PV10 for those with under 4 years
left, so their PV10 is overstated.

GOR is held at each well's last 12 producing months; Midland Basin GORs usually
rise with time, so gas revenue is if anything understated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

GOR_MONTHS = 12  # trailing producing months the GOR is taken from


@dataclass(frozen=True)
class Economics:
    """Price deck and cost assumptions (flat, nominal). The defaults are the
    base case; see README "Economics" for the reasoning behind each."""
    oil_price: float = 65.0        # $/bbl realized at the lease
    gas_price: float = 2.00        # $/mcf realized wellhead casinghead gas, NGL value net of processing
    nri: float = 0.75              # net revenue interest (25% royalty, 100% WI)
    oil_sev: float = 0.046         # Texas oil production tax, 4.6% of value
    gas_sev: float = 0.075         # Texas gas production tax, 7.5% of value
    ad_valorem: float = 0.025      # property tax, share of revenue
    loe_fixed: float = 6000.0      # $/well-month
    loe_var: float = 4.00          # $/bbl oil (lifting, water handling)
    discount: float = 0.10         # annual rate for PV

    def as_columns(self) -> dict:
        """The assumptions as results-table columns."""
        return {f"econ_{k}": v for k, v in asdict(self).items()}

    @classmethod
    def from_row(cls, row) -> "Economics":
        return cls(**{k: float(row[f"econ_{k}"]) for k in asdict(cls())})


def trailing_gor(oil: pd.Series, gas: pd.Series, months: int = GOR_MONTHS) -> float:
    """Gas-oil ratio (mcf/bbl) over the last `months` months with oil."""
    producing = oil > 0
    o, g = oil[producing].tail(months).sum(), gas[producing].tail(months).sum()
    return float(g / o) if o > 0 else 0.0


def net_revenue_per_bbl(e: Economics, gor: float) -> float:
    """Net revenue to the working interest per barrel of oil, gas included."""
    gross = e.oil_price * (1 - e.oil_sev) + gor * e.gas_price * (1 - e.gas_sev)
    return e.nri * (1 - e.ad_valorem) * gross


def economic_limit(e: Economics, gor: float) -> float:
    """Oil rate (bbl/month) below which monthly cash flow is negative; inf if
    no rate covers the variable cost."""
    margin = net_revenue_per_bbl(e, gor) - e.loe_var
    return e.loe_fixed / margin if margin > 0 else float("inf")


def cash_flow(q: np.ndarray, e: Economics, gor: float) -> np.ndarray:
    """Monthly net cash flow ($) for oil rates q (bbl/month); any shape."""
    q = np.asarray(q, dtype=float)
    return (net_revenue_per_bbl(e, gor) - e.loe_var) * q - e.loe_fixed * (q > 0)


def discount_factors(n: int, rate: float) -> np.ndarray:
    """Mid-month discount factors for months 0..n-1."""
    return (1.0 + rate) ** (-(np.arange(n) + 0.5) / 12.0)


def present_value(ncf: np.ndarray, rate: float) -> np.ndarray:
    """PV of monthly cash flows along the last axis (month 0 = first forecast month)."""
    ncf = np.asarray(ncf, dtype=float)
    return (ncf * discount_factors(ncf.shape[-1], rate)).sum(axis=-1)


def shut_in_price(e: Economics, gor: float, q: float) -> float:
    """Oil price ($/bbl) at which a month at rate q just covers its operating
    cost, gas revenue included: the price below which the well would be shut in
    now. Can be negative when gas revenue alone covers the cost."""
    if not q > 0:
        return float("nan")
    gas = gor * e.gas_price * (1 - e.gas_sev)
    needed = (e.loe_fixed / q + e.loe_var) / (e.nri * (1 - e.ad_valorem))
    return float((needed - gas) / (1 - e.oil_sev))
