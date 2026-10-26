"""Tests for production loading and the fitting time axis."""
import pandas as pd
import pytest

from src.data_loader import PROD_COLS, fitting_series, load_production


def _write(tmp_path, rows):
    df = pd.DataFrame(rows)
    for col in PROD_COLS:
        if col not in df:
            df[col] = 0
    path = tmp_path / "prod.csv"
    df.to_csv(path, index=False)
    return str(path)


def _well(api8, oils, gas=None, start=(2024, 1)):
    gas = gas if gas is not None else [o * 2 for o in oils]
    rows = []
    for i, (o, g) in enumerate(zip(oils, gas)):
        y, m = divmod(start[1] - 1 + i, 12)
        rows.append({"api8": api8, "cycle_year_month": (start[0] + y) * 100 + m + 1,
                     "month_on_production": i + 1, "oil_bbl": o, "casinghead_gas_mcf": g})
    return rows


def test_cohort_wide_trailing_zero_months_are_dropped(tmp_path):
    rows = _well(1, [100, 900, 800, 700, 0, 0]) + _well(2, [500, 400, 300, 200, 0, 0])
    prod = load_production(_write(tmp_path, rows))
    assert prod["cycle_year_month"].max() == 202404
    assert len(prod) == 8


def test_single_well_trailing_zeros_are_kept(tmp_path):
    rows = _well(1, [100, 900, 800, 700, 0, 0]) + _well(2, [500, 400, 300, 200, 150, 120])
    prod = load_production(_write(tmp_path, rows))
    assert prod["cycle_year_month"].max() == 202406
    assert (prod[prod["api8"] == 1]["oil_bbl"].tail(2) == 0).all()


def test_month_with_gas_only_counts_as_reported(tmp_path):
    rows = _well(1, [100, 900, 800, 0], gas=[1, 1, 1, 50])
    prod = load_production(_write(tmp_path, rows))
    assert prod["cycle_year_month"].max() == 202404


def test_row_counter_month_on_production_is_rejected(tmp_path):
    rows = _well(1, [100, 900, 800, 700, 600])
    del rows[2]  # a missing calendar month...
    for i, r in enumerate(rows):
        r["month_on_production"] = i + 1  # ...with month_on_production counting rows
    with pytest.raises(ValueError, match="calendar offset"):
        load_production(_write(tmp_path, rows))


def test_fitting_series_starts_at_peak_with_t0():
    prod = pd.DataFrame(_well(1, [100, 900, 800, 700, 600]))
    w = fitting_series(prod, 1)
    assert w["oil_bbl"].iloc[0] == 900
    assert list(w["t"]) == [0, 1, 2, 3]


def test_fitting_series_keeps_calendar_gaps_for_shut_ins():
    prod = pd.DataFrame(_well(1, [100, 900, 800, 0, 0, 600, 500]))
    w = fitting_series(prod, 1)
    assert list(w["oil_bbl"]) == [900, 800, 600, 500]
    assert list(w["t"]) == [0, 1, 4, 5]
