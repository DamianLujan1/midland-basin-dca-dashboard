"""End-to-end checks of the pipeline on the shipped Martin County data."""
import os

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.arps import _switch_time, terminal_rate
from src.data_loader import load_production
from src.figures import well_figure


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "RESULTS", str(out))
    mp.setattr(pipeline, "RESULTS_CSV", str(out / "results.csv"))
    res = pipeline.run()
    yield res
    mp.undo()


@pytest.fixture(scope="module")
def prod():
    return load_production(os.path.join(
        pipeline.DATA, "martin_selected_30_monthly_production_normalized.csv"))


def test_all_wells_fit(results):
    assert len(results) == 30
    assert (results["status"] == "ok").all()


def test_history_ends_at_last_reported_month(results, prod):
    # The source pads Apr-Jun 2026 with zeros for every well; none is history.
    assert results["last_reported"].eq("202603").all()
    assert prod["cycle_year_month"].max() == 202603
    assert results["months_history"].between(30, 51).all()


def test_eur_is_cum_plus_remaining(results):
    np.testing.assert_allclose(results["eur_bbl"],
                               results["cum_oil_bbl"] + results["remaining_bbl"])


def test_qi_never_exceeds_observed_peak(results, prod):
    peak = prod.groupby("api8")["oil_bbl"].max()
    assert (results["qi"] <= results["api8"].map(peak) * (1 + 1e-9)).all()


def test_chart_forecast_matches_reported_remaining(results, prod):
    for _, row in results.iterrows():
        fig = well_figure(prod, row)
        fc = [t for t in fig.data if t.name.startswith("Forecast")]
        total = float(np.sum(fc[0].y)) if fc else 0.0
        assert total == pytest.approx(row["remaining_bbl"], rel=1e-9, abs=1e-6)
        assert len(fc[0].y) == row["forecast_months"] if fc else row["forecast_months"] == 0
        if fc:
            assert pd.Timestamp(fc[0].x[0]) == pd.Timestamp("2026-04-01")


def test_forecast_continues_from_recent_rates(results):
    # The forecast's first month should be close to recent production for
    # wells that are producing normally; the rest are flagged.
    f = pipeline.FORECAST_JUMP_FACTOR
    within = results["forecast_vs_recent"].between(1 / f, f)
    assert within.sum() >= 27
    assert results.loc[~within, "low_confidence"].all()
    assert results.loc[~within, "qc_flags"].str.contains("recent rate").all()


def test_partial_month_no_longer_pins_b(results):
    # WELCH-COX E39V: a 4-bbl restart month used to drive b to the 0.05 bound
    # and leave ~2.5k bbl remaining on a well making ~5.3k bbl/mo.
    row = results.set_index("api8").loc[31743549]
    assert row["b"] > 0.5
    assert row["remaining_bbl"] > 100_000


def test_assumptions_and_flags_recorded(results):
    assert (results["econ_limit_bpm"] == pipeline.ECON_LIMIT_BPM).all()
    assert (results["dmin_annual"] == pipeline.DMIN_ANNUAL).all()
    capped = results["forecast_months"] == results["max_forecast_months"]
    assert (results["forecast_capped"] == capped).all()
    assert results.loc[capped, "qc_flags"].str.contains("capped").all()
    assert (results["county"] == "MARTIN").all()
    assert (results["district"] == "08").all()


def test_results_round_trip(results):
    back = pipeline.load_results(pipeline.RESULTS_CSV)
    assert back["district"].eq("08").all()
    assert back["first_prod"].str.len().eq(6).all()
    assert back["qc_flags"].map(type).eq(str).all()


def test_fitted_curves_continuous_at_terminal_switch(results):
    for _, row in results.iterrows():
        args = (row["qi"], row["Di_monthly"], row["b"], row["dmin_annual"] / 12)
        t_star = _switch_time(*args)
        q = terminal_rate(np.array([t_star - 1e-6, t_star + 1e-6]), *args)
        assert q[0] == pytest.approx(q[1], rel=1e-6)


def test_single_partial_month_does_not_make_a_well_low_confidence(results):
    # GLYNN-ATCHISON 44N fits well apart from one curtailed 600-bbl month.
    row = results.set_index("api8").loc[31743755]
    assert row["n_partial_months"] >= 1
    assert row["r_squared_log"] > pipeline.R2_LOW_CONFIDENCE
    assert not row["low_confidence"]
