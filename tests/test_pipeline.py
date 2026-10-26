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
    mp.setattr(pipeline, "TYPE_CURVE_CSV", str(out / "type_curve.csv"))
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
    capped = results["forecast_months"] == results["forecast_month_cap"]
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
    # GLYNN-ATCHISON 44N fits well apart from a curtailed 600-bbl month, which
    # is read as downtime instead of dragging the fit.
    row = results.set_index("api8").loc[31743755]
    assert row["curtailed_months"] >= 1
    assert row["r_squared_log"] > pipeline.R2_LOW_CONFIDENCE
    assert not row["low_confidence"]


def test_forecast_cap_is_total_life(results):
    life = results["max_life_months"]
    assert (results["forecast_month_cap"] == life - results["months_history"]).all()
    assert (results["months_history"] + results["forecast_months"] <= life).all()


def test_sensitivity_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS", str(tmp_path / "r"))
    monkeypatch.setattr(pipeline, "RESULTS_CSV", str(tmp_path / "r" / "x.csv"))
    monkeypatch.setattr(pipeline, "TYPE_CURVE_CSV", str(tmp_path / "r" / "tc.csv"))
    base = pipeline.run(write=False)
    harsher = pipeline.run(dmin_annual=0.12, econ_limit_bpm=300.0, max_life_months=360,
                           write=False)
    assert not (tmp_path / "r").exists()
    assert (harsher["eur_bbl"] <= base["eur_bbl"] + 1e-6).all()
    assert harsher["dmin_annual"].eq(0.12).all() and harsher["max_life_months"].eq(360).all()


def test_type_curve_ignores_unreported_months_and_stops_when_wells_drop_out(prod):
    from src.type_curve import MIN_SHARE, observed_type_curve
    tc = observed_type_curve(prod)
    assert (tc["well_count"] >= MIN_SHARE * 30).all()
    assert tc["month_on_production"].max() == 33  # 27 of 30 wells through month 33
    assert tc.loc[tc["well_count"] == 30, "month_on_production"].max() == 30
    m31 = prod[prod["month_on_production"] == 31]["oil_bbl"]
    assert tc.set_index("month_on_production").loc[31, "p50_oil_bbl"] == m31.median()


def test_restart_wells_no_longer_get_flat_tails(results):
    r = results.set_index("api8")
    assert (results["b"] <= 1.5 + 1e-9).all()
    # EPLEY-MARTIN 31E: one calendar-time curve through its 2023 dip and rebound
    # gave b 1.84 and 864k bbl at the base-case assumptions. On producing time
    # the dip is downtime and the rebound is the well resuming its trend.
    e = r.loc[31744051]
    assert e["b"] < 1.0 and e["eur_bbl"] < 700_000 and e["curtailed_months"] >= 5
    assert "left out of the fit" in e["qc_flags"]
    # DAVIS-LACY 31K (b 2.0 on the old bound, 776k): still wants b > 1.5.
    k = r.loc[31744055]
    assert k["b_at_cap"] and k["eur_bbl"] < 750_000 and "b held at the 1.5 cap" in k["qc_flags"]


def test_downtime_is_bounded_by_history(results):
    ok = results[results["status"] == "ok"]
    span = ok["months_history"] - (ok["producing_months"] + ok["downtime_months"])
    assert (span >= 0).all()  # producing + downtime months fit inside the record
    assert ok["downtime_months"].between(0, ok["months_history"]).all()


def test_wells_ending_shut_in_or_curtailed_are_flagged(results):
    r = results.set_index("api8")
    for api8 in (31743549, 31744069):  # WELCH-COX E39V, DAVIS-LACY 31D
        assert r.loc[api8, "curtailed_months_at_end"] >= 2
        assert "confirm status" in r.loc[api8, "qc_flags"]
        assert r.loc[api8, "low_confidence"]


def test_clusters_are_named_by_shape_and_agree_with_retention(results):
    steep = results[results["cluster"] == "Steep decline"]
    shallow = results[results["cluster"] == "Shallow decline"]
    assert len(steep) and len(shallow)
    assert steep["retention_12mo"].mean() < shallow["retention_12mo"].mean()
    assert steep["retention_36mo"].mean() < shallow["retention_36mo"].mean()
