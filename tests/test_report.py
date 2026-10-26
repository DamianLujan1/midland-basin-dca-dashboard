"""Display labels: PV labels follow the discount rate the results were valued at."""
from dataclasses import replace

import pytest

from src import pipeline
from src.arps import B_CAP
from src.report import (RANGE_NOTE, RANKING, cohort_metrics, pv_label, ranking_cells,
                        ranking_kinds, ranking_table, well_detail, well_metrics, well_ranges)


@pytest.mark.parametrize("rate, label", [(0.10, "PV10"), (0.1, "PV10"), (0.2, "PV20"),
                                         (0.05, "PV5"), (0.075, "PV7.5")])
def test_pv_label(rate, label):
    assert pv_label(rate) == label


def test_pv_label_has_no_float_noise_at_any_slider_rate():
    # The app turns the 5-20 %/yr slider into a rate by dividing by 100;
    # 0.07 * 100 is 7.000000000000001.
    for pct in range(5, 21):
        assert pv_label(pct / 100) == f"PV{pct}"


@pytest.fixture(scope="module", params=[0.10, 0.20])
def case(request):
    """(results at this discount rate, expected PV label); no ranges, for speed."""
    results = pipeline.run(economics=replace(pipeline.ECONOMICS, discount=request.param),
                           ranges=False, write=False)
    assert results["econ_discount"].eq(request.param).all()
    return results[results["status"] == "ok"], f"PV{round(request.param * 100)}"


def test_well_labels_follow_the_rate(case):
    ok, pv = case
    row = ok.iloc[0]
    metrics, ranges = well_metrics(row), well_ranges(row)
    # app.py and the static page look ranges up by the metric's key.
    assert set(ranges) <= set(metrics)
    assert f"{pv} (P50)" in metrics and f"{pv} (P50)" in ranges
    assert f"{pv} P90 / P50 / P10" in well_detail(row)
    assert f"Total {pv} (P50)" in cohort_metrics(ok)
    other = "PV20" if pv == "PV10" else "PV10"
    labels = [*metrics, *ranges, *well_detail(row), *cohort_metrics(ok)]
    assert not any(other in k for k in labels)


def test_ranking_columns_follow_the_rate(case):
    ok, pv = case
    table = ranking_table(ok)
    kinds = ranking_kinds(table)
    assert kinds[f"{pv} P50 ($MM)"] == "usdmm"
    assert kinds[f"{pv} P90–P10 ($MM)"] == "pvrange"
    assert [k for _, _, k in RANKING] == list(kinds.values())
    other = "PV20" if pv == "PV10" else "PV10"
    assert not any(other in c or "{" in c for c in table.columns)
    cells = ranking_cells(table)
    assert len(cells) == len(ok) and all(len(r) == len(table.columns) for r in cells)


def test_ranking_kinds_rejects_other_tables(case):
    ok, _ = case
    with pytest.raises(ValueError):
        ranking_kinds(ranking_table(ok).iloc[:, :3])


def test_range_note_names_the_b_cap():
    assert f"held at the {B_CAP:g} cap" in RANGE_NOTE
