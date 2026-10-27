"""The CI check that committed outputs match a fresh build."""
import base64
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import check_outputs as co  # noqa: E402


def test_displayed_numbers_may_differ_by_one_rounding_step():
    assert co.same_text("537,000–764,000 bbl", "538,000–764,000 bbl")
    assert co.same_text("$3.9–8.4", "$3.9–8.5")   # small values flip by more than 1%
    assert co.same_text("999,000 bbl", "1,000,000 bbl")
    assert not co.same_text("537,000 bbl", "539,000 bbl")    # two steps
    assert not co.same_text("PV10 (P50)", "PV20 (P50)")
    assert not co.same_text("shut in", "shut-in")


def test_a_change_of_format_is_caught():
    assert not co.same_text("R² 0.965", "R² 0.97")            # three decimals to two
    assert not co.same_text("537,000 bbl", "537,400 bbl")     # thousands to units
    assert co.same_text("999 Mbbl", "1,000 Mbbl")             # a flip, not a format change


def _typed(values):
    return {"dtype": "f8", "bdata": base64.b64encode(np.asarray(values, "f8").tobytes()).decode()}


def test_chart_arrays_are_compared_as_numbers():
    a = {"name": "Fitted decline", "y": _typed([1000.0, 900.0])}
    assert co.same_json(a, {"name": "Fitted decline", "y": _typed([1000.0 * (1 + 1e-9), 900.0])}) == []
    assert co.same_json(a, {"name": "Fitted decline", "y": _typed([1000.0 * (1 + 1e-4), 900.0])})
    band = {"name": "P90 (low)", "y": _typed([1000.0, 900.0])}
    assert co.same_json(band, {"name": "P90 (low)", "y": _typed([1000.0 * (1 + 1e-4), 900.0])}) == []
    assert co.same_json(band, {"name": "P90 (low)", "y": _typed([1000.0 * (1 + 1e-2), 900.0])})


def _outputs(root, p90=100000.0, ens_b=0.9, eur=500000.0, bdata_y=1000.0, months=300,
             forecast=(500.0, 300.0, 140.0)):
    os.makedirs(root / "results")
    os.makedirs(root / "public")
    pd.DataFrame({"api8": [1], "eur_bbl": [eur], "eur_p90_bbl": [p90], "qc_flags": ["ok"],
                  "forecast_months": [months], "econ_limit_bpm": [130.0]}).to_csv(
        root / "results" / "midland_basin_dca_results.csv", index=False)
    pd.DataFrame({"api8": [1], "member": [0], "qi": [1e4], "Di_monthly": [0.1], "b": [ens_b]}).to_csv(
        root / "results" / "forecast_ensemble.csv", index=False)
    payload = {"wells": {"1": {"metrics": {"EUR (P50)": f"{round(eur, -3):,.0f} bbl"},
                               "fig": {"data": [
                                   {"name": "Fitted decline", "y": _typed([bdata_y])},
                                   {"name": "Forecast to 130 bbl/mo limit",
                                    "x": [f"2026-{m + 4:02d}-01" for m in range(len(forecast))],
                                    "y": _typed(forecast)}]}}}}
    (root / "public" / "index.html").write_text(
        f'<html><script id="payload" type="application/json">{json.dumps(payload)}</script></html>',
        encoding="utf-8")


def test_whole_check_tolerates_noise_and_catches_real_changes(tmp_path):
    _outputs(tmp_path / "a")
    # Noise of the size measured between CPUs, a few times over.
    _outputs(tmp_path / "noise", p90=100000.0 * (1 + 2e-5), ens_b=0.9 * (1 + 2e-4),
             eur=500000.0 * (1 + 1e-6), bdata_y=1000.0 * (1 + 1e-6))
    assert co.main(str(tmp_path / "a"), str(tmp_path / "noise")) == 0
    for name, change in (("eur", {"eur": 500000.0 * (1 + 1e-3)}), ("p90", {"p90": 100400.0}),
                         ("ens", {"ens_b": 0.9 * (1 + 5e-3)}), ("chart", {"bdata_y": 1000.1})):
        _outputs(tmp_path / name, **change)
        assert co.main(str(tmp_path / "a"), str(tmp_path / name)) == 1, name
    # A forecast month within CPU noise of the 130 bbl/mo limit: one month and up to
    # one limit-month of volume more on one machine.
    _outputs(tmp_path / "month", eur=500000.0 + 129.99, months=301,
             forecast=(500.0, 300.0, 140.0, 130.0))
    assert co.main(str(tmp_path / "a"), str(tmp_path / "month")) == 0
    _outputs(tmp_path / "months2", months=302)
    assert co.main(str(tmp_path / "a"), str(tmp_path / "months2")) == 1
    _outputs(tmp_path / "longer", forecast=(500.0, 300.0, 140.0, 130.0, 129.0))
    assert co.main(str(tmp_path / "a"), str(tmp_path / "longer")) == 1
    shutil.copytree(tmp_path / "a", tmp_path / "extra")
    (tmp_path / "extra" / "public" / "new.csv").write_text("x\n1\n")
    assert co.main(str(tmp_path / "a"), str(tmp_path / "extra")) == 1
