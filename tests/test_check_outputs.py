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
    assert co.same_text("$13.4 MM", "$13.5 MM")
    assert not co.same_text("537,000 bbl", "539,000 bbl")    # two steps
    assert not co.same_text("$0.3 MM", "$0.4 MM")            # one step but 25%
    assert not co.same_text("PV10 (P50)", "PV20 (P50)")
    assert not co.same_text("b held at the 1.2 cap", "b held at the 1.3 cap")
    assert not co.same_text("shut in", "shut-in")


def _typed(values):
    return {"dtype": "f8", "bdata": base64.b64encode(np.asarray(values, "f8").tobytes()).decode()}


def test_chart_arrays_are_compared_as_numbers():
    a = {"name": "Fitted decline", "y": _typed([1000.0, 900.0])}
    assert co.same_json(a, {"name": "Fitted decline", "y": _typed([1000.0 * (1 + 1e-9), 900.0])}) == []
    assert co.same_json(a, {"name": "Fitted decline", "y": _typed([1000.0 * (1 + 1e-4), 900.0])})
    band = {"name": "P90 (low)", "y": _typed([1000.0, 900.0])}
    assert co.same_json(band, {"name": "P90 (low)", "y": _typed([1000.0 * (1 + 1e-4), 900.0])}) == []
    assert co.same_json(band, {"name": "P90 (low)", "y": _typed([1000.0 * (1 + 1e-2), 900.0])})


def _outputs(root, p90=100000.0, ens_b=0.9, eur=500000.0, bdata_y=1000.0):
    os.makedirs(root / "results")
    os.makedirs(root / "public")
    pd.DataFrame({"api8": [1], "eur_bbl": [eur], "eur_p90_bbl": [p90], "qc_flags": ["ok"]}).to_csv(
        root / "results" / "midland_basin_dca_results.csv", index=False)
    pd.DataFrame({"api8": [1], "member": [0], "qi": [1e4], "Di_monthly": [0.1], "b": [ens_b]}).to_csv(
        root / "results" / "forecast_ensemble.csv", index=False)
    payload = {"wells": {"1": {"metrics": {"EUR (P50)": f"{round(eur, -3):,.0f} bbl"},
                               "fig": {"data": [{"name": "Fitted decline", "y": _typed([bdata_y])}]}}}}
    (root / "public" / "index.html").write_text(
        f'<html><script id="payload" type="application/json">{json.dumps(payload)}</script></html>',
        encoding="utf-8")


def test_whole_check_tolerates_noise_and_catches_real_changes(tmp_path):
    _outputs(tmp_path / "a")
    # Noise of the size measured between CPUs, a few times over.
    _outputs(tmp_path / "noise", p90=100000.0 * (1 + 2e-5), ens_b=0.9 * (1 + 2e-4),
             eur=500000.0 * (1 + 1e-6), bdata_y=1000.0 * (1 + 1e-6))
    assert co.main(str(tmp_path / "a"), str(tmp_path / "noise")) == 0
    for name, change in (("eur", {"eur": 500000.0 * (1 + 1e-4)}), ("p90", {"p90": 100400.0}),
                         ("ens", {"ens_b": 0.9 * (1 + 5e-3)}), ("chart", {"bdata_y": 1000.1})):
        _outputs(tmp_path / name, **change)
        assert co.main(str(tmp_path / "a"), str(tmp_path / name)) == 1, name
    shutil.copytree(tmp_path / "a", tmp_path / "extra")
    (tmp_path / "extra" / "public" / "new.csv").write_text("x\n1\n")
    assert co.main(str(tmp_path / "a"), str(tmp_path / "extra")) == 1
