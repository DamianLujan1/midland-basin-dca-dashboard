"""Drive the Streamlit app headless: the PV labels follow the discount slider."""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _labels(at):
    assert not at.exception, at.exception
    metrics = [m.label for m in at.metric]
    detail = list(at.table[0].value.index)
    ranking = [str(c) for c in at.dataframe[0].value.columns]
    return metrics, detail, ranking


def test_pv_labels_follow_the_discount_slider():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    metrics, detail, ranking = _labels(at)
    assert {"PV10 (P50)", "Total PV10 (P50)"} <= set(metrics)
    assert "PV10 P50 ($MM)" in ranking

    at.slider(key="discount").set_value(20.0).run()
    metrics, detail, ranking = _labels(at)
    assert {"PV20 (P50)", "Total PV20 (P50)"} <= set(metrics)
    assert "PV20 P90 / P50 / P10" in detail
    assert {"PV20 P50 ($MM)", "PV20 P90–P10 ($MM)"} <= set(ranking)
    assert not any("PV10" in s for s in metrics + detail + ranking)
    # The P90-P10 caption under the PV metric is found by the same key.
    assert any(c.value.startswith("P90–P10:") and "MM" in c.value for c in at.caption)
