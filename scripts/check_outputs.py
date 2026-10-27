"""Check that committed outputs match a fresh build (used by CI).

Usage:  python scripts/check_outputs.py COMMITTED_DIR FRESH_DIR

Both directories hold results/ and public/. A fresh build on another machine
can differ in the last digits (its CPU's vector math and BLAS round
differently), which is not staleness, so:

- numbers must agree to a relative RTOL; values computed from the bootstrap
  ensemble (results/forecast_ensemble.csv, the P90/P10 columns, the chart's
  P90-P10 band) to RTOL_ENSEMBLE, since some of its 1,000 refits per well sit
  on flat stretches of the fit's cost. A rebuild with the vector math and BLAS
  of a typical CI runner (AVX2, no AVX-512) differed from one on an AVX-512
  machine by at most 3e-7 in the deterministic values, 2e-5 in the P90/P10
  values and band and 8e-5 in the ensemble members; a real change -- e.g.
  LEVEL_SIGMA 0.23 -> 0.231 -- moves P90/P10 by ~4e-3;
- a forecast month whose rate lands within that noise of the economic limit
  can be in the forecast on one machine and not on another, so forecast
  lengths may differ by one month: volumes by up to the well's limit (one
  month at the limit), month counts by one, and the chart's forecast and band
  traces by one point (a month at the limit has ~zero cash flow, so PV needs
  no allowance);
- Plotly's base64 chart arrays are decoded and compared as numbers;
- text (CSV text cells, the page outside its chart data, the formatted values
  inside it) must match except that a displayed number may differ by one step
  of its rounding (537,000 vs 538,000 bbl; $8.4 vs $8.5 MM) in the same
  format: a value sitting on a rounding boundary can flip, and the numbers
  behind it are checked above. A change of format (decimals, rounding) is
  caught; a one-step edit of a number in fixed prose is not.

Exits 1 and lists the differences otherwise.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-5, 1e-9
RTOL_ENSEMBLE = 1e-3
DIRS = ("results", "public")
PAYLOAD = re.compile(r'(<script id="payload" type="application/json">)(.*?)(</script>)', re.S)
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
ENSEMBLE_FILES = ("forecast_ensemble.csv",)
ENSEMBLE_TRACES = ("P90 (low)", "P90–P10 range")
FORECAST_TRACES = ("Forecast",) + ENSEMBLE_TRACES   # may differ by one month at the limit
LIMIT_VOLUMES = ("remaining_bbl", "eur_bbl", "remaining_p90_bbl", "remaining_p10_bbl",
                 "eur_p90_bbl", "eur_p10_bbl")


def rtol_for_column(file: str, column: str) -> float:
    if os.path.basename(file) in ENSEMBLE_FILES or "_p90_" in column or "_p10_" in column:
        return RTOL_ENSEMBLE
    return RTOL


def atol_for_column(column: str, fresh: pd.DataFrame):
    """Absolute tolerance: one forecast month at the economic limit (see top)."""
    if "econ_limit_bpm" not in fresh:
        return ATOL
    limit = pd.to_numeric(fresh["econ_limit_bpm"], errors="coerce").fillna(0.0)
    if column == "forecast_months":
        return 1.0
    if column in LIMIT_VOLUMES:
        return limit + ATOL
    if column == "eur_per_1000ft" and "lateral_ft" in fresh:
        return limit * 1000.0 / pd.to_numeric(fresh["lateral_ft"], errors="coerce") + ATOL
    return ATOL


def _step(token: str) -> float:
    """One step of the rounding a displayed number shows: 0.01 for '0.70',
    1,000 for '537,000' (volumes are shown to the nearest thousand barrels)."""
    if "." in token:
        return 10.0 ** -len(token.split(".")[1])
    return 1000.0 if token.endswith(",000") else 1.0


def _shape(token) -> str:
    """A displayed number's format: '#' plus its decimals, or ',000' when it is
    rounded to thousands."""
    token = token.group(0) if hasattr(token, "group") else token
    if "." in token:
        return "#." + "#" * len(token.split(".")[1])
    return "#,000" if token.endswith(",000") else "#"


def same_text(a: str, b: str) -> bool:
    """Equal, or equal but for displayed numbers in the same format one
    rounding step apart."""
    if a == b:
        return True
    if NUMBER.sub(_shape, a) != NUMBER.sub(_shape, b):
        return False
    for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
        u, v = float(x.replace(",", "")), float(y.replace(",", ""))
        if x != y and abs(u - v) > 1.0001 * _step(x):
            return False
    return True


def files(root: str) -> set[str]:
    out = set()
    for d in DIRS:
        for name in os.listdir(os.path.join(root, d)):
            if not name.startswith("."):
                out.add(os.path.join(d, name))
    return out


def compare_csv(a: str, b: str) -> list[str]:
    x, y = pd.read_csv(a, dtype=str), pd.read_csv(b, dtype=str)
    if list(x.columns) != list(y.columns):
        return [f"columns differ: {sorted(set(x.columns) ^ set(y.columns))}"]
    if len(x) != len(y):
        return [f"{len(x)} rows committed vs {len(y)} fresh"]
    problems = []
    for col in x.columns:
        xs, ys = x[col].fillna(""), y[col].fillna("")
        xn, yn = pd.to_numeric(xs, errors="coerce"), pd.to_numeric(ys, errors="coerce")
        numeric = xn.notna() & yn.notna()
        rtol, atol = rtol_for_column(a, col), atol_for_column(col, y)
        atol = atol[numeric] if isinstance(atol, pd.Series) else atol
        text_diff = sum(not same_text(u, v) for u, v in zip(xs[~numeric], ys[~numeric]))
        if text_diff:
            problems.append(f"{col}: text differs in {text_diff} rows")
        elif not np.isclose(xn[numeric], yn[numeric], rtol=rtol, atol=atol,
                            equal_nan=True).all():
            worst = float(np.max(np.abs(xn[numeric] - yn[numeric])
                                 / np.maximum(np.abs(yn[numeric]), ATOL)))
            problems.append(f"{col}: numbers differ (worst relative {worst:.2e}, "
                            f"tolerance {rtol:g})")
    return problems


def typed_array(x) -> np.ndarray | None:
    """A Plotly typed array ({"dtype": "f8", "bdata": base64, "shape": ...}) as numbers."""
    if not (isinstance(x, dict) and "bdata" in x and "dtype" in x):
        return None
    arr = np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"]))
    return arr.reshape([int(n) for n in str(x["shape"]).split(",")]) if "shape" in x else arr


def same_json(a, b, path="payload", rtol=RTOL, ragged=False) -> list[str]:
    name = str(a.get("name", "")) if isinstance(a, dict) else ""
    if name.startswith(ENSEMBLE_TRACES):
        rtol = RTOL_ENSEMBLE  # the chart's P90-P10 band
    if name.startswith(FORECAST_TRACES) and isinstance(b, dict):
        ragged = True  # may be one month longer or shorter, see top
        if name != str(b.get("name", "")) and same_text(name, str(b.get("name", ""))):
            b = {**b, "name": name}  # "Forecast to 128 bbl/mo limit" on a rounding edge
    xa, xb = typed_array(a), typed_array(b)
    if xa is not None or xb is not None:
        if xa is None or xb is None or xa.dtype != xb.dtype or xa.ndim != xb.ndim:
            return [f"{path}: arrays differ in type or shape"]
        if xa.shape != xb.shape:
            if not (ragged and xa.ndim == 1 and abs(len(xa) - len(xb)) == 1):
                return [f"{path}: arrays differ in type or shape"]
            n = min(len(xa), len(xb))
            xa, xb = xa[:n], xb[:n]
        return [] if np.allclose(xa, xb, rtol=rtol, atol=ATOL, equal_nan=True) else [
            f"{path}: numbers differ (tolerance {rtol:g})"]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}"]
        return [p for k in a for p in same_json(a[k], b[k], f"{path}.{k}", rtol, ragged)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b) and not (ragged and abs(len(a) - len(b)) == 1):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [p for i, (u, v) in enumerate(zip(a, b))
                for p in same_json(u, v, f"{path}[{i}]", rtol)]
    if (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return [] if math.isclose(a, b, rel_tol=rtol, abs_tol=ATOL) else [f"{path}: {a} vs {b}"]
    if isinstance(a, str) and isinstance(b, str):
        return [] if same_text(a, b) else [f"{path}: {a[:60]!r} vs {b[:60]!r}"]
    return [] if a == b else [f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}"]


def compare_html(a: str, b: str) -> list[str]:
    x, y = (open(p, encoding="utf-8").read() for p in (a, b))
    mx, my = PAYLOAD.search(x), PAYLOAD.search(y)
    if not (mx and my):
        return ["payload script not found"]
    if not same_text(PAYLOAD.sub("", x), PAYLOAD.sub("", y)):
        return ["page text outside the chart payload differs"]
    return same_json(json.loads(mx.group(2)), json.loads(my.group(2)))[:10]


def main(committed: str, fresh: str) -> int:
    problems = []
    names_c, names_f = files(committed), files(fresh)
    for n in sorted(names_c ^ names_f):
        problems.append(f"{n}: only in {'committed' if n in names_c else 'fresh'} outputs")
    for n in sorted(names_c & names_f):
        a, b = os.path.join(committed, n), os.path.join(fresh, n)
        diff = (compare_csv(a, b) if n.endswith(".csv") else
                compare_html(a, b) if n.endswith(".html") else
                ([] if open(a, "rb").read() == open(b, "rb").read() else ["bytes differ"]))
        problems += [f"{n}: {d}" for d in diff]
    if problems:
        print("Committed outputs are stale; rerun `python scripts/build_static.py` and commit:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"Committed outputs match a fresh build ({len(names_c)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
