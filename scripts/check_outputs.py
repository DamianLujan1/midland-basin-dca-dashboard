"""Check that committed outputs match a fresh build (used by CI).

Usage:  python scripts/check_outputs.py COMMITTED_DIR FRESH_DIR

Both directories hold results/ and public/. CSVs must have the same columns
and rows, identical text cells and numbers equal to a relative 1e-6: a fresh
build on another machine can differ in the last digits of fitted values
(floating-point differences in the solver), which is not staleness. In
public/index.html everything outside the embedded JSON payload must be
identical, and the payload must match with the same numeric tolerance.
Exits 1 and lists the differences otherwise.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-6, 1e-9
DIRS = ("results", "public")
PAYLOAD = re.compile(r'(<script id="payload" type="application/json">)(.*?)(</script>)', re.S)


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
        if not (xs[~numeric] == ys[~numeric]).all():
            problems.append(f"{col}: text differs in {int((xs[~numeric] != ys[~numeric]).sum())} rows")
        elif not np.allclose(xn[numeric], yn[numeric], rtol=RTOL, atol=ATOL, equal_nan=True):
            worst = float(np.max(np.abs(xn[numeric] - yn[numeric])
                                 / np.maximum(np.abs(yn[numeric]), ATOL)))
            problems.append(f"{col}: numbers differ (worst relative {worst:.2e})")
    return problems


def same_json(a, b, path="payload") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}"]
        return [p for k in a for p in same_json(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [p for i, (u, v) in enumerate(zip(a, b)) for p in same_json(u, v, f"{path}[{i}]")]
    if (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return [] if math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}"]


def compare_html(a: str, b: str) -> list[str]:
    x, y = (open(p, encoding="utf-8").read() for p in (a, b))
    mx, my = PAYLOAD.search(x), PAYLOAD.search(y)
    if not (mx and my):
        return ["payload script not found"]
    if PAYLOAD.sub("", x) != PAYLOAD.sub("", y):
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
