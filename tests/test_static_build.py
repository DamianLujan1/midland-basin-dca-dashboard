"""Escaping and formatting used by the static build."""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from build_static import script_json  # noqa: E402
from src.report import fmt  # noqa: E402


def test_script_json_cannot_close_script_or_open_comment():
    s = script_json({"x": "</script><!--<script> & {{PAYLOAD}}"})
    assert not re.search(r"[<>&]", s)
    assert json.loads(s)["x"] == "</script><!--<script> & {{PAYLOAD}}"


def test_fmt_handles_missing_values():
    assert fmt(np.nan, ",.0f", " bbl") == "—"
    assert fmt(12345.6, ",.0f", " bbl") == "12,346 bbl"
