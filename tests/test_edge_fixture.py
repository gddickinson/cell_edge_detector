"""tests/fixtures/edge_fixture.json (used by the JS port's node test) must match what the Python code produces now.

Floats are compared with a tolerance (libm differs in the last digit between macOS and Linux).
Regenerate with: python scripts/make_edge_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_edge_fixture  # noqa: E402


def test_edge_fixture_is_current():
    saved = json.loads(make_edge_fixture.OUT.read_text())
    now = make_edge_fixture.compute()
    for key in ("height", "width", "raw", "polygon", "polygon_mask"):
        assert saved[key] == now[key], key
    assert [list(c) for c in now["cases"]] == saved["cases"]
    np.testing.assert_allclose(saved["stretch"], now["stretch"], rtol=1e-6, atol=1e-7)
    for name in now["maps"]:
        np.testing.assert_allclose(saved["maps"][name], now["maps"][name], rtol=1e-9)
    assert saved["edges"] == now["edges"]
