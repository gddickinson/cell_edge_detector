"""Write tests/fixtures/edge_fixture.json: inputs and outputs of the repo's Python image steps for the JS port.

Covers DataLoader.preprocess_for_visualization (percentile contrast stretch), annotator.polygon_mask (lasso fill) and
EdgeDetector.refine_edges (threshold + erosion, or contours for wider edges). web/static/edge_engine.test.mjs checks
edge_engine.js against it; tests/test_edge_fixture.py checks the fixture is still what the Python code produces.
The inputs are synthetic (seeded noise, a disc like tests/test_inference.py, smooth blobs): no microscopy data.

Run: python scripts/make_edge_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "tests" / "fixtures" / "edge_fixture.json"
H, W = 36, 44
CASES = [(0.5, 1), (0.5, 2), (0.5, 3), (0.3, 1), (0.7, 2), (0.42, 4)]
POLYGON = [[4.3, 3.7], [30.6, 6.2], [38.2, 27.9], [20.4, 32.6], [9.1, 22.3], [15.7, 14.2]]


def inputs():
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 4000, size=(H, W)).astype(np.uint16)
    raw[5:9, 5:9] = 65000  # bright outliers, clipped by the 99th percentile
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((xx - 20.0) ** 2 + (yy - 17.0) ** 2 <= 11.0 ** 2).astype(np.float64)
    blobs = (np.exp(-((xx - 14.3) ** 2 + (yy - 12.1) ** 2) / 60.0)
             + 0.8 * np.exp(-((xx - 30.7) ** 2 + (yy - 22.6) ** 2) / 40.0))
    return raw, {"disc": disc, "blobs": blobs}


def compute():
    from annotator import polygon_mask
    from data_loader import DataLoader
    from inference import EdgeDetector

    raw, maps = inputs()
    det = EdgeDetector(None)
    edges = {}
    for name, pred in maps.items():
        for thr, width in CASES:
            e = det.refine_edges(pred, threshold=thr, edge_width=width)
            edges[f"{name}|{thr}|{width}"] = np.flatnonzero(np.asarray(e).ravel() > 0).tolist()
    return {
        "height": H, "width": W, "cases": CASES, "polygon": POLYGON,
        "raw": raw.ravel().tolist(),
        "stretch": [float(v) for v in DataLoader(None).preprocess_for_visualization(raw).ravel()],
        "maps": {k: [float(v) for v in m.ravel()] for k, m in maps.items()},
        "polygon_mask": np.flatnonzero(polygon_mask(POLYGON, (H, W)).ravel()).tolist(),
        "edges": edges,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(compute(), separators=(",", ":")) + "\n")
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
