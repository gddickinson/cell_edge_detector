"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import web.predict as P  # noqa: E402
import web.app as W  # noqa: E402


def _tf():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _tiff(frames=3, h=48, w=64, seed=0):
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.random.default_rng(seed).integers(0, 4000, (frames, h, w)).astype(np.uint16))
    return buf.getvalue()


def test_info_numbers_match_spec():
    import unet_spec
    info = client.get("/api/info").json()
    assert info["total_params"] == unet_spec.summary()["total_params"] == 31031169
    assert info["default_input"] == [512, 512, 2] and info["trained_weights_in_repo"] is False


def test_architecture_sizes():
    a = client.get("/api/architecture/768").json()
    assert a["output_shape"] == [768, 768, 1] and len(a["layers"]) == 34
    assert client.get("/api/architecture/100").status_code == 400


def test_pipeline_defaults_from_code():
    p = client.get("/api/pipeline").json()
    assert p["dataset"]["fractions"] == {"train": 0.7, "val": 0.1, "test": 0.2}
    assert p["train"] == {"batch_size": 4, "epochs": 50}
    assert p["refine"] == {"threshold": 0.5, "edge_width": 1}


def test_hardcoded_facts_still_in_source():
    model_src = (ROOT / "model.py").read_text()
    for s in W.TRAINING_SETTINGS:
        assert s["src"] in model_src, s
    assert "min_lr=1e-6" in model_src and "random_state=42" in (ROOT / "dataset.py").read_text()
    loader = (ROOT / "data_loader.py").read_text()
    p = client.get("/api/pipeline").json()["patterns"]
    assert f'"{p["tirf"]}"' in loader and f'"{p["dic"]}"' in loader
    gen = (ROOT / "dataset.py").read_text()
    for s in ("np.fliplr", "np.flipud", "np.random.randint(0, 4)", "np.random.uniform(0.8, 1.2)"):
        assert s in gen


def test_refine_endpoint_matches_fixture():
    fx = json.loads((ROOT / "tests" / "fixtures" / "edge_fixture.json").read_text())
    r = client.post("/api/refine", json={"height": fx["height"], "width": fx["width"], "pred": fx["maps"]["blobs"],
                                         "threshold": 0.42, "edge_width": 4})
    assert r.status_code == 200 and r.json()["edges"] == fx["edges"]["blobs|0.42|4"]
    assert client.post("/api/refine", json={"height": 3, "width": 3, "pred": [0.0]}).status_code == 400


def test_models_listing(tmp_path):
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "best_model.h5").write_bytes(b"x")
    assert [m["name"] for m in P.list_models(tmp_path)] == ["run1/best_model.h5"]
    with pytest.raises(ValueError):
        P.resolve_model("../secret.h5", tmp_path)
    assert "models" in client.get("/api/models").json()


def test_preview_tiff():
    r = client.post("/api/preview", files={"file": ("cell_DIC.tif", _tiff(), "image/tiff")}, data={"frame": "1"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["frames"], j["height"], j["width"], j["frame"]) == (3, 48, 64, 1) and len(j["png"]) > 100
    bad = client.post("/api/preview", files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 400


@pytest.mark.skipif(_tf(), reason="TensorFlow installed")
def test_predict_without_tensorflow_is_503():
    files = {"tirf": ("a_piezo.tif", _tiff(), "image/tiff"), "dic": ("a_DIC.tif", _tiff(seed=1), "image/tiff")}
    r = client.post("/api/predict", files=files, data={"model": "x.h5"})
    assert r.status_code == 503 and "TensorFlow" in r.text


@pytest.mark.skipif(not _tf(), reason="TensorFlow not installed")
def test_predict_with_untrained_model(tmp_path, monkeypatch):
    """Shape/plumbing check only: an untrained 64x64 network, built (not trained) by build_unet."""
    monkeypatch.chdir(tmp_path)
    from model import UNetModel
    unet = UNetModel(input_shape=(64, 64, 2))
    unet.build_unet()
    unet.model.save(tmp_path / "m.h5")  # the format UNetModel.train writes
    monkeypatch.setattr(P, "MODELS_DIR", tmp_path)
    files = {"tirf": ("a_piezo.tif", _tiff(), "image/tiff"), "dic": ("a_DIC.tif", _tiff(seed=1), "image/tiff")}
    r = client.post("/api/predict", files=files, data={"model": "m.h5", "edge_width": "2"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["height"], j["width"], j["model_input"]) == (48, 64, [64, 64, 2])
    assert j["prediction_png"] and j["tirf_overlay_png"] and j["dic_overlay_png"]
