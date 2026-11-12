"""
unet_spec.layer_table must describe exactly the Keras model built by model.UNetModel.build_unet.

The fixture comes from scripts/make_unet_fixture.py (real Keras model, TensorFlow 2.20); the live comparison runs only
when TensorFlow is installed.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unet_spec  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "unet_keras_64.json"


def _strip(rows):
    return [{k: r[k] for k in ("name", "kind", "shape", "params")} for r in rows]


def test_matches_keras_fixture():
    fx = json.loads(FIXTURE.read_text())
    rows = unet_spec.layer_table(tuple(fx["input_shape"]))
    assert _strip(rows) == fx["layers"]
    assert sum(r["params"] for r in rows) == fx["total_params"]


def test_params_do_not_depend_on_image_size():
    assert unet_spec.summary((256, 256, 2))["total_params"] == unet_spec.summary((768, 768, 2))["total_params"]


def test_output_shape_follows_input():
    assert unet_spec.summary((768, 512, 2), n_classes=1)["output_shape"] == [768, 512, 1]


def test_rejects_size_not_divisible_by_16():
    with pytest.raises(ValueError, match="divisible by 16"):
        unet_spec.layer_table((100, 100, 2))


def _tensorflow_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _tensorflow_available(), reason="TensorFlow not installed")
def test_matches_live_keras_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # UNetModel() creates models/ in the working directory
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_unet_fixture
    live = make_unet_fixture.keras_layers((96, 64, 2))
    assert _strip(unet_spec.layer_table((96, 64, 2))) == live["layers"]
