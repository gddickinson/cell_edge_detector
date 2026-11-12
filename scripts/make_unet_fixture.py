"""Write tests/fixtures/unet_keras_64.json from the real Keras model (needs TensorFlow; never run in CI).

The fixture records every layer of model.UNetModel(input_shape=(64, 64, 2)).build_unet(): name, class, output shape
and parameter count. tests/test_unet_spec.py checks unet_spec.layer_table against it without TensorFlow.

Run: python scripts/make_unet_fixture.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def keras_layers(input_shape=(64, 64, 2)):
    import tensorflow as tf
    from model import UNetModel

    tf.keras.backend.clear_session()  # reset Keras' layer-name counters (conv2d, conv2d_1, ...)
    model = UNetModel(input_shape=input_shape).build_unet()
    rows = [{"name": layer.name, "kind": type(layer).__name__, "shape": list(layer.output.shape[1:]),
             "params": int(layer.count_params())} for layer in model.layers]
    return {"tensorflow": tf.__version__, "input_shape": list(input_shape),
            "total_params": int(model.count_params()), "layers": rows}


if __name__ == "__main__":
    out = ROOT / "tests" / "fixtures" / "unet_keras_64.json"
    out.write_text(json.dumps(keras_layers(), indent=1) + "\n")
    print("wrote", out)
