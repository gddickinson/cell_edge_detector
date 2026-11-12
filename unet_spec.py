"""
TensorFlow-free description of the U-Net built by model.UNetModel.build_unet.

layer_table() lists every layer in the order Keras creates it, with its output
shape and parameter count, so the architecture can be shown (web pages, docs)
without installing TensorFlow. tests/test_unet_spec.py checks it against a
fixture written from the real Keras model (scripts/make_unet_fixture.py) and,
when TensorFlow is installed, against build_unet() directly.
"""

ENCODER_FILTERS = (64, 128, 256, 512)   # conv1..conv4, two 3x3 convs per level
BRIDGE_FILTERS = 1024                   # conv5
KERNEL = 3                              # 3x3 convolutions, padding='same', ReLU
UP_KERNEL = 2                           # 2x2 transposed convolutions, stride 2
DROPOUT_RATE = 0.5                      # after conv4 and conv5
DOWNSAMPLE = 2 ** len(ENCODER_FILTERS)  # input height/width must be divisible by this


def _conv(cin: int, cout: int, k: int) -> int:
    return k * k * cin * cout + cout


def layer_table(input_shape: tuple = (512, 512, 2), n_classes: int = 1) -> list:
    """Layers of build_unet() for an input of shape (height, width, channels).

    Returns a list of dicts: name (Keras default name), kind (layer class),
    block (encoder level, bridge, decoder level or output), shape
    ([h, w, c] output shape) and params.
    """
    h, w, c = (int(v) for v in input_shape)
    if h % DOWNSAMPLE or w % DOWNSAMPLE:
        raise ValueError(
            f"height and width must be divisible by {DOWNSAMPLE} (four 2x2 poolings); got {h}x{w}"
        )
    rows, counts = [], {}

    def add(kind, block, shape, params=0):
        base = {"InputLayer": "input_layer", "Conv2D": "conv2d", "MaxPooling2D": "max_pooling2d",
                "Dropout": "dropout", "Conv2DTranspose": "conv2d_transpose",
                "Concatenate": "concatenate"}[kind]
        n = counts.get(base, 0)
        counts[base] = n + 1
        rows.append({"name": base if n == 0 else f"{base}_{n}", "kind": kind, "block": block,
                     "shape": list(shape), "params": params})

    add("InputLayer", "input", (h, w, c))
    skips = []
    for level, f in enumerate(ENCODER_FILTERS, start=1):
        block = f"encoder {level}"
        add("Conv2D", block, (h, w, f), _conv(c, f, KERNEL))
        add("Conv2D", block, (h, w, f), _conv(f, f, KERNEL))
        c = f
        if level == len(ENCODER_FILTERS):
            add("Dropout", block, (h, w, f))
        skips.append((h, w, f))
        h, w = h // 2, w // 2
        add("MaxPooling2D", block, (h, w, f))
    add("Conv2D", "bridge", (h, w, BRIDGE_FILTERS), _conv(c, BRIDGE_FILTERS, KERNEL))
    add("Conv2D", "bridge", (h, w, BRIDGE_FILTERS), _conv(BRIDGE_FILTERS, BRIDGE_FILTERS, KERNEL))
    add("Dropout", "bridge", (h, w, BRIDGE_FILTERS))
    c = BRIDGE_FILTERS
    for level, (sh, sw, f) in zip(range(6, 10), reversed(skips)):
        block = f"decoder {level - 5}"
        h, w = h * 2, w * 2
        add("Conv2DTranspose", block, (h, w, f), _conv(c, f, UP_KERNEL))
        add("Concatenate", block, (h, w, 2 * f))
        add("Conv2D", block, (h, w, f), _conv(2 * f, f, KERNEL))
        add("Conv2D", block, (h, w, f), _conv(f, f, KERNEL))
        c = f
    add("Conv2D", "output", (h, w, n_classes), _conv(c, n_classes, 1))
    return rows


def summary(input_shape: tuple = (512, 512, 2), n_classes: int = 1) -> dict:
    """Layer table plus totals: parameter count, float32 weight size and layer counts by kind."""
    rows = layer_table(input_shape, n_classes)
    total = sum(r["params"] for r in rows)
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {
        "input_shape": list(input_shape), "output_shape": rows[-1]["shape"], "n_classes": n_classes,
        "total_params": total, "weights_mb_float32": round(total * 4 / 1e6, 1),
        "layer_counts": kinds, "layers": rows,
        "encoder_filters": list(ENCODER_FILTERS), "bridge_filters": BRIDGE_FILTERS,
        "dropout_rate": DROPOUT_RATE, "divisible_by": DOWNSAMPLE,
    }
