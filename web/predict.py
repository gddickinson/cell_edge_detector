"""Python side of the web app's image endpoints: reading uploads, running a trained U-Net, encoding PNGs.

Everything goes through the repo's own classes: DataLoader (TIFF loading, contrast stretch, model preprocessing),
UNetModel.load_model, EdgeDetector.detect_edges / refine_edges. TensorFlow is optional for the web app: without it,
the prediction endpoint reports that inference needs TensorFlow and a trained model, and the rest still works.
Edge widths and positions are in pixels of the uploaded image; no physical pixel size is used or assumed.
"""
import base64
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_loader import DataLoader  # noqa: E402
from inference import EdgeDetector  # noqa: E402

MODELS_DIR = ROOT / "models"            # where UNetModel.train saves models/unet_training_*/best_model.h5
MODEL_SUFFIXES = (".h5",)              # the format ModelCheckpoint in UNetModel.train writes
TIFF_SUFFIXES = (".tif", ".tiff")
MAX_PIXELS = 4096 * 4096
_loaded = {}


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def list_models(models_dir: Path = None) -> list:
    """Trained model files under models/, newest first, as paths relative to that folder."""
    d = Path(models_dir or MODELS_DIR)
    if not d.is_dir():
        return []
    files = [p for p in d.rglob("*") if p.suffix in MODEL_SUFFIXES and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": str(p.relative_to(d)), "mb": round(p.stat().st_size / 1e6, 1)} for p in files]


def resolve_model(name: str, models_dir: Path = None) -> Path:
    d = Path(models_dir or MODELS_DIR).resolve()
    p = (d / name).resolve()
    if d not in p.parents or p.suffix not in MODEL_SUFFIXES or not p.is_file():
        raise ValueError(f"No trained model named {name!r} under models/.")
    return p


def load_model(path: Path):
    """UNetModel with weights loaded from path (cached per file and modification time)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _loaded:
        from model import UNetModel
        unet = UNetModel.__new__(UNetModel)  # __init__ would create models/ in the working directory
        unet.input_shape, unet.n_classes, unet.model, unet.models_dir = None, 1, None, str(MODELS_DIR)
        unet.load_model(str(path))
        unet.input_shape = tuple(unet.model.input_shape[1:])
        _loaded.clear()
        _loaded[key] = unet
    return _loaded[key]


def read_stack(name: str, data: bytes) -> np.ndarray:
    """(frames, height, width) array from an uploaded TIFF, via DataLoader.load_tiff_stack."""
    if not name.lower().endswith(TIFF_SUFFIXES):
        raise ValueError("Upload TIFF files (.tif/.tiff), as the desktop pipeline expects.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + Path(name).suffix.lower())
        Path(path).write_bytes(data)
        stack = DataLoader(None).load_tiff_stack(path)
    if stack is None:
        raise ValueError(f"Could not read {name} as an image.")
    if stack.ndim != 3:
        raise ValueError(f"{name}: expected a greyscale stack (frames, height, width), got shape {stack.shape}.")
    if stack.shape[1] * stack.shape[2] > MAX_PIXELS:
        raise ValueError(f"{name}: frames larger than {MAX_PIXELS} pixels are not handled by the web app.")
    return stack


def png(img: np.ndarray, gray: bool = True) -> str:
    """Base64 PNG of a [0, 1] greyscale image or an RGB image."""
    buf = io.BytesIO()
    if gray:
        plt.imsave(buf, np.asarray(img, dtype=float), cmap="gray", vmin=0, vmax=1, format="png")
    else:
        plt.imsave(buf, np.clip(img, 0, 1), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview(name: str, data: bytes, frame: int = 0) -> dict:
    """Shape and a contrast-stretched PNG (DataLoader.preprocess_for_visualization) of one frame."""
    stack = read_stack(name, data)
    f = min(max(int(frame), 0), stack.shape[0] - 1)
    vis = DataLoader(None).preprocess_for_visualization(stack[f])
    return {"name": name, "frames": int(stack.shape[0]), "height": int(stack.shape[1]),
            "width": int(stack.shape[2]), "dtype": str(stack.dtype), "frame": f, "png": png(vis)}


def overlay(img: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Red edges on the stretched image, as EdgeDetector.visualize_edges and batch_process draw them."""
    vis = DataLoader(None).preprocess_for_visualization(img)
    rgb = np.stack([vis] * 3, axis=-1)
    rgb[np.asarray(edges) > 0] = [1, 0, 0]
    return rgb


def predict(tirf: tuple, dic: tuple, model_name: str, frame: int = 0, threshold: float = 0.5,
            edge_width: int = 1, models_dir: Path = None) -> dict:
    """Run EdgeDetector on one TIRF/DIC frame pair. tirf and dic are (filename, bytes)."""
    if not tensorflow_available():
        raise RuntimeError("Inference needs TensorFlow (pip install tensorflow) and a trained model in models/.")
    t_stack, d_stack = read_stack(*tirf), read_stack(*dic)
    if t_stack.shape[1:] != d_stack.shape[1:]:
        raise ValueError(f"TIRF frames {t_stack.shape[1:]} and DIC frames {d_stack.shape[1:]} differ in size.")
    f = min(max(int(frame), 0), min(t_stack.shape[0], d_stack.shape[0]) - 1)
    unet = load_model(resolve_model(model_name, models_dir))
    detector = EdgeDetector(unet)
    prediction = detector.detect_edges(t_stack[f], d_stack[f], target_size=unet.input_shape[:2])
    edges = detector.refine_edges(prediction, threshold=threshold, edge_width=edge_width)
    return {
        "model": model_name, "frame": f, "threshold": threshold, "edge_width": edge_width,
        "height": int(prediction.shape[0]), "width": int(prediction.shape[1]),
        "model_input": list(unet.input_shape),
        "fraction_above_threshold": float((prediction > threshold).mean()),
        "edge_pixels": int(np.count_nonzero(edges)),
        "prediction_png": png(np.clip(prediction, 0, 1)),
        "tirf_overlay_png": png(overlay(t_stack[f], edges), gray=False),
        "dic_overlay_png": png(overlay(d_stack[f], edges), gray=False),
    }


def refine(pred: list, height: int, width: int, threshold: float, edge_width: int) -> list:
    """Indices (row-major) of edge pixels from EdgeDetector.refine_edges, for the edge lab's Python check."""
    arr = np.asarray(pred, dtype=np.float64)
    if arr.size != height * width or arr.size > 512 * 512:
        raise ValueError("pred must hold height * width values (at most 512 x 512).")
    edges = EdgeDetector(None).refine_edges(arr.reshape(height, width), threshold, edge_width)
    return np.flatnonzero(np.asarray(edges).ravel() > 0).tolist()
