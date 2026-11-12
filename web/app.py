"""FastAPI web app for Cell Edge Detector: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8147, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The POST
routes (TIFF preview, U-Net prediction, Python edge refinement) and GET /api/models only exist in the local app;
on the static site static_api.js answers them or the page shows a "needs the local app" notice.
"""
import inspect
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web import predict as P  # noqa: I001  (puts the repo root on sys.path first)
import unet_spec  # noqa: E402
from data_loader import DataLoader  # noqa: E402
from dataset import EdgeDetectionDataset  # noqa: E402
from inference import EdgeDetectionApp, EdgeDetector  # noqa: E402
from model import UNetModel  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
SIZES = (256, 512, 768)

# Hard-coded settings in model.py (build_unet / train). tests/test_web.py checks each snippet is still in model.py.
TRAINING_SETTINGS = [
    {"what": "Optimiser", "value": "Adam, learning rate 1e-4", "src": "learning_rate=1e-4"},
    {"what": "Loss", "value": "Dice loss = 1 - Dice coefficient (smooth = 1)", "src": "return 1 - self.dice_coef"},
    {"what": "Checkpoint", "value": "best_model.h5, best validation Dice", "src": "monitor='val_dice_coef'"},
    {"what": "Learning-rate schedule", "value": "halve on validation-loss plateau of 5 epochs, floor 1e-6",
     "src": "factor=0.5, patience=5"},
    {"what": "Early stopping", "value": "15 epochs without validation-loss improvement, best weights restored",
     "src": "patience=15"},
    {"what": "Logs", "value": "TensorBoard logs, training_history.pkl and training_history.png", "src": "TensorBoard"},
]
AUGMENTATION = [  # dataset.py get_data_generator, training batches only
    "left-right flip (p = 0.5)", "up-down flip (p = 0.5)", "rotation by 0, 90, 180 or 270 degrees",
    "DIC channel brightness x U(0.8, 1.2), clipped to [0, 1] (p = 0.5)",
]

app = FastAPI(title="Cell Edge Detector")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _defaults(fn) -> dict:
    return {k: v.default for k, v in inspect.signature(fn).parameters.items()
            if v.default is not inspect.Parameter.empty and isinstance(v.default, (int, float, tuple, type(None)))}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary and headline numbers (all computed from the repo's code)."""
    s = unet_spec.summary(UNetModel.__init__.__defaults__[0])
    return {
        "title": "Cell Edge Detector", "repo": "cell_edge_detector",
        "description": "A U-Net that detects cell edges in paired TIRF (total internal reflection fluorescence) and "
                       "DIC (differential interference contrast) microscopy images: annotate cells, train, predict.",
        "default_input": s["input_shape"], "total_params": s["total_params"],
        "weights_mb_float32": s["weights_mb_float32"], "conv_layers": s["layer_counts"]["Conv2D"]
        + s["layer_counts"]["Conv2DTranspose"], "levels": len(s["encoder_filters"]) + 1,
        "trained_weights_in_repo": False, "sizes": list(SIZES),
    }


@app.get("/api/architecture/{size}")
def architecture(size: int):
    """unet_spec.summary for a square size x size x 2 input (TIRF + DIC channels)."""
    try:
        return unet_spec.summary((size, size, 2), 1)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/pipeline")
def pipeline():
    """Defaults read from the repo's function signatures, plus the settings hard-coded in model.py."""
    prep = _defaults(EdgeDetectionDataset.prepare_dataset)
    test, val = prep["test_split"], prep["val_split"]
    return {
        "dataset": {**prep, "target_size": list(_defaults(EdgeDetectionDataset.__init__)["target_size"]),
                    "fractions": {"train": round(1 - test - val, 6), "val": val, "test": test},
                    "random_state": 42},
        "train": _defaults(EdgeDetectionApp.train_model),
        "stretch": _defaults(DataLoader.preprocess_for_visualization),
        "refine": _defaults(EdgeDetector.refine_edges),
        "patterns": {"tirf": "*piezo*.tif", "dic": "*DIC*.tif"},
        "training_settings": [{k: r[k] for k in ("what", "value")} for r in TRAINING_SETTINGS],
        "augmentation": AUGMENTATION,
    }


@app.get("/api/models")
def models():
    """Trained models found under models/ in this checkout (local app only)."""
    return {"tensorflow": P.tensorflow_available(), "models": P.list_models(), "folder": "models/"}


async def _read(f: UploadFile) -> tuple:
    data = await f.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large for the web app (200 MB); use the Python API for big stacks.")
    return f.filename or "upload.tif", data


@app.post("/api/preview")
async def preview(file: UploadFile = File(...), frame: int = Form(0)):
    try:
        return P.preview(*(await _read(file)), frame)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/predict")
async def predict(tirf: UploadFile = File(...), dic: UploadFile = File(...), model: str = Form(...),
                  frame: int = Form(0), threshold: float = Form(0.5), edge_width: int = Form(1)):
    if not 0 < threshold < 1 or not 1 <= edge_width <= 15:
        raise HTTPException(400, "threshold must be in (0, 1) and edge_width in 1..15.")
    try:
        return P.predict(await _read(tirf), await _read(dic), model, frame, threshold, edge_width)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


class RefineBody(BaseModel):
    height: int
    width: int
    pred: list[float]
    threshold: float = 0.5
    edge_width: int = 1


@app.post("/api/refine")
def refine(body: RefineBody):
    """EdgeDetector.refine_edges in Python, so the edge lab can show that the JS port agrees."""
    if not 1 <= body.edge_width <= 15:
        raise HTTPException(400, "edge_width must be in 1..15.")
    try:
        return {"edges": P.refine(body.pred, body.height, body.width, body.threshold, body.edge_width)}
    except ValueError as e:
        raise HTTPException(400, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/pipeline"] + [f"/architecture/{s}" for s in SIZES]
