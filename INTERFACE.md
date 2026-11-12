# Cell Edge Detector - Interface Map

## Module Structure

### Core Modules (split from the legacy `_archive/cell_edge_detection.py`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `data_loader.py` | Load and preprocess TIRF/DIC microscopy images | `DataLoader` |
| `annotator.py` | Interactive lasso annotation tool for ground truth masks | `AnnotationTool`, `polygon_mask()` |
| `dataset.py` | Dataset preparation, HDF5 storage, data augmentation | `EdgeDetectionDataset` |
| `model.py` | U-Net architecture, training, evaluation | `UNetModel` |
| `inference.py` | Edge detection, refinement, visualization, main app | `EdgeDetector`, `EdgeDetectionApp` |
| `gpu_config.py` | TensorFlow GPU memory configuration | `configure_gpu()` |
| `unet_spec.py` | TensorFlow-free description of the U-Net in `model.py` (layer table, shapes, parameter counts); tested against Keras | `layer_table()`, `summary()` |

### Legacy File

| File | Purpose | Notes |
|------|---------|-------|
| `_archive/cell_edge_detection.py` | Original monolithic file (1077 lines) | Superseded, has known bugs (see `_archive/README.md`); do not use |

### Tools (setup utilities, not core project code)

| File | Purpose |
|------|---------|
| `tools/mac_gpu_setup.py` | Apple Silicon Metal GPU configuration |
| `tools/tensorflow-metal-setup.py` | TensorFlow Metal backend setup |

### Tests

| File | Purpose |
|------|---------|
| `tests/test_data_loader.py` | DataLoader smoke tests |
| `tests/test_model.py` | UNet model architecture tests |
| `tests/test_inference.py` | Edge refinement and app tests |
| `tests/test_annotator.py` | Mask-to-DIC-file matching on reload |
| `tests/test_unet_spec.py` | `unet_spec` vs the Keras model (fixture + live when TensorFlow is installed) |
| `tests/test_edge_fixture.py` | `tests/fixtures/edge_fixture.json` still matches the Python code |
| `tests/test_web.py` | Web API, static build, node test of the JS engine |

### Web interface (`web/`, see `web/__init__.py`)

| File | Purpose |
|------|---------|
| `run_web.py` | Starts the local app on http://127.0.0.1:8147 |
| `web/app.py` | FastAPI: `GET /api/info`, `/api/pipeline`, `/api/architecture/{size}`, `/api/models`; `POST /api/preview`, `/api/predict`, `/api/refine`; `STATIC_ENDPOINTS` |
| `web/predict.py` | Upload reading, trained-model loading (`models/**/*.h5`), PNG encoding around `DataLoader` / `EdgeDetector` |
| `web/static/` | Front end: `core.js` (router, shared kit), `main.js`, one module per page (`home`, `architecture` + `unet_diagram`, `pipeline`, `edges` + `edge_sources`, `predict`, `about`), `static_api.js` (static site) |
| `web/static/edge_engine.js` | JS port of `refine_edges`, `polygon_mask`, `preprocess_for_visualization`; `edge_engine.test.mjs` checks it against the fixture |
| `scripts/build_static.py` | Exports the static site (GitHub Pages, `.github/workflows/pages.yml`) |
| `scripts/screenshots.py` | Smoke test in Chrome + README screenshots (`docs/img/`) |
| `scripts/make_edge_fixture.py`, `scripts/make_unet_fixture.py` | Write the test fixtures (the U-Net one needs TensorFlow) |

## Data Flow

```
DataLoader --> AnnotationTool --> EdgeDetectionDataset --> UNetModel --> EdgeDetector
    |                                                                        |
    +--- loads TIRF/DIC .tif files                          detects edges ---+
```

## Entry Points

- `python inference.py` -- Main application (interactive)
- `python -m pytest tests/` -- Run test suite
- `python run_web.py` -- Web app (port 8147)
