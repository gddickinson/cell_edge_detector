# Cell Edge Detector - Interface Map

## Module Structure

### Core Modules (split from `cell_edge_detection.py`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `data_loader.py` | Load and preprocess TIRF/DIC microscopy images | `DataLoader` |
| `annotator.py` | Interactive lasso annotation tool for ground truth masks | `AnnotationTool` |
| `dataset.py` | Dataset preparation, HDF5 storage, data augmentation | `EdgeDetectionDataset` |
| `model.py` | U-Net architecture, training, evaluation | `UNetModel` |
| `inference.py` | Edge detection, refinement, visualization, main app | `EdgeDetector`, `EdgeDetectionApp` |
| `gpu_config.py` | TensorFlow GPU memory configuration | `configure_gpu()` |

### Legacy File

| File | Purpose | Notes |
|------|---------|-------|
| `cell_edge_detection.py` | Original monolithic file (1077 lines) | Retained for reference; use split modules instead |

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

## Data Flow

```
DataLoader --> AnnotationTool --> EdgeDetectionDataset --> UNetModel --> EdgeDetector
    |                                                                        |
    +--- loads TIRF/DIC .tif files                          detects edges ---+
```

## Entry Points

- `python inference.py` -- Main application (interactive)
- `python -m pytest tests/` -- Run test suite
