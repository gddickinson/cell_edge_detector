# Cell Edge Detector -- Roadmap

## Current State
A U-Net-based deep learning system for cell edge detection in TIRF/DIC microscopy images. All code lives in a single large file `cell_edge_detection.py` containing `DataLoader`, `UNetModel`, annotation tools, training pipeline, and batch processing. Includes GPU setup scripts (`mac_gpu_setup.py`, `tensorflow-metal-setup.py`). Has `models/` and `test_data/` directories. No tests, no `requirements.txt`.

## Short-term Improvements
- [x] Split `cell_edge_detection.py` into separate modules: `data_loader.py`, `model.py`, `annotator.py`, `dataset.py`, `inference.py`
- [x] Add `requirements.txt` (tensorflow, opencv-python, numpy, matplotlib, scikit-image, h5py)
- [x] Add error handling for missing/corrupt image files in `DataLoader`
- [x] Add type hints and docstrings to all classes and public methods
- [x] Validate that TIRF and DIC images have matching dimensions before processing
- [x] Add logging instead of bare `print()` statements throughout

## Feature Enhancements
- [ ] Add data augmentation options (elastic deformation, color jitter) beyond flips/rotations
- [ ] Implement model evaluation metrics (IoU, Dice score, boundary F1) with visualization
- [ ] Add transfer learning from pre-trained encoders (ResNet, EfficientNet)
- [ ] Support ONNX export for deployment without TensorFlow
- [ ] Add interactive annotation refinement (correct model predictions, retrain)
- [ ] Implement ensemble prediction from multiple trained models

## Long-term Vision
- [ ] Add support for Cellpose and StarDist as alternative backends
- [ ] Implement active learning loop: model predicts, user corrects, model retrains
- [ ] Create a napari plugin for annotation and inference
- [ ] Support 3D segmentation for z-stack microscopy data
- [ ] Package as a pip-installable tool with CLI entry points

## Technical Debt
- [x] Single-file architecture (`cell_edge_detection.py`) is unmaintainable -- needs immediate refactoring
- [x] `mac_gpu_setup.py` and `tensorflow-metal-setup.py` are setup utilities, not project code -- move to `tools/`
- [x] GPU configuration at module import time (top of file) can cause issues when imported as library
- [x] No `.gitignore` -- `models/`, `test_data/`, and `data/` directories likely contain large binaries
- [x] No automated tests -- at minimum add model architecture tests and data pipeline tests
- [x] Hard-coded image shape (768x768) limits flexibility -- make configurable
