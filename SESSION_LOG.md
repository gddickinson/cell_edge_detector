# Session log

## 2026-11-12: web interface and static site (branch `web-interface`)

- Added a FastAPI web app (`run_web.py`, port 8147) and a static GitHub Pages site (`scripts/build_static.py`,
  `.github/workflows/pages.yml`): home, U-Net architecture, training pipeline, edge lab, predict (local app), about.
- No trained weights or microscopy data are committed, so no predictions are shown on the site. The architecture
  table comes from the new `unet_spec.py` (checked against the real Keras model via `tests/fixtures/unet_keras_64.json`,
  written with TensorFlow 2.20 in a scratch venv; 31,031,169 parameters). The edge lab runs a JS port of
  `refine_edges`, `polygon_mask` and the percentile stretch, node-tested against Python on a synthetic fixture.
- `annotator.polygon_mask` factored out of `AnnotationTool.on_select` (same behaviour) so the lasso fill can be tested.
- Bugs fixed (each its own commit, regression test first):
  - `refine_edges` drew edges `edge_width + 1` pixels wide for `edge_width > 1`.
  - `train_model` raised TypeError on the first run (no saved dataset) and when there were no annotations.
  - `load_annotations` attached masks to the wrong DIC file when one file name contained another (`s1_DIC` / `rs1_DIC`).
  - `load_tiff_stack` read 3- and 4-frame TIFF stacks as RGB/RGBA images (now uses tifffile for TIFFs).
- Not changed: the legacy `cell_edge_detection.py` (kept for reference) still has the same four bugs; the root copies of
  `mac_gpu_setup.py` / `tensorflow-metal-setup.py` duplicate `tools/`.
- Tests: 41 passed, 8 skipped without TensorFlow; 48 passed, 1 skipped with TensorFlow 2.20 (scratch venv).

## 2026-11-12: archive legacy file, remove duplicate GPU scripts

- Moved `cell_edge_detection.py` to `_archive/` with a README noting it is superseded and listing its known bugs.
- Removed the root `mac_gpu_setup.py` and `tensorflow-metal-setup.py` (byte-identical to `tools/`; nothing imported them).
- README quick start now imports `EdgeDetectionApp` from `inference`; `tools/tensorflow-metal-setup.py` suggests
  `python inference.py`; INTERFACE.md and ROADMAP.md updated.
