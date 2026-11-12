# Archive

`cell_edge_detection.py` is the original single-file version of this project (1077 lines). It is **superseded** by
the split modules at the repo root (`data_loader.py`, `annotator.py`, `dataset.py`, `model.py`, `inference.py`) and is
kept for reference only. Do not import or run it.

Known bugs, fixed in the split modules but not here:
- `refine_edges` draws edges `edge_width + 1` pixels wide when `edge_width > 1`.
- `train_model` raises TypeError when no dataset has been saved yet (first run) or when there are no annotations.
- `load_annotations` matches masks to DIC files by substring, so a mask can be attached to the wrong file.
- `load_tiff_stack` reads 3- and 4-frame TIFF stacks as RGB/RGBA images.
