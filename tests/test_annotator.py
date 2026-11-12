"""
Tests for AnnotationTool.load_annotations: saved masks are matched to the DIC file they were drawn on.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotator import AnnotationTool  # noqa: E402
from data_loader import DataLoader  # noqa: E402


def test_mask_matched_to_exact_dic_file(tmp_path):
    """A mask for s1_DIC.tif must not be attached to rs1_DIC.tif, whose name merely contains 's1_DIC'."""
    dl = DataLoader(str(tmp_path))
    dl.dic_files = sorted([str(tmp_path / "rs1_DIC.tif"), str(tmp_path / "s1_DIC.tif")])
    tool = AnnotationTool(dl)
    np.save(os.path.join(tool.annotations_dir, "s1_DIC_frame3_mask.npy"), np.ones((4, 4), np.uint8))

    tool.load_annotations()

    assert list(tool.masks) == [f"{tmp_path / 's1_DIC.tif'}_3"]
