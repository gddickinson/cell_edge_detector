"""
Smoke tests for the inference module.

Tests edge refinement and detection utilities.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


class TestEdgeRefinement:
    """Tests for edge refinement that do not require TensorFlow."""

    @pytest.mark.skipif(not _tensorflow_available(), reason="TensorFlow not installed")
    def test_refine_edges_basic(self):
        """Test basic edge refinement from a synthetic prediction."""
        from inference import EdgeDetector
        from model import UNetModel

        model = UNetModel()

        detector = EdgeDetector(model)

        # Create a synthetic prediction with a circle
        prediction = np.zeros((100, 100), dtype=np.float32)
        y, x = np.ogrid[-50:50, -50:50]
        circle = (x ** 2 + y ** 2 <= 30 ** 2).astype(np.float32)
        prediction = circle

        edges = detector.refine_edges(prediction, threshold=0.5, edge_width=1)

        assert edges.shape == (100, 100)
        assert edges.max() <= 1
        # Should have some edge pixels
        assert edges.sum() > 0
        # Should have fewer edge pixels than filled pixels
        assert edges.sum() < circle.sum()


class TestEdgeDetectionApp:
    """Tests for the EdgeDetectionApp class."""

    @pytest.mark.skipif(not _tensorflow_available(), reason="TensorFlow not installed")
    def test_app_initialization(self, tmp_path):
        """Test app can be initialized with a temp directory."""
        from inference import EdgeDetectionApp

        app = EdgeDetectionApp(data_dir=str(tmp_path))
        app.initialize()

        assert len(app.data_loader.tirf_files) == 0
        assert len(app.data_loader.dic_files) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
