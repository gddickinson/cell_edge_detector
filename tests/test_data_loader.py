"""
Smoke tests for the DataLoader module.

Tests basic functionality without requiring actual microscopy data.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader


class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        dl = DataLoader("/tmp/test_data")
        assert dl.data_dir == "/tmp/test_data"
        assert dl.tirf_files == []
        assert dl.dic_files == []

    def test_preprocess_for_visualization(self):
        """Test image preprocessing normalizes to [0, 1]."""
        dl = DataLoader("/tmp/test_data")
        image = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)
        result = dl.preprocess_for_visualization(image)

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_for_visualization_constant_image(self):
        """Test preprocessing handles constant images without division by zero."""
        dl = DataLoader("/tmp/test_data")
        image = np.ones((50, 50), dtype=np.float32) * 42.0
        result = dl.preprocess_for_visualization(image)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_preprocess_for_model(self):
        """Test model preprocessing produces correct output shape."""
        dl = DataLoader("/tmp/test_data")
        tirf = np.random.rand(100, 100).astype(np.float32)
        dic = np.random.rand(100, 100).astype(np.float32)

        result = dl.preprocess_for_model(tirf, dic, target_size=(64, 64))

        assert result.shape == (64, 64, 2)
        assert result.dtype == np.float32

    def test_load_tiff_stack_missing_file(self):
        """Test loading a missing file returns None."""
        dl = DataLoader("/tmp/test_data")
        result = dl.load_tiff_stack("/tmp/nonexistent_file.tif")
        assert result is None

    def test_load_image_pair_out_of_range(self):
        """Test loading an out-of-range index returns None, None."""
        dl = DataLoader("/tmp/test_data")
        tirf, dic = dl.load_image_pair(999)
        assert tirf is None
        assert dic is None

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DataLoader(tmpdir)
            tirf_files, dic_files = dl.scan_directory()
            assert len(tirf_files) == 0
            assert len(dic_files) == 0


class TestDataLoaderWithFiles:
    """Tests requiring temporary TIFF files."""

    def test_load_tiff_stack(self):
        """Test loading a synthetic TIFF file."""
        from skimage import io

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a synthetic TIFF stack
            stack = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
            filepath = os.path.join(tmpdir, "test_piezo1.tif")
            io.imsave(filepath, stack)

            dl = DataLoader(tmpdir)
            result = dl.load_tiff_stack(filepath)

            assert result is not None
            assert result.shape == (5, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
