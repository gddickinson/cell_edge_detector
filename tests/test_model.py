"""
Smoke tests for the UNet model module.

Tests model architecture construction and basic inference.
Requires TensorFlow to be installed.
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


@pytest.mark.skipif(not _tensorflow_available(), reason="TensorFlow not installed")
class TestUNetModel:
    """Tests for the UNetModel class."""

    def test_build_small_unet(self):
        """Test building a small U-Net model."""
        from model import UNetModel

        # Use small input size for fast testing
        unet = UNetModel(input_shape=(64, 64, 2), n_classes=1)
        model = unet.build_unet()

        assert model is not None
        assert model.input_shape == (None, 64, 64, 2)
        assert model.output_shape == (None, 64, 64, 1)

    def test_predict_shape(self):
        """Test that prediction produces correct output shape."""
        from model import UNetModel

        unet = UNetModel(input_shape=(64, 64, 2), n_classes=1)
        unet.build_unet()

        dummy_input = np.random.rand(1, 64, 64, 2).astype(np.float32)
        prediction = unet.predict(dummy_input)

        assert prediction.shape == (1, 64, 64, 1)
        assert prediction.min() >= 0.0
        assert prediction.max() <= 1.0

    def test_predict_without_model_raises(self):
        """Test that predicting without a model raises ValueError."""
        from model import UNetModel

        unet = UNetModel()
        with pytest.raises(ValueError, match="Model not built"):
            unet.predict(np.zeros((1, 64, 64, 2)))

    def test_dice_coef(self):
        """Test dice coefficient calculation."""
        import tensorflow as tf
        from model import UNetModel

        unet = UNetModel()

        # Perfect prediction should give dice ~1.0
        y_true = tf.constant([[1.0, 1.0, 0.0, 0.0]])
        y_pred = tf.constant([[1.0, 1.0, 0.0, 0.0]])
        dice = unet.dice_coef(y_true, y_pred)

        assert float(dice) > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
