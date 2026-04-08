"""
GPU configuration for TensorFlow.

Configures GPU memory growth to avoid memory allocation issues.
Should be imported before any TensorFlow model operations.
"""

import logging

logger = logging.getLogger(__name__)


def configure_gpu():
    """Configure GPU memory growth for TensorFlow."""
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("GPU detected and configured: %s", physical_devices)
        except Exception as e:
            logger.warning("Error configuring GPU: %s", e)
    else:
        logger.info("No GPU detected. Using CPU.")
