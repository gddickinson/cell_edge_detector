"""
Data loading and preprocessing for TIRF and DIC microscopy images.

Handles scanning directories, loading TIFF stacks, and preprocessing
images for visualization and model input.
"""

import os
import glob
import logging

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading, preprocessing, and visualizing TIRF and DIC images.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tirf_files = []
        self.dic_files = []
        self.loaded_tirf_stacks = {}
        self.loaded_dic_stacks = {}

    def scan_directory(self):
        """Scan directory for TIRF and DIC image files."""
        self.tirf_files = sorted(
            glob.glob(os.path.join(self.data_dir, "*piezo*.tif"))
        )
        self.dic_files = sorted(
            glob.glob(os.path.join(self.data_dir, "*DIC*.tif"))
        )

        logger.info(
            "Found %d TIRF files and %d DIC files",
            len(self.tirf_files),
            len(self.dic_files),
        )
        return self.tirf_files, self.dic_files

    def load_tiff_stack(self, file_path: str) -> np.ndarray:
        """Load a TIFF stack as a 3D numpy array (frames, height, width).

        Parameters
        ----------
        file_path : str
            Path to TIFF file.

        Returns
        -------
        np.ndarray or None
            3D array of shape (frames, height, width), or None on error.
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None

        try:
            img_stack = io.imread(file_path)

            if len(img_stack.shape) == 2:
                img_stack = np.expand_dims(img_stack, axis=0)

            return img_stack
        except Exception as e:
            logger.error("Error loading %s: %s", file_path, e)
            return None

    def load_image_pair(self, index: int = 0):
        """Load a corresponding TIRF and DIC image pair by index.

        Parameters
        ----------
        index : int
            Index into the file lists.

        Returns
        -------
        tuple
            (tirf_stack, dic_stack) arrays, or (None, None) on error.
        """
        if index >= len(self.tirf_files) or index >= len(self.dic_files):
            logger.error("Index %d out of range", index)
            return None, None

        tirf_path = self.tirf_files[index]
        dic_path = self.dic_files[index]

        if tirf_path not in self.loaded_tirf_stacks:
            self.loaded_tirf_stacks[tirf_path] = self.load_tiff_stack(tirf_path)
        if dic_path not in self.loaded_dic_stacks:
            self.loaded_dic_stacks[dic_path] = self.load_tiff_stack(dic_path)

        tirf_stack = self.loaded_tirf_stacks[tirf_path]
        dic_stack = self.loaded_dic_stacks[dic_path]

        # Validate matching dimensions
        if tirf_stack is not None and dic_stack is not None:
            if tirf_stack.shape[1:] != dic_stack.shape[1:]:
                logger.warning(
                    "Dimension mismatch: TIRF %s vs DIC %s",
                    tirf_stack.shape,
                    dic_stack.shape,
                )

        return tirf_stack, dic_stack

    def preprocess_for_visualization(
        self, image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99
    ) -> np.ndarray:
        """Preprocess an image for better visualization with contrast stretching.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        percentile_low : float
            Lower percentile for clipping.
        percentile_high : float
            Upper percentile for clipping.

        Returns
        -------
        np.ndarray
            Normalized image in [0, 1] range.
        """
        img = image.astype(np.float32)
        low = np.percentile(img, percentile_low)
        high = np.percentile(img, percentile_high)

        img = np.clip(img, low, high)
        if high - low > 0:
            img = (img - low) / (high - low)
        else:
            img = np.zeros_like(img)

        return img

    def display_image_pair(self, tirf_stack, dic_stack, frame_idx: int = 0):
        """Display a TIRF and DIC image pair side by side."""
        if tirf_stack is None or dic_stack is None:
            logger.warning("No image data to display")
            return

        tirf_img = tirf_stack[frame_idx] if frame_idx < tirf_stack.shape[0] else tirf_stack[0]
        dic_img = dic_stack[frame_idx] if frame_idx < dic_stack.shape[0] else dic_stack[0]

        tirf_vis = self.preprocess_for_visualization(tirf_img)
        dic_vis = self.preprocess_for_visualization(dic_img)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(tirf_vis, cmap='gray')
        axes[0].set_title(f'TIRF Image (Frame {frame_idx})')
        axes[0].axis('off')

        axes[1].imshow(dic_vis, cmap='gray')
        axes[1].set_title(f'DIC Image (Frame {frame_idx})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        return tirf_vis, dic_vis

    def preprocess_for_model(
        self, tirf_img: np.ndarray, dic_img: np.ndarray, target_size: tuple = (512, 512)
    ) -> np.ndarray:
        """Preprocess images for model input.

        Parameters
        ----------
        tirf_img : np.ndarray
            TIRF image.
        dic_img : np.ndarray
            DIC image.
        target_size : tuple
            Target (height, width) for resizing.

        Returns
        -------
        np.ndarray
            Combined 2-channel image of shape (*target_size, 2).
        """
        if tirf_img.shape[:2] != target_size:
            tirf_img = transform.resize(tirf_img, target_size, preserve_range=True)
        if dic_img.shape[:2] != target_size:
            dic_img = transform.resize(dic_img, target_size, preserve_range=True)

        tirf_img = self.preprocess_for_visualization(tirf_img)
        dic_img = self.preprocess_for_visualization(dic_img)

        combined_img = np.stack([tirf_img, dic_img], axis=-1)

        return combined_img
