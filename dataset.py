"""
Dataset preparation and management for training the U-Net model.

Handles preparing training/validation/test splits from annotated data,
saving/loading HDF5 datasets, and data augmentation generators.
"""

import os
import logging

import numpy as np
import h5py
from skimage import transform
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class EdgeDetectionDataset:
    """
    Prepares and manages datasets for training the U-Net model.
    """

    def __init__(self, data_loader, annotation_tool, target_size: tuple = (512, 512)):
        self.data_loader = data_loader
        self.annotation_tool = annotation_tool
        self.target_size = target_size
        if data_loader is not None:
            self.dataset_dir = os.path.join(data_loader.data_dir, 'dataset')
        else:
            self.dataset_dir = None

        if self.dataset_dir and not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def prepare_dataset(self, test_split: float = 0.2, val_split: float = 0.1):
        """Prepare training, validation, and test datasets from annotations.

        Parameters
        ----------
        test_split : float
            Fraction of data for test set.
        val_split : float
            Fraction of data for validation set.

        Returns
        -------
        tuple
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        logger.info("Preparing dataset...")

        if not self.annotation_tool.masks:
            self.annotation_tool.load_annotations()

        if not self.annotation_tool.masks:
            logger.error("No annotations found. Please annotate some images first.")
            return None, None, None

        X = []
        y = []

        for image_key, mask in self.annotation_tool.masks.items():
            file_path, frame = image_key.rsplit('_', 1)
            frame = int(frame)

            try:
                index = self.data_loader.dic_files.index(file_path)
            except ValueError:
                logger.warning("Could not find %s in data loader files", file_path)
                continue

            tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
            if tirf_stack is None or dic_stack is None:
                continue

            tirf_img = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
            dic_img = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

            combined_img = self.data_loader.preprocess_for_model(
                tirf_img, dic_img, self.target_size
            )

            if mask.shape[:2] != self.target_size:
                mask = transform.resize(mask, self.target_size, preserve_range=True)
                mask = (mask > 0.5).astype(np.float32)

            X.append(combined_img)
            y.append(mask)

        if not X:
            logger.error("No valid data found for dataset preparation")
            return None, None, None

        X = np.array(X)
        y = np.array(y)[..., np.newaxis]

        logger.info("Dataset prepared with %d samples", len(X))
        logger.info("Input shape: %s, Output shape: %s", X.shape, y.shape)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )

        val_split_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split_adjusted, random_state=42
        )

        logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

        dataset_path = os.path.join(self.dataset_dir, 'cell_edge_dataset.h5')
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('X_train', data=X_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('X_val', data=X_val)
            f.create_dataset('y_val', data=y_val)
            f.create_dataset('X_test', data=X_test)
            f.create_dataset('y_test', data=y_test)

        logger.info("Dataset saved to %s", dataset_path)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def load_dataset(self):
        """Load previously prepared dataset.

        Returns
        -------
        tuple
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        dataset_path = os.path.join(self.dataset_dir, 'cell_edge_dataset.h5')

        if not os.path.exists(dataset_path):
            logger.error("Dataset not found. Please prepare the dataset first.")
            return None, None, None

        with h5py.File(dataset_path, 'r') as f:
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            X_val = f['X_val'][:]
            y_val = f['y_val'][:]
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]

        logger.info(
            "Dataset loaded: %d train, %d val, %d test",
            len(X_train), len(X_val), len(X_test),
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_data_generator(self, X, y, batch_size: int = 4, augment: bool = True):
        """Create a data generator with optional augmentation.

        Parameters
        ----------
        X : np.ndarray
            Input images.
        y : np.ndarray
            Target masks.
        batch_size : int
            Batch size.
        augment : bool
            Whether to apply data augmentation.

        Yields
        ------
        tuple
            (batch_X, batch_y) arrays.
        """
        indexes = np.arange(len(X))

        while True:
            np.random.shuffle(indexes)

            for start_idx in range(0, len(X), batch_size):
                batch_indexes = indexes[start_idx:start_idx + batch_size]

                batch_X = X[batch_indexes].copy()
                batch_y = y[batch_indexes].copy()

                if augment:
                    for i in range(len(batch_X)):
                        if np.random.rand() > 0.5:
                            batch_X[i] = np.fliplr(batch_X[i])
                            batch_y[i] = np.fliplr(batch_y[i])

                        if np.random.rand() > 0.5:
                            batch_X[i] = np.flipud(batch_X[i])
                            batch_y[i] = np.flipud(batch_y[i])

                        k = np.random.randint(0, 4)
                        if k > 0:
                            batch_X[i] = np.rot90(batch_X[i], k)
                            batch_y[i] = np.rot90(batch_y[i], k)

                        if np.random.rand() > 0.5:
                            factor = np.random.uniform(0.8, 1.2)
                            batch_X[i, :, :, 1] = np.clip(
                                batch_X[i, :, :, 1] * factor, 0, 1
                            )

                yield batch_X, batch_y
