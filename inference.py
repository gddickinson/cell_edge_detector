"""
Edge detection inference and visualization.

Provides the EdgeDetector class for detecting and refining cell edges
from trained model predictions, plus the main EdgeDetectionApp that
integrates all components.
"""

import os
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology, measure

from data_loader import DataLoader
from annotator import AnnotationTool
from dataset import EdgeDetectionDataset
from model import UNetModel

logger = logging.getLogger(__name__)


class EdgeDetector:
    """
    Detects and refines cell edges from model predictions.

    Parameters
    ----------
    model : UNetModel
        A trained UNetModel instance.
    """

    def __init__(self, model: UNetModel):
        self.model = model

    def detect_edges(
        self, tirf_img: np.ndarray, dic_img: np.ndarray, target_size: tuple = (512, 512)
    ) -> np.ndarray:
        """Detect cell edges from a TIRF and DIC image pair.

        Parameters
        ----------
        tirf_img : np.ndarray
            TIRF image (2D).
        dic_img : np.ndarray
            DIC image (2D).
        target_size : tuple
            Model input size.

        Returns
        -------
        np.ndarray
            Prediction probability map.
        """
        dl = DataLoader(None)
        combined_img = dl.preprocess_for_model(tirf_img, dic_img, target_size)
        combined_img = np.expand_dims(combined_img, axis=0)

        prediction = self.model.predict(combined_img)[0, :, :, 0]

        if prediction.shape[:2] != tirf_img.shape[:2]:
            prediction = transform.resize(
                prediction, tirf_img.shape[:2], preserve_range=True
            )

        return prediction

    def refine_edges(
        self, prediction: np.ndarray, threshold: float = 0.5, edge_width: int = 1
    ) -> np.ndarray:
        """Refine edges from model prediction.

        Parameters
        ----------
        prediction : np.ndarray
            Prediction probability map.
        threshold : float
            Binarization threshold.
        edge_width : int
            Width of output edges in pixels.

        Returns
        -------
        np.ndarray
            Binary edge map.
        """
        binary_mask = (prediction > threshold).astype(np.uint8)

        if edge_width == 1:
            kernel = np.ones((3, 3), np.uint8)
            eroded = morphology.erosion(binary_mask, kernel)
            edges = binary_mask - eroded
        else:
            contours = measure.find_contours(prediction, threshold)
            edges = np.zeros_like(prediction)

            for contour in contours:
                contour = np.round(contour).astype(int)

                for i in range(len(contour)):
                    x, y = contour[i]
                    if 0 <= x < edges.shape[0] and 0 <= y < edges.shape[1]:
                        edges[x, y] = 1

                        if edge_width > 1:
                            for dx in range(-edge_width // 2, edge_width // 2 + 1):
                                for dy in range(-edge_width // 2, edge_width // 2 + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < edges.shape[0] and 0 <= ny < edges.shape[1]:
                                        edges[nx, ny] = 1

        return edges

    def visualize_edges(
        self, tirf_img: np.ndarray, dic_img: np.ndarray, edges: np.ndarray, alpha: float = 0.7
    ):
        """Visualize detected edges overlaid on original images.

        Parameters
        ----------
        tirf_img : np.ndarray
            Original TIRF image.
        dic_img : np.ndarray
            Original DIC image.
        edges : np.ndarray
            Binary edge map.
        alpha : float
            Edge overlay alpha.

        Returns
        -------
        tuple
            (tirf_overlay, dic_overlay) RGB arrays.
        """
        dl = DataLoader(None)
        tirf_vis = dl.preprocess_for_visualization(tirf_img)
        dic_vis = dl.preprocess_for_visualization(dic_img)

        tirf_rgb = np.stack([tirf_vis] * 3, axis=-1)
        dic_rgb = np.stack([dic_vis] * 3, axis=-1)

        tirf_overlay = tirf_rgb.copy()
        tirf_overlay[edges > 0] = [1, 0, 0]

        dic_overlay = dic_rgb.copy()
        dic_overlay[edges > 0] = [1, 0, 0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(tirf_vis, cmap='gray')
        axes[0, 0].set_title('TIRF Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dic_vis, cmap='gray')
        axes[0, 1].set_title('DIC Image')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(tirf_overlay)
        axes[1, 0].set_title('TIRF with Cell Edges')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(dic_overlay)
        axes[1, 1].set_title('DIC with Cell Edges')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        return tirf_overlay, dic_overlay


class EdgeDetectionApp:
    """
    Main application class that integrates all components.

    Parameters
    ----------
    data_dir : str, optional
        Path to data directory. Defaults to './data'.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data')

        self.data_loader = DataLoader(self.data_dir)
        self.annotation_tool = AnnotationTool(self.data_loader)
        self.dataset = EdgeDetectionDataset(self.data_loader, self.annotation_tool)
        self.model = UNetModel()
        self.edge_detector = None

    def initialize(self):
        """Initialize the application by scanning for data and loading annotations."""
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_loader.scan_directory()
        self.annotation_tool.load_annotations()

    def annotate_images(self):
        """Start the annotation tool."""
        self.annotation_tool.start_annotation_session()

    def train_model(self, batch_size: int = 4, epochs: int = 50):
        """Prepare dataset and train the model.

        Parameters
        ----------
        batch_size : int
            Training batch size.
        epochs : int
            Number of training epochs.

        Returns
        -------
        tuple or None
            (history, output_dir) on success, None on failure.
        """
        dataset_ready = False

        try:
            train_data, val_data, test_data = self.dataset.load_dataset()
            if train_data[0] is not None:
                dataset_ready = True
        except (FileNotFoundError, OSError) as e:
            logger.debug("Could not load existing dataset: %s", e)

        if not dataset_ready:
            train_data, val_data, test_data = self.dataset.prepare_dataset()
            if train_data[0] is None:
                logger.error("Failed to prepare dataset. Please annotate some images first.")
                return None

        self.model.build_unet()
        history, output_dir = self.model.train(
            train_data, val_data, batch_size=batch_size, epochs=epochs
        )

        self.edge_detector = EdgeDetector(self.model)

        logger.info("Evaluating model on test data:")
        test_results = self.model.evaluate(test_data[0], test_data[1])

        with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_results[0]:.4f}\n")
            f.write(f"Test Dice Coefficient: {test_results[1]:.4f}\n")

        return history, output_dir

    def load_trained_model(self, model_path: str):
        """Load a previously trained model.

        Parameters
        ----------
        model_path : str
            Path to saved model file.
        """
        self.model.load_model(model_path)
        self.edge_detector = EdgeDetector(self.model)

    def process_image_pair(
        self, index: int = 0, frame: int = 0, threshold: float = 0.5, edge_width: int = 1
    ):
        """Process a TIRF and DIC image pair to detect cell edges.

        Returns
        -------
        tuple or None
            (prediction, edges, tirf_overlay, dic_overlay) on success.
        """
        tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
        if tirf_stack is None or dic_stack is None:
            logger.error("Failed to load images")
            return None

        tirf_img = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
        dic_img = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

        prediction = self.edge_detector.detect_edges(tirf_img, dic_img)
        edges = self.edge_detector.refine_edges(prediction, threshold, edge_width)

        tirf_overlay, dic_overlay = self.edge_detector.visualize_edges(
            tirf_img, dic_img, edges
        )

        return prediction, edges, tirf_overlay, dic_overlay

    def batch_process(
        self, output_dir: str = None, threshold: float = 0.5, edge_width: int = 1
    ):
        """Process all image pairs in the dataset.

        Parameters
        ----------
        output_dir : str, optional
            Output directory for results.
        threshold : float
            Binarization threshold.
        edge_width : int
            Edge width in pixels.
        """
        if self.edge_detector is None:
            logger.error("Edge detector not initialized. Train or load a model first.")
            return

        if output_dir is None:
            output_dir = os.path.join(
                self.data_dir, 'results',
                f"batch_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )

        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(self.data_loader.tirf_files)):
            tirf_stack, dic_stack = self.data_loader.load_image_pair(i)
            if tirf_stack is None or dic_stack is None:
                continue

            for frame in range(min(tirf_stack.shape[0], dic_stack.shape[0])):
                tirf_img = tirf_stack[frame]
                dic_img = dic_stack[frame]

                prediction = self.edge_detector.detect_edges(tirf_img, dic_img)
                edges = self.edge_detector.refine_edges(prediction, threshold, edge_width)

                base_name = (
                    f"{os.path.splitext(os.path.basename(self.data_loader.tirf_files[i]))[0]}"
                    f"_frame{frame}"
                )

                np.save(os.path.join(output_dir, f"{base_name}_pred.npy"), prediction)
                np.save(os.path.join(output_dir, f"{base_name}_edges.npy"), edges)

                dl = DataLoader(None)
                tirf_vis = dl.preprocess_for_visualization(tirf_img)
                dic_vis = dl.preprocess_for_visualization(dic_img)

                tirf_rgb = np.stack([tirf_vis] * 3, axis=-1)
                dic_rgb = np.stack([dic_vis] * 3, axis=-1)

                tirf_overlay = tirf_rgb.copy()
                tirf_overlay[edges > 0] = [1, 0, 0]

                dic_overlay = dic_rgb.copy()
                dic_overlay[edges > 0] = [1, 0, 0]

                plt.imsave(
                    os.path.join(output_dir, f"{base_name}_tirf_overlay.png"),
                    tirf_overlay,
                )
                plt.imsave(
                    os.path.join(output_dir, f"{base_name}_dic_overlay.png"),
                    dic_overlay,
                )

                logger.info("Processed %s", base_name)

        logger.info("Batch processing complete. Results saved in %s", output_dir)


if __name__ == "__main__":
    from gpu_config import configure_gpu
    configure_gpu()

    logging.basicConfig(level=logging.INFO)

    app = EdgeDetectionApp()
    app.initialize()

    if len(app.data_loader.tirf_files) == 0:
        print("No TIRF/DIC image files found. Please add data to the data directory.")
    else:
        app.annotate_images()

        print("Cell Edge Detection System initialized successfully.")
        print("Available options:")
        print("1. app.annotate_images() - Start the annotation tool")
        print("2. app.train_model() - Train the U-Net model")
        print("3. app.load_trained_model(path) - Load an existing model")
        print("4. app.process_image_pair() - Process a single image pair")
        print("5. app.batch_process() - Process all images")
