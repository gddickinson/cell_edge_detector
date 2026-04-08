"""
Interactive annotation tool for cell boundaries in microscopy images.

Provides a matplotlib-based lasso annotation interface for creating
ground truth masks for training the U-Net model.
"""

import os
import glob
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

logger = logging.getLogger(__name__)


class AnnotationTool:
    """
    Tool for annotating cell boundaries in microscopy images.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.current_tirf = None
        self.current_dic = None
        self.current_frame = 0
        self.current_index = 0
        self.mask = None
        self.fig = None
        self.axes = None
        self.points = []
        self.masks = {}
        self.annotations_dir = os.path.join(data_loader.data_dir, 'annotations')
        self.lasso = None
        self.current_tirf_vis = None
        self.current_dic_vis = None
        self.mask_display = None
        self.dic_img_display = None

        if not os.path.exists(self.annotations_dir):
            os.makedirs(self.annotations_dir)

    def load_image_for_annotation(self, index: int = 0, frame: int = 0) -> bool:
        """Load and prepare an image for annotation.

        Parameters
        ----------
        index : int
            Image pair index.
        frame : int
            Frame index within the stack.

        Returns
        -------
        bool
            True if successfully loaded.
        """
        self.current_index = index
        self.current_frame = frame

        tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
        if tirf_stack is None or dic_stack is None:
            logger.error("Failed to load images for annotation")
            return False

        self.current_tirf = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
        self.current_dic = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

        self.current_tirf_vis = self.data_loader.preprocess_for_visualization(self.current_tirf)
        self.current_dic_vis = self.data_loader.preprocess_for_visualization(self.current_dic)

        image_key = f"{self.data_loader.dic_files[index]}_{frame}"
        if image_key in self.masks:
            self.mask = self.masks[image_key].copy()
        else:
            self.mask = np.zeros_like(self.current_dic, dtype=np.uint8)

        return True

    def start_annotation_session(self, index: int = 0, frame: int = 0):
        """Start interactive annotation session."""
        success = self.load_image_for_annotation(index, frame)
        if not success:
            return

        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))

        self.axes[0].imshow(self.current_tirf_vis, cmap='gray')
        self.axes[0].set_title('TIRF Image')
        self.axes[0].axis('off')

        self.dic_img_display = self.axes[1].imshow(self.current_dic_vis, cmap='gray')
        self.axes[1].set_title('DIC Image (Draw here)')

        self.mask_display = self.axes[2].imshow(self.mask, cmap='jet', alpha=0.7)
        self.axes[2].imshow(self.current_dic_vis, cmap='gray', alpha=0.3)
        self.axes[2].set_title('Annotation Mask')
        self.axes[2].axis('off')

        self.points = []

        self.status_text = self.fig.text(
            0.5, 0.01, "Draw cell boundaries using the lasso tool",
            ha='center', fontsize=12
        )

        self.lasso = LassoSelector(self.axes[1], self.on_select)
        self.lasso.lineprops = {'color': 'red', 'linewidth': 2}

        plt.subplots_adjust(bottom=0.15)
        ax_save = plt.axes([0.35, 0.05, 0.1, 0.075])
        self.btn_save = plt.Button(ax_save, 'Save Mask')
        self.btn_save.on_clicked(self.save_mask)

        ax_clear = plt.axes([0.5, 0.05, 0.1, 0.075])
        self.btn_clear = plt.Button(ax_clear, 'Clear Mask')
        self.btn_clear.on_clicked(self.clear_mask)

        ax_next = plt.axes([0.65, 0.05, 0.1, 0.075])
        self.btn_next = plt.Button(ax_next, 'Next Image')
        self.btn_next.on_clicked(self.next_image)

        help_text = (
            "Instructions:\n"
            "- Draw around cell edges with the lasso tool\n"
            "- Selections will remain visible and be added to the mask\n"
            "- Click 'Save Mask' when done with this image\n"
            "- Click 'Clear Mask' to start over\n"
            "- Click 'Next Image' to proceed to the next frame/image"
        )
        self.fig.text(
            0.02, 0.5, help_text, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
        )

        plt.tight_layout(rect=[0.1, 0.1, 1, 0.9])
        plt.show()

    def on_select(self, verts):
        """Callback when lasso selection is made."""
        if not verts:
            return

        verts = np.asarray(verts)
        path = Path(verts)

        height, width = self.current_dic.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.vstack((x.flatten(), y.flatten())).T

        mask = path.contains_points(points)
        mask = mask.reshape(height, width)

        self.mask[mask] = 1
        self.points.append(verts)
        self.mask_display.set_data(self.mask)

        try:
            if verts.size > 0 and len(verts.shape) == 2 and verts.shape[1] >= 2:
                x_coords = verts[:, 0]
                y_coords = verts[:, 1]
                self.axes[1].plot(x_coords, y_coords, 'r-', lw=2)
        except (IndexError, ValueError) as e:
            logger.warning("Could not draw selection outline: %s", e)

        self.fig.canvas.draw_idle()

    def save_mask(self, event):
        """Save the current mask."""
        image_key = f"{self.data_loader.dic_files[self.current_index]}_{self.current_frame}"
        self.masks[image_key] = self.mask.copy()

        mask_filename = os.path.basename(self.data_loader.dic_files[self.current_index])
        mask_filename = f"{os.path.splitext(mask_filename)[0]}_frame{self.current_frame}_mask.npy"
        mask_path = os.path.join(self.annotations_dir, mask_filename)

        np.save(mask_path, self.mask)
        logger.info("Mask saved to %s", mask_path)

        png_path = os.path.join(
            self.annotations_dir,
            f"{os.path.splitext(mask_filename)[0]}.png",
        )
        plt.imsave(png_path, self.mask, cmap='gray')

    def clear_mask(self, event):
        """Clear the current mask."""
        self.mask = np.zeros_like(self.current_dic, dtype=np.uint8)
        self.mask_display.set_data(self.mask)
        self.points = []

        for i in range(len(self.axes[1].lines)):
            if len(self.axes[1].lines) > 0:
                self.axes[1].lines[0].remove()

        self.fig.canvas.draw_idle()

    def next_image(self, event):
        """Move to the next image or frame."""
        plt.close(self.fig)

        next_frame = self.current_frame + 1
        tirf_stack, dic_stack = self.data_loader.load_image_pair(self.current_index)

        if next_frame < tirf_stack.shape[0]:
            self.start_annotation_session(self.current_index, next_frame)
        else:
            next_index = self.current_index + 1
            if next_index < len(self.data_loader.tirf_files):
                self.start_annotation_session(next_index, 0)
            else:
                logger.info("Reached the end of the dataset")

    def load_annotations(self):
        """Load existing annotations from files."""
        mask_files = glob.glob(os.path.join(self.annotations_dir, "*_mask.npy"))

        for mask_file in mask_files:
            filename = os.path.basename(mask_file)
            parts = filename.split('_')

            original_prefix = '_'.join(parts[:-2])
            matched_dic_files = [f for f in self.data_loader.dic_files if original_prefix in f]

            if matched_dic_files:
                dic_path = matched_dic_files[0]
                frame_num = int(parts[-2].replace('frame', ''))

                mask = np.load(mask_file)

                image_key = f"{dic_path}_{frame_num}"
                self.masks[image_key] = mask

        logger.info("Loaded %d annotation masks", len(self.masks))
