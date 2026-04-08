"""
Cell Edge Detection System
A deep learning-based pipeline for detecting cell boundaries in TIRF and DIC microscopy images
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from skimage import io, transform, filters, measure, morphology
from sklearn.model_selection import train_test_split
import json
import datetime
import pickle
import h5py

# Configure GPU memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU detected and configured: {physical_devices}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU detected. Using CPU.")


class DataLoader:
    """
    Class to handle loading, preprocessing and visualizing TIRF and DIC images
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tirf_files = []
        self.dic_files = []
        self.loaded_tirf_stacks = {}
        self.loaded_dic_stacks = {}

    def scan_directory(self):
        """Scan directory for TIRF and DIC image files"""
        # Using naming convention to distinguish between TIRF and DIC files
        self.tirf_files = sorted(glob.glob(os.path.join(self.data_dir, "*piezo*.tif")))
        self.dic_files = sorted(glob.glob(os.path.join(self.data_dir, "*DIC*.tif")))

        print(f"Found {len(self.tirf_files)} TIRF files and {len(self.dic_files)} DIC files")
        return self.tirf_files, self.dic_files

    def load_tiff_stack(self, file_path):
        """Load a TIFF stack as a 3D numpy array (frames, height, width)"""
        try:
            # Using skimage to handle multi-frame TIFFs
            img_stack = io.imread(file_path)

            # If the stack is 2D (single frame), add a dimension
            if len(img_stack.shape) == 2:
                img_stack = np.expand_dims(img_stack, axis=0)

            return img_stack
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_image_pair(self, index=0):
        """Load a corresponding TIRF and DIC image pair by index"""
        if index >= len(self.tirf_files) or index >= len(self.dic_files):
            print("Index out of range")
            return None, None

        tirf_path = self.tirf_files[index]
        dic_path = self.dic_files[index]

        # Load or retrieve from cache
        if tirf_path not in self.loaded_tirf_stacks:
            self.loaded_tirf_stacks[tirf_path] = self.load_tiff_stack(tirf_path)
        if dic_path not in self.loaded_dic_stacks:
            self.loaded_dic_stacks[dic_path] = self.load_tiff_stack(dic_path)

        return self.loaded_tirf_stacks[tirf_path], self.loaded_dic_stacks[dic_path]

    def preprocess_for_visualization(self, image, percentile_low=1, percentile_high=99):
        """Preprocess an image for better visualization"""
        # Handle float64 data type common in microscopy images
        img = image.astype(np.float32)

        # Calculate percentiles for contrast stretching
        low = np.percentile(img, percentile_low)
        high = np.percentile(img, percentile_high)

        # Clip and normalize to 0-1 range
        img = np.clip(img, low, high)
        img = (img - low) / (high - low)

        return img

    def display_image_pair(self, tirf_stack, dic_stack, frame_idx=0):
        """Display a TIRF and DIC image pair side by side"""
        if tirf_stack is None or dic_stack is None:
            print("No image data to display")
            return

        # Get the specific frame
        tirf_img = tirf_stack[frame_idx] if frame_idx < tirf_stack.shape[0] else tirf_stack[0]
        dic_img = dic_stack[frame_idx] if frame_idx < dic_stack.shape[0] else dic_stack[0]

        # Preprocess for better visualization
        tirf_vis = self.preprocess_for_visualization(tirf_img)
        dic_vis = self.preprocess_for_visualization(dic_img)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display TIRF image
        axes[0].imshow(tirf_vis, cmap='gray')
        axes[0].set_title(f'TIRF Image (Frame {frame_idx})')
        axes[0].axis('off')

        # Display DIC image
        axes[1].imshow(dic_vis, cmap='gray')
        axes[1].set_title(f'DIC Image (Frame {frame_idx})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        return tirf_vis, dic_vis

    def preprocess_for_model(self, tirf_img, dic_img, target_size=(512, 512)):
        """Preprocess images for model input"""
        # Resize if needed
        if tirf_img.shape[:2] != target_size:
            tirf_img = transform.resize(tirf_img, target_size, preserve_range=True)
        if dic_img.shape[:2] != target_size:
            dic_img = transform.resize(dic_img, target_size, preserve_range=True)

        # Normalize to 0-1 range
        tirf_img = self.preprocess_for_visualization(tirf_img)
        dic_img = self.preprocess_for_visualization(dic_img)

        # Stack channels (TIRF and DIC)
        combined_img = np.stack([tirf_img, dic_img], axis=-1)

        return combined_img


class AnnotationTool:
    """
    Tool for annotating cell boundaries in microscopy images
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
        self.masks = {}  # Dictionary to store masks by image path and frame
        self.annotations_dir = os.path.join(data_loader.data_dir, 'annotations')
        self.lasso = None
        self.current_tirf_vis = None
        self.current_dic_vis = None
        self.mask_display = None
        self.dic_img_display = None

        # Create annotations directory if it doesn't exist
        if not os.path.exists(self.annotations_dir):
            os.makedirs(self.annotations_dir)

    def load_image_for_annotation(self, index=0, frame=0):
        """Load and prepare an image for annotation"""
        self.current_index = index
        self.current_frame = frame

        # Load image pair
        tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
        if tirf_stack is None or dic_stack is None:
            print("Failed to load images for annotation")
            return False

        # Get specific frame
        self.current_tirf = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
        self.current_dic = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

        # Preprocess for visualization
        self.current_tirf_vis = self.data_loader.preprocess_for_visualization(self.current_tirf)
        self.current_dic_vis = self.data_loader.preprocess_for_visualization(self.current_dic)

        # Initialize or retrieve mask
        image_key = f"{self.data_loader.dic_files[index]}_{frame}"
        if image_key in self.masks:
            self.mask = self.masks[image_key].copy()
        else:
            self.mask = np.zeros_like(self.current_dic, dtype=np.uint8)

        return True

    def start_annotation_session(self, index=0, frame=0):
        """Start interactive annotation session"""
        success = self.load_image_for_annotation(index, frame)
        if not success:
            return

        # Create figure for annotation
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))

        # Display images
        self.axes[0].imshow(self.current_tirf_vis, cmap='gray')
        self.axes[0].set_title('TIRF Image')
        self.axes[0].axis('off')

        self.dic_img_display = self.axes[1].imshow(self.current_dic_vis, cmap='gray')
        self.axes[1].set_title('DIC Image (Draw here)')

        self.mask_display = self.axes[2].imshow(self.mask, cmap='jet', alpha=0.7)
        self.axes[2].imshow(self.current_dic_vis, cmap='gray', alpha=0.3)
        self.axes[2].set_title('Annotation Mask')
        self.axes[2].axis('off')

        # Clear any stored points from previous sessions
        self.points = []

        # Add status text
        self.status_text = self.fig.text(0.5, 0.01, "Draw cell boundaries using the lasso tool",
                                          ha='center', fontsize=12)

        # Set up lasso selector for annotation
        self.lasso = LassoSelector(self.axes[1], self.on_select)
        # Configure lasso appearance
        self.lasso.lineprops = {'color': 'red', 'linewidth': 2}

        # Add controls
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

        # Add help text
        help_text = """
        Instructions:
        - Draw around cell edges with the lasso tool
        - Selections will remain visible and be added to the mask
        - Click 'Save Mask' when done with this image
        - Click 'Clear Mask' to start over
        - Click 'Next Image' to proceed to the next frame/image
        """
        self.fig.text(0.02, 0.5, help_text, fontsize=10,
                     verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5",
                                                         fc="white", ec="gray", alpha=0.8))

        plt.tight_layout(rect=[0.1, 0.1, 1, 0.9])  # Adjust layout to make room for text
        plt.show()

    def on_select(self, verts):
        """Callback when lasso selection is made"""
        # Check if verts is empty
        if not verts:
            return

        # Convert vertices to numpy array if it's not already
        verts = np.asarray(verts)

        # Create a Path from vertices
        path = Path(verts)

        # Create a grid of points covering the image
        height, width = self.current_dic.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.vstack((x.flatten(), y.flatten())).T

        # Find points inside the path
        mask = path.contains_points(points)
        mask = mask.reshape(height, width)

        # Update the mask - make this a PERSISTENT change
        self.mask[mask] = 1

        # Store the selected vertices for visual feedback
        self.points.append(verts)

        # Update mask display with the modified mask
        self.mask_display.set_data(self.mask)

        try:
            # Keep the selection visible by drawing the outline
            # Make sure verts is the right shape and not empty
            if verts.size > 0 and len(verts.shape) == 2 and verts.shape[1] >= 2:
                x_coords = verts[:, 0]
                y_coords = verts[:, 1]
                self.axes[1].plot(x_coords, y_coords, 'r-', lw=2)
        except Exception as e:
            print(f"Warning: Could not draw selection outline: {e}")

        # Force redraw
        self.fig.canvas.draw_idle()

    def save_mask(self, event):
        """Save the current mask"""
        image_key = f"{self.data_loader.dic_files[self.current_index]}_{self.current_frame}"
        self.masks[image_key] = self.mask.copy()

        # Save to file
        mask_filename = os.path.basename(self.data_loader.dic_files[self.current_index])
        mask_filename = f"{os.path.splitext(mask_filename)[0]}_frame{self.current_frame}_mask.npy"
        mask_path = os.path.join(self.annotations_dir, mask_filename)

        np.save(mask_path, self.mask)
        print(f"Mask saved to {mask_path}")

        # Also save a PNG for visual inspection
        png_path = os.path.join(self.annotations_dir, f"{os.path.splitext(mask_filename)[0]}.png")
        plt.imsave(png_path, self.mask, cmap='gray')

    def clear_mask(self, event):
        """Clear the current mask"""
        self.mask = np.zeros_like(self.current_dic, dtype=np.uint8)
        self.mask_display.set_data(self.mask)

        # Clear the stored points and redraw the canvas
        self.points = []

        # Clear any line plots from previous selections
        for i in range(len(self.axes[1].lines)):
            if len(self.axes[1].lines) > 0:
                self.axes[1].lines[0].remove()

        self.fig.canvas.draw_idle()

    def next_image(self, event):
        """Move to the next image or frame"""
        plt.close(self.fig)

        # First try to advance the frame
        next_frame = self.current_frame + 1
        tirf_stack, dic_stack = self.data_loader.load_image_pair(self.current_index)

        if next_frame < tirf_stack.shape[0]:
            self.start_annotation_session(self.current_index, next_frame)
        else:
            # Move to the next image pair
            next_index = self.current_index + 1
            if next_index < len(self.data_loader.tirf_files):
                self.start_annotation_session(next_index, 0)
            else:
                print("Reached the end of the dataset")

    def load_annotations(self):
        """Load existing annotations from files"""
        mask_files = glob.glob(os.path.join(self.annotations_dir, "*_mask.npy"))

        for mask_file in mask_files:
            # Extract image path and frame number from filename
            filename = os.path.basename(mask_file)
            parts = filename.split('_')

            # Find the original image file
            original_prefix = '_'.join(parts[:-2])  # Exclude "_frameX_mask"
            matched_dic_files = [f for f in self.data_loader.dic_files if original_prefix in f]

            if matched_dic_files:
                dic_path = matched_dic_files[0]
                frame_num = int(parts[-2].replace('frame', ''))

                # Load the mask
                mask = np.load(mask_file)

                # Store in the masks dictionary
                image_key = f"{dic_path}_{frame_num}"
                self.masks[image_key] = mask

        print(f"Loaded {len(self.masks)} annotation masks")


class EdgeDetectionDataset:
    """
    Class to prepare and manage datasets for training the U-Net model
    """
    def __init__(self, data_loader, annotation_tool, target_size=(512, 512)):
        self.data_loader = data_loader
        self.annotation_tool = annotation_tool
        self.target_size = target_size
        self.dataset_dir = os.path.join(data_loader.data_dir, 'dataset')

        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def prepare_dataset(self, test_split=0.2, val_split=0.1):
        """Prepare training, validation and test datasets"""
        print("Preparing dataset...")

        # Load annotations if not already loaded
        if not self.annotation_tool.masks:
            self.annotation_tool.load_annotations()

        if not self.annotation_tool.masks:
            print("No annotations found. Please annotate some images first.")
            return None, None, None

        # Prepare data arrays
        X = []  # Input images (TIRF + DIC)
        y = []  # Target masks

        # Process each annotated image
        for image_key, mask in self.annotation_tool.masks.items():
            # Parse the image key to get file path and frame
            file_path, frame = image_key.rsplit('_', 1)
            frame = int(frame)

            # Find the index of this file in the data loader
            try:
                index = self.data_loader.dic_files.index(file_path)
            except ValueError:
                print(f"Warning: Could not find {file_path} in data loader files")
                continue

            # Load the image pair
            tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
            if tirf_stack is None or dic_stack is None:
                continue

            # Get specific frame
            tirf_img = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
            dic_img = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

            # Preprocess for model
            combined_img = self.data_loader.preprocess_for_model(tirf_img, dic_img, self.target_size)

            # Resize mask if needed
            if mask.shape[:2] != self.target_size:
                mask = transform.resize(mask, self.target_size, preserve_range=True)
                mask = (mask > 0.5).astype(np.float32)  # Binarize after resize

            # Add to dataset
            X.append(combined_img)
            y.append(mask)

        if not X:
            print("No valid data found for dataset preparation")
            return None, None, None

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)[..., np.newaxis]  # Add channel dimension

        print(f"Dataset prepared with {len(X)} samples")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")

        # Split into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # Adjust validation split
        val_split_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split_adjusted, random_state=42)

        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Save the dataset
        dataset_path = os.path.join(self.dataset_dir, 'cell_edge_dataset.h5')
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('X_train', data=X_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('X_val', data=X_val)
            f.create_dataset('y_val', data=y_val)
            f.create_dataset('X_test', data=X_test)
            f.create_dataset('y_test', data=y_test)

        print(f"Dataset saved to {dataset_path}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def load_dataset(self):
        """Load previously prepared dataset"""
        dataset_path = os.path.join(self.dataset_dir, 'cell_edge_dataset.h5')

        if not os.path.exists(dataset_path):
            print("Dataset not found. Please prepare the dataset first.")
            return None, None, None

        with h5py.File(dataset_path, 'r') as f:
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            X_val = f['X_val'][:]
            y_val = f['y_val'][:]
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]

        print(f"Dataset loaded with {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_data_generator(self, X, y, batch_size=4, augment=True):
        """Create a data generator with optional augmentation"""
        indexes = np.arange(len(X))

        while True:
            # Shuffle indexes
            np.random.shuffle(indexes)

            for start_idx in range(0, len(X), batch_size):
                batch_indexes = indexes[start_idx:start_idx+batch_size]

                batch_X = X[batch_indexes].copy()
                batch_y = y[batch_indexes].copy()

                if augment:
                    for i in range(len(batch_X)):
                        # Random horizontal flip
                        if np.random.rand() > 0.5:
                            batch_X[i] = np.fliplr(batch_X[i])
                            batch_y[i] = np.fliplr(batch_y[i])

                        # Random vertical flip
                        if np.random.rand() > 0.5:
                            batch_X[i] = np.flipud(batch_X[i])
                            batch_y[i] = np.flipud(batch_y[i])

                        # Random rotation
                        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
                        if k > 0:
                            batch_X[i] = np.rot90(batch_X[i], k)
                            batch_y[i] = np.rot90(batch_y[i], k)

                        # Random brightness adjustment for DIC channel
                        if np.random.rand() > 0.5:
                            factor = np.random.uniform(0.8, 1.2)
                            batch_X[i, :, :, 1] = np.clip(batch_X[i, :, :, 1] * factor, 0, 1)

                yield batch_X, batch_y


class UNetModel:
    """
    U-Net model for cell edge detection
    """
    def __init__(self, input_shape=(512, 512, 2), n_classes=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = None
        self.models_dir = os.path.join(os.getcwd(), 'models')

        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def build_unet(self):
        """Build the U-Net architecture"""
        # Input layer
        inputs = keras.Input(shape=self.input_shape)

        # Encoder path (downsampling)
        # Level 1
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        # Level 2
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Level 3
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Level 4
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bridge
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)

        # Decoder path (upsampling)
        # Level 4
        up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
        concat6 = layers.Concatenate()([drop4, up6])
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(concat6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

        # Level 3
        up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
        concat7 = layers.Concatenate()([conv3, up7])
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

        # Level 2
        up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
        concat8 = layers.Concatenate()([conv2, up8])
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

        # Level 1
        up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
        concat9 = layers.Concatenate()([conv1, up9])
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

        # Output layer
        outputs = layers.Conv2D(self.n_classes, 1, activation='sigmoid')(conv9)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model with dice loss and dice coefficient metric
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.dice_loss,
            metrics=[self.dice_coef]
        )

        print(self.model.summary())
        return self.model

    def dice_coef(self, y_true, y_pred, smooth=1):
        """Dice coefficient metric for model evaluation"""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        """Dice loss function for training"""
        return 1 - self.dice_coef(y_true, y_pred)

    def train(self, train_data, val_data, batch_size=4, epochs=50, use_generator=True):
        """Train the U-Net model"""
        if self.model is None:
            self.build_unet()

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Create output folder for this training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(self.models_dir, f"unet_training_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_dice_coef',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                verbose=1,
                restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(output_dir, 'logs')
            )
        ]

        # Create dataset generator
        if use_generator:
            # Create dataset instance
            dataset = EdgeDetectionDataset(None, None)

            # Create generators
            train_gen = dataset.get_data_generator(X_train, y_train, batch_size=batch_size, augment=True)
            val_gen = dataset.get_data_generator(X_val, y_val, batch_size=batch_size, augment=False)

            # Number of steps per epoch
            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size

            # Train with generator
            history = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks
            )
        else:
            # Train with regular fit
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )

        # Save training history
        with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        # Plot training history
        self.plot_training_history(history, output_dir)

        return history, output_dir

    def plot_training_history(self, history, output_dir):
        """Plot and save training history graphs"""
        # Plot loss
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['dice_coef'], label='Training Dice Coefficient')
        plt.plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

    def load_model(self, model_path):
        """Load a trained model from a file"""
        # Custom objects for loading model with custom loss/metrics
        custom_objects = {
            'dice_loss': self.dice_loss,
            'dice_coef': self.dice_coef
        }

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded from {model_path}")
        return self.model

    def predict(self, X):
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model not built or loaded yet")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not built or loaded yet")

        results = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Dice Coefficient: {results[1]:.4f}")

        return results


class EdgeDetector:
    """
    Class for detecting and refining cell edges from model predictions
    """
    def __init__(self, model):
        self.model = model

    def detect_edges(self, tirf_img, dic_img, target_size=(512, 512)):
        """Detect cell edges from a TIRF and DIC image pair"""
        # Preprocess images
        dl = DataLoader(None)  # Temporary instance for preprocessing
        combined_img = dl.preprocess_for_model(tirf_img, dic_img, target_size)
        combined_img = np.expand_dims(combined_img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = self.model.predict(combined_img)[0, :, :, 0]

        # Resize prediction back to original size if needed
        if prediction.shape[:2] != tirf_img.shape[:2]:
            prediction = transform.resize(prediction, tirf_img.shape[:2], preserve_range=True)

        return prediction

    def refine_edges(self, prediction, threshold=0.5, edge_width=1):
        """Refine edges from model prediction"""
        # Binarize the prediction
        binary_mask = (prediction > threshold).astype(np.uint8)

        # Find edges using morphological operations
        if edge_width == 1:
            # Simple edge detection with erosion
            kernel = np.ones((3, 3), np.uint8)
            eroded = morphology.erosion(binary_mask, kernel)
            edges = binary_mask - eroded
        else:
            # Get contours for thicker edges
            contours = measure.find_contours(prediction, threshold)
            edges = np.zeros_like(prediction)

            for contour in contours:
                # Round contour points to integers
                contour = np.round(contour).astype(int)

                # Draw contour in the edge image
                for i in range(len(contour)):
                    x, y = contour[i]
                    if 0 <= x < edges.shape[0] and 0 <= y < edges.shape[1]:
                        edges[x, y] = 1

                        # Make edges thicker if requested
                        if edge_width > 1:
                            for dx in range(-edge_width//2, edge_width//2 + 1):
                                for dy in range(-edge_width//2, edge_width//2 + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < edges.shape[0] and 0 <= ny < edges.shape[1]:
                                        edges[nx, ny] = 1

        return edges

    def visualize_edges(self, tirf_img, dic_img, edges, alpha=0.7):
        """Visualize detected edges overlaid on original images"""
        # Preprocess images for visualization
        dl = DataLoader(None)
        tirf_vis = dl.preprocess_for_visualization(tirf_img)
        dic_vis = dl.preprocess_for_visualization(dic_img)

        # Create RGB versions of grayscale images
        tirf_rgb = np.stack([tirf_vis]*3, axis=-1)
        dic_rgb = np.stack([dic_vis]*3, axis=-1)

        # Create edge overlay (red channel)
        edges_rgb = np.zeros_like(tirf_rgb)
        edges_rgb[:, :, 0] = edges  # Red channel

        # Overlay edges on images
        tirf_overlay = tirf_rgb.copy()
        tirf_overlay[edges > 0] = [1, 0, 0]  # Red edges

        dic_overlay = dic_rgb.copy()
        dic_overlay[edges > 0] = [1, 0, 0]  # Red edges

        # Display results
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
    Main application class that integrates all components
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data')

        # Create main components
        self.data_loader = DataLoader(self.data_dir)
        self.annotation_tool = AnnotationTool(self.data_loader)
        self.dataset = EdgeDetectionDataset(self.data_loader, self.annotation_tool)
        self.model = UNetModel()
        self.edge_detector = None  # Will be initialized after model is trained/loaded

    def initialize(self):
        """Initialize the application"""
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)

        # Scan for data files
        self.data_loader.scan_directory()

        # Load any existing annotations
        self.annotation_tool.load_annotations()

    def annotate_images(self):
        """Start the annotation tool"""
        self.annotation_tool.start_annotation_session()

    def train_model(self, batch_size=4, epochs=50):
        """Prepare dataset and train the model"""
        # Prepare or load dataset
        dataset_ready = False

        try:
            # Try loading existing dataset
            train_data, val_data, test_data = self.dataset.load_dataset()
            if train_data[0] is not None:
                dataset_ready = True
        except (FileNotFoundError, OSError):
            pass

        if not dataset_ready:
            # Prepare new dataset
            train_data, val_data, test_data = self.dataset.prepare_dataset()
            if train_data[0] is None:
                print("Failed to prepare dataset. Please annotate some images first.")
                return None

        # Build and train the model
        self.model.build_unet()
        history, output_dir = self.model.train(train_data, val_data, batch_size=batch_size, epochs=epochs)

        # Initialize edge detector with trained model
        self.edge_detector = EdgeDetector(self.model)

        # Evaluate on test data
        print("\nEvaluating model on test data:")
        test_results = self.model.evaluate(test_data[0], test_data[1])

        # Save test evaluation results
        with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_results[0]:.4f}\n")
            f.write(f"Test Dice Coefficient: {test_results[1]:.4f}\n")

        return history, output_dir

    def load_trained_model(self, model_path):
        """Load a previously trained model"""
        self.model.load_model(model_path)
        self.edge_detector = EdgeDetector(self.model)

    def process_image_pair(self, index=0, frame=0, threshold=0.5, edge_width=1):
        """Process a TIRF and DIC image pair to detect cell edges"""
        # Load image pair
        tirf_stack, dic_stack = self.data_loader.load_image_pair(index)
        if tirf_stack is None or dic_stack is None:
            print("Failed to load images")
            return None

        # Get specific frame
        tirf_img = tirf_stack[frame] if frame < tirf_stack.shape[0] else tirf_stack[0]
        dic_img = dic_stack[frame] if frame < dic_stack.shape[0] else dic_stack[0]

        # Detect edges
        prediction = self.edge_detector.detect_edges(tirf_img, dic_img)
        edges = self.edge_detector.refine_edges(prediction, threshold, edge_width)

        # Visualize results
        tirf_overlay, dic_overlay = self.edge_detector.visualize_edges(tirf_img, dic_img, edges)

        return prediction, edges, tirf_overlay, dic_overlay

    def batch_process(self, output_dir=None, threshold=0.5, edge_width=1):
        """Process all image pairs in the dataset"""
        if self.edge_detector is None:
            print("Edge detector not initialized. Please train or load a model first.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'results',
                                     f"batch_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        os.makedirs(output_dir, exist_ok=True)

        # Process each image pair
        for i in range(len(self.data_loader.tirf_files)):
            tirf_stack, dic_stack = self.data_loader.load_image_pair(i)
            if tirf_stack is None or dic_stack is None:
                continue

            # Process each frame
            for frame in range(min(tirf_stack.shape[0], dic_stack.shape[0])):
                tirf_img = tirf_stack[frame]
                dic_img = dic_stack[frame]

                # Detect edges
                prediction = self.edge_detector.detect_edges(tirf_img, dic_img)
                edges = self.edge_detector.refine_edges(prediction, threshold, edge_width)

                # Save results
                base_name = f"{os.path.splitext(os.path.basename(self.data_loader.tirf_files[i]))[0]}_frame{frame}"

                # Save prediction
                prediction_path = os.path.join(output_dir, f"{base_name}_pred.npy")
                np.save(prediction_path, prediction)

                # Save edges
                edges_path = os.path.join(output_dir, f"{base_name}_edges.npy")
                np.save(edges_path, edges)

                # Save visualizations
                dl = DataLoader(None)
                tirf_vis = dl.preprocess_for_visualization(tirf_img)
                dic_vis = dl.preprocess_for_visualization(dic_img)

                # Create overlays
                tirf_rgb = np.stack([tirf_vis]*3, axis=-1)
                dic_rgb = np.stack([dic_vis]*3, axis=-1)

                tirf_overlay = tirf_rgb.copy()
                tirf_overlay[edges > 0] = [1, 0, 0]  # Red edges

                dic_overlay = dic_rgb.copy()
                dic_overlay[edges > 0] = [1, 0, 0]  # Red edges

                # Save overlay images
                plt.imsave(os.path.join(output_dir, f"{base_name}_tirf_overlay.png"), tirf_overlay)
                plt.imsave(os.path.join(output_dir, f"{base_name}_dic_overlay.png"), dic_overlay)

                print(f"Processed {base_name}")

        print(f"Batch processing complete. Results saved in {output_dir}")


# Main execution example
if __name__ == "__main__":
    # Example usage
    app = EdgeDetectionApp()
    app.initialize()

    # Check if we have data
    if len(app.data_loader.tirf_files) == 0:
        print("No TIRF/DIC image files found. Please add data to the data directory.")
    else:
        # Option 1: Annotate images (if needed)
        app.annotate_images()

        # Option 2: Train model (if annotations exist)
        # history, output_dir = app.train_model(epochs=30)

        # Option 3: Load an existing model
        # app.load_trained_model('models/best_model.h5')

        # Option 4: Process a single image pair
        # prediction, edges, tirf_overlay, dic_overlay = app.process_image_pair(index=0, frame=0)

        # Option 5: Batch process all images
        # app.batch_process()

        print("Cell Edge Detection System initialized successfully.")
        print("Available options:")
        print("1. app.annotate_images() - Start the annotation tool")
        print("2. app.train_model() - Train the U-Net model")
        print("3. app.load_trained_model(path) - Load an existing model")
        print("4. app.process_image_pair() - Process a single image pair")
        print("5. app.batch_process() - Process all images")
