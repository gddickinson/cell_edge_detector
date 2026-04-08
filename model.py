"""
U-Net model for cell edge detection.

Defines the U-Net architecture, training loop, evaluation, and
model persistence for cell boundary segmentation.
"""

import os
import datetime
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Lazy imports for tensorflow to avoid GPU config issues at import time
_tf = None
_keras = None
_layers = None


def _import_tf():
    """Lazily import TensorFlow and Keras."""
    global _tf, _keras, _layers
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers as tf_layers
        _tf = tf
        _keras = keras
        _layers = tf_layers
    return _tf, _keras, _layers


class UNetModel:
    """
    U-Net model for cell edge detection.

    Parameters
    ----------
    input_shape : tuple
        Input image shape (height, width, channels).
    n_classes : int
        Number of output classes.
    """

    def __init__(self, input_shape: tuple = (512, 512, 2), n_classes: int = 1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = None
        self.models_dir = os.path.join(os.getcwd(), 'models')

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def build_unet(self):
        """Build the U-Net architecture.

        Returns
        -------
        keras.Model
            Compiled U-Net model.
        """
        tf, keras, layers = _import_tf()

        inputs = keras.Input(shape=self.input_shape)

        # Encoder path
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bridge
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)

        # Decoder path
        up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
        concat6 = layers.Concatenate()([drop4, up6])
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(concat6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
        concat7 = layers.Concatenate()([conv3, up7])
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
        concat8 = layers.Concatenate()([conv2, up8])
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
        concat9 = layers.Concatenate()([conv1, up9])
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

        outputs = layers.Conv2D(self.n_classes, 1, activation='sigmoid')(conv9)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.dice_loss,
            metrics=[self.dice_coef],
        )

        logger.info("U-Net model built successfully")
        return self.model

    def dice_coef(self, y_true, y_pred, smooth=1):
        """Dice coefficient metric for model evaluation."""
        tf, _, _ = _import_tf()
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    def dice_loss(self, y_true, y_pred):
        """Dice loss function for training."""
        return 1 - self.dice_coef(y_true, y_pred)

    def train(
        self,
        train_data,
        val_data,
        batch_size: int = 4,
        epochs: int = 50,
        use_generator: bool = True,
    ):
        """Train the U-Net model.

        Parameters
        ----------
        train_data : tuple
            (X_train, y_train).
        val_data : tuple
            (X_val, y_val).
        batch_size : int
            Training batch size.
        epochs : int
            Number of training epochs.
        use_generator : bool
            Whether to use a data generator with augmentation.

        Returns
        -------
        tuple
            (history, output_dir).
        """
        _, keras, _ = _import_tf()
        from dataset import EdgeDetectionDataset

        if self.model is None:
            self.build_unet()

        X_train, y_train = train_data
        X_val, y_val = val_data

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(self.models_dir, f"unet_training_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_dice_coef',
                mode='max',
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                verbose=1, min_lr=1e-6,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15,
                verbose=1, restore_best_weights=True,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(output_dir, 'logs')
            ),
        ]

        if use_generator:
            dataset = EdgeDetectionDataset(None, None)
            train_gen = dataset.get_data_generator(
                X_train, y_train, batch_size=batch_size, augment=True
            )
            val_gen = dataset.get_data_generator(
                X_val, y_val, batch_size=batch_size, augment=False
            )

            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size

            history = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )

        with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        self.plot_training_history(history, output_dir)

        return history, output_dir

    def plot_training_history(self, history, output_dir: str):
        """Plot and save training history graphs."""
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

    def load_model(self, model_path: str):
        """Load a trained model from a file.

        Parameters
        ----------
        model_path : str
            Path to the saved model file.

        Returns
        -------
        keras.Model
            The loaded model.
        """
        _, keras, _ = _import_tf()

        custom_objects = {
            'dice_loss': self.dice_loss,
            'dice_coef': self.dice_coef,
        }

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded from %s", model_path)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model."""
        if self.model is None:
            raise ValueError("Model not built or loaded yet")
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not built or loaded yet")

        results = self.model.evaluate(X_test, y_test)
        logger.info("Test Loss: %.4f", results[0])
        logger.info("Test Dice Coefficient: %.4f", results[1])

        return results
