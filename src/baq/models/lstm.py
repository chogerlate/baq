"""
LSTM Model Module for Time Series Forecasting.

This module implements a deep learning model for PM2.5 forecasting using LSTM architecture.
The model supports both single-step and multi-step forecasting with configurable
architecture and training parameters.

Features:
- Dual-layer LSTM architecture
- Configurable hyperparameters
- Dropout for regularization
- Learning rate scheduling
- Model checkpointing
- Early stopping

Example:
    >>> model = LSTMForecaster(input_shape=(24, 50))
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
    >>> predictions = model.predict(X_test)
"""

from typing import Tuple, Union, List, Dict, Any
from pathlib import Path
import numpy as np
import logging
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
logger = logging.getLogger(__name__)

class LSTMForecaster:
    """
    LSTM-based model for time series forecasting.

    This class implements a deep learning model with:
    - Two LSTM layers with configurable units
    - Dropout for regularization
    - Dense output layer
    - Adam optimizer with learning rate scheduling
    - Early stopping and model checkpointing

    Attributes:
        input_shape (Tuple[int, int]): Shape of input data (sequence_length, n_features)
        lstm_units (Tuple[int, int]): Number of units in first and second LSTM layers
        dropout_rate (float): Dropout rate between LSTM layers
        learning_rate (float): Initial learning rate for Adam optimizer
        checkpoint_path (Union[str, Path]): Path to save best model weights
        early_stopping_patience (int): Epochs to wait before early stopping
        reduce_lr_patience (int): Epochs to wait before reducing learning rate
        model (Model): Keras model instance
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        checkpoint_path: Union[str, Path] = "best_lstm.h5",
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
    ):
        """
        Initialize the LSTM forecaster.

        Args:
            input_shape: Tuple of (sequence_length, n_features)
            lstm_units: Units in first and second LSTM layers
            dropout_rate: Dropout rate between LSTM layers
            learning_rate: Initial learning rate
            checkpoint_path: Path to save best model
            early_stopping_patience: Epochs before early stopping
            reduce_lr_patience: Epochs before reducing learning rate

        Note:
            The model is built and compiled upon initialization
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.checkpoint_path = str(checkpoint_path)
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        
        logger.info(f"Initializing LSTM model with input shape: {input_shape}")
        self.model = self._build_model()
        logger.info("Model built and compiled successfully")

    def _build_model(self) -> Model:
        """
        Build and compile the LSTM model.

        Returns:
            Compiled Keras model

        Architecture:
            1. Input layer
            2. First LSTM layer with return sequences
            3. Dropout layer
            4. Second LSTM layer
            5. Dense output layer
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer
        x = LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            name="lstm_1"
        )(inputs)
        
        # Dropout for regularization
        x = Dropout(self.dropout_rate, name="dropout")(x)
        
        # Second LSTM layer
        x = LSTM(
            units=self.lstm_units[1],
            name="lstm_2"
        )(x)
        
        # Output layer
        outputs = Dense(1, name="output")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name="lstm_forecaster")
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        
        return model

    def _create_callbacks(self) -> List[Any]:
        """
        Create training callbacks for the model.

        Returns:
            List of Keras callbacks:
            - ModelCheckpoint: Save best model
            - EarlyStopping: Stop if not improving
            - ReduceLROnPlateau: Reduce learning rate if plateau

        Note:
            All callbacks monitor validation loss
        """
        return [
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                mode="min",
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.reduce_lr_patience,
                mode="min",
                verbose=1
            )
        ]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        shuffle: bool = False,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features (n_samples, seq_length, n_features)
            y_train: Training targets
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            verbose: Verbosity level
            **kwargs: Additional arguments for model.fit()

        Returns:
            Training history with metrics

        Note:
            Uses early stopping and learning rate reduction
        """
        logger.info("Starting model training...")
        callbacks = self._create_callbacks()
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )
        
        logger.info("Training completed")
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.

        Args:
            X: Input features (n_samples, seq_length, n_features)

        Returns:
            Model predictions

        Note:
            Returns flattened predictions for single-step forecasting
        """
        logger.info(f"Generating predictions for {len(X)} samples")
        return self.model.predict(X).reshape(-1)

    def save_weights(self, path: Union[str, Path]) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save weights file
        """
        path = str(path)
        logger.info(f"Saving model weights to: {path}")
        self.model.save_weights(path)

    def load_weights(self, path: Union[str, Path]) -> None:
        """
        Load model weights from file.

        Args:
            path: Path to weights file
        """
        path = str(path)
        logger.info(f"Loading model weights from: {path}")
        self.model.load_weights(path)

    def summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
            String representation of model architecture
        """
        return self.model.summary()