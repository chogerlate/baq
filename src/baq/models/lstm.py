from typing import Tuple, Union, List, Dict, Any
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import numpy as np


class LSTMForecaster:
    """LSTM model for time series forecasting."""
    
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
        Initialize the LSTM model.

        Args:
            input_shape: Tuple of (sequence_length, n_features)
            lstm_units: Tuple of units for first and second LSTM layers
            dropout_rate: Dropout rate between LSTM layers
            learning_rate: Initial learning rate for Adam optimizer
            checkpoint_path: Path to save best model weights
            early_stopping_patience: Number of epochs to wait before early stopping
            reduce_lr_patience: Number of epochs to wait before reducing learning rate
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        
        # Build and compile model
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build and compile the LSTM model."""
        inputs = Input(shape=self.input_shape)
        x = LSTM(self.lstm_units[0], return_sequences=True)(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(self.lstm_units[1])(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        return model
    
    def _create_callbacks(self) -> List:
        """Create training callbacks."""
        return [
            ModelCheckpoint(
                str(self.checkpoint_path),
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=self.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ),
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
            X_train: Training features of shape (n_samples, seq_length, n_features)
            y_train: Training targets
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            verbose: Verbosity level
            **kwargs: Additional arguments passed to model.fit()

        Returns:
            Training history
        """
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
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features of shape (n_samples, seq_length, n_features)

        Returns:
            Model predictions
        """
        return self.model.predict(X).reshape(-1)
    
    def save_weights(self, path: Union[str, Path]) -> None:
        """Save model weights."""
        self.model.save_weights(str(path))
    
    def load_weights(self, path: Union[str, Path]) -> None:
        """Load model weights."""
        self.model.load_weights(str(path))