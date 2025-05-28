from typing import Tuple, Union, List
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path

def create_lstm_model(input_shape: Tuple[int, int]) -> Model:
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def create_lstm_callbacks(
    checkpoint_path: Union[str, Path] = "best_lstm.keras",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
) -> List:
    return [
        ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_loss", verbose=1),
        EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=reduce_lr_patience, min_lr=1e-6, verbose=1),
    ]