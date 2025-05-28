import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class TimeSeriesDataProcessor:
    """
    All-in-one processor for PM2.5 forecasting:
      1) cleaning (resample, linear-interpolate, seasonal-median fill)
      2) encoding (weather code → integer)
      3) feature engineering (time features, lags, rolling means, AQI tier)
      4) splitting (train/val/test)
      5) scaling (MinMaxScale features & target)
    """

    def __init__(
        self,
        target_col: str = "pm2_5_(μg/m³)",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42,
    ):
        self.target_col     = target_col
        self.train_ratio    = train_ratio
        self.val_ratio      = val_ratio
        self.test_ratio     = test_ratio
        self.random_state   = random_state

        # internal objects
        self.weather_encoder = LabelEncoder()
        self.feature_scaler  = MinMaxScaler()
        self.target_scaler   = MinMaxScaler()

        # to be set at fit time
        self.feature_cols: list[str] = []

    def fit_transform(
        self,
        raw: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series
    ]:
        """
        Clean, engineer, split and scale the input DataFrame.

        Returns:
          X_train, y_train, X_val, y_val, X_test, y_test
        """
        # 1) clean & engineer
        df = self._clean(raw)
        df = self._feature_engineer(df)

        # 2) split
        train_df, val_df, test_df = self._split(df)

        # 3) define feature columns & fit scalers
        self.feature_cols = [c for c in df.columns if c != self.target_col]
        self.feature_scaler.fit(train_df[self.feature_cols])
        self.target_scaler.fit(train_df[[self.target_col]])

        # 4) apply scaling
        def _scale(df_: pd.DataFrame):
            X = pd.DataFrame(
                self.feature_scaler.transform(df_[self.feature_cols]),
                index=df_.index, columns=self.feature_cols
            )
            y = pd.Series(
                self.target_scaler.transform(df_[[self.target_col]]).ravel(),
                index=df_.index, name=self.target_col
            )
            return X, y

        X_train, y_train = _scale(train_df)
        X_val,   y_val   = _scale(val_df)
        X_test,  y_test  = _scale(test_df)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Clean, engineer and scale new data (no re-fitting).
        Returns X only.
        """
        df = self._clean(raw)
        df = self._feature_engineer(df)
        return pd.DataFrame(
            self.feature_scaler.transform(df[self.feature_cols]),
            index=df.index, columns=self.feature_cols
        )
    
    def inverse_transform_target(self, y: pd.Series) -> pd.Series:
        """
        Inverse transform the target variable to its original scale.
        This is useful for interpreting the model's predictions.
        Args:
            y: Series of scaled target values
        Returns:
            Series of original target values
        """
        arr = self.target_scaler.inverse_transform(
            y.values.reshape(-1, 1)
        ).ravel()
        return pd.Series(arr, index=y.index, name=y.name)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) normalize column names
        df.columns = (
            df.columns
            .str.replace('\s*\(\)', '', regex=True)
            .str.replace(' ', '_', regex=False)
        )

        # 2) datetime index
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time").sort_index()

        # 3) drop unwanted columns
        to_drop = [
            "carbon_dioxide_ppm",
            "methane_μg/m³",
            "snowfall_cm",
            "snow_depth_m"
        ]
        df = df.drop(columns=to_drop, errors="ignore")

        # 4) resample & interpolate
        df = df.resample("1h").asfreq().interpolate("linear")

        # 5) seasonal-median fill
        df = self._fill_seasonal_median(df)

        # 6) encode weather_code_* → weather_code
        weather_cols = [c for c in df.columns if c.startswith("weather_code")]
        if weather_cols:
            orig = weather_cols[0]
            df["weather_code"] = self.weather_encoder.fit_transform(
                df[orig].astype(str)
            )
            df = df.drop(columns=[orig])

        return df

    def _fill_seasonal_median(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = df.index
        df["hour"], df["day"], df["month"], df["year"] = (
            idx.hour, idx.day, idx.month, idx.year
        )
        feats = df.columns.difference(["hour", "day", "month", "year"])
        for col in feats:
            missing = df[col].isna()
            for ts in df[missing].index:
                m, d, h, yr = ts.month, ts.day, ts.hour, ts.year
                mask = (
                    (df["month"] == m) &
                    (df["day"] == d) &
                    (df["hour"] == h) &
                    (df["year"] != yr) &
                    df[col].notna()
                )
                med = df.loc[mask, col].median()
                if pd.notna(med):
                    df.at[ts, col] = med
        return df.drop(columns=["hour", "day", "month", "year"])

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # time features
        df["hour"]      = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"]     = df.index.month
        df["is_weekend"]= df["dayofweek"].isin([5,6]).astype(int)
        df["is_night"]  = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
        df["sin_hour"]  = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"]  = np.cos(2 * np.pi * df["hour"] / 24)

        # lag features (1,3,6,12,24)
        for lag in (1,3,6,12,24):
            df[f"{self.target_col}_lag{lag}"] = df[self.target_col].shift(lag)

        # rolling means (3,6,12)
        for w in (3,6,12):
            df[f"{self.target_col}_rollmean{w}"] = df[self.target_col].rolling(w).mean()

        # AQI tier based on pm2.5
        df["pm2_5_tier"] = df[self.target_col].apply(self._pm25_to_aqi)

        return df.dropna()

    @staticmethod
    def _pm25_to_aqi(x: float) -> int:
        if   x <=  12.0: return 0
        elif x <=  35.4: return 1
        elif x <=  55.4: return 2
        elif x <= 150.4: return 3
        elif x <= 250.4: return 4
        else:            return 5

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n   = len(df)
        i1  = int(n * self.train_ratio)
        i2  = i1  + int(n * self.val_ratio)
        return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
