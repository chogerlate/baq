"""
Data Processing Pipeline for Time Series Data

This module provides preprocessing utilities for time series data using feature-engine:
- Standardized pipelines for numerical feature scaling and transformation
- Custom transformers for handling time series specific operations
- Preprocessing class that can be saved and reused during inference
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler


from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import MeanMedianImputer
from feature_engine.transformation import YeoJohnsonTransformer, LogTransformer
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder
from feature_engine.creation import CyclicalFeatures
from feature_engine.selection import DropFeatures
from feature_engine.scaling import MeanNormalizationScaler

def create_target_scaler():
    return MinMaxScaler()

def series_to_numpy(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series to a numpy array suitable for machine learning models.
    
    Args:
        series: Input pandas Series
        
    Returns:
        np.ndarray: The series values as a numpy array with shape (n_samples, 1)
    """
    # Convert to numpy and reshape to 2D array required by sklearn
    return series.values.reshape(-1, 1)

def numpy_to_series(array: np.ndarray, index=None, name=None) -> pd.Series:
    """
    Convert a numpy array back to a pandas Series.
    
    Args:
        array: Input numpy array
        index: Optional index for the Series
        name: Optional name for the Series
        
    Returns:
        pd.Series: The array values as a pandas Series
    """
    # Ensure the array is 1D
    if array.ndim > 1:
        array = array.ravel()
    
    return pd.Series(array, index=index, name=name)

class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for time series data.
    
    Handles:
    - Missing value imputation
    - Outlier capping
    - Feature scaling
    - Feature transformation
    - Cyclical encoding for temporal features
    """
    
    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        cyclical_features: List[str] = None,
        drop_features: List[str] = None,
        log_transform_features: List[str] = None,
        use_yeo_johnson: bool = True,
        use_scaling: bool = True,
        preserve_columns: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            cyclical_features: List of cyclical temporal features (hour, day, month, etc.)
            drop_features: List of features to drop
            log_transform_features: List of features to apply log transformation
            use_yeo_johnson: Whether to use Yeo-Johnson transformation
            use_scaling: Whether to apply mean normalization scaling
            preserve_columns: Whether to keep columns not mentioned in any feature list
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.cyclical_features = cyclical_features or []
        self.drop_features = drop_features or []
        self.log_transform_features = log_transform_features or []
        self.use_yeo_johnson = use_yeo_johnson
        self.use_scaling = use_scaling
        self.preserve_columns = preserve_columns
        
        # Initialize feature lists that will be populated during fit
        self.regular_features = []
        self.low_var_features = []
        
        # Initialize pipelines
        self.numerical_pipeline = None
        self.low_var_pipeline = None
        self.categorical_pipeline = None
        self.cyclical_pipeline = None
        self.full_pipeline = None
        
        # Create pipelines
        self._create_pipelines()
        
    def _create_pipelines(self):
        """Create the preprocessing pipelines"""
        
        if self.numerical_features:
            # Split numerical features into regular and low-variation features
            # Low variation features often have many zeros (like precipitation)
            low_var_features = []
            regular_features = []
            
            # This will be populated during fit
            self.low_var_features = []
            self.regular_features = []
            
            # Pipeline for numerical features with regular variation
            numerical_steps = []
            
            # We'll only add steps if we have features to process
            self.numerical_pipeline = None
            self.low_var_pipeline = None
        else:
            self.numerical_pipeline = None
            self.low_var_pipeline = None
            self.low_var_features = []
            self.regular_features = []
        
        # Pipeline for categorical features
        self.categorical_pipeline = None
        if self.categorical_features:
            self.categorical_pipeline = Pipeline([
                ('encoder', OneHotEncoder(
                    variables=self.categorical_features,
                    drop_last=True
                ))
            ])
        
        # Pipeline for cyclical features
        self.cyclical_pipeline = None
        if self.cyclical_features:
            self.cyclical_pipeline = Pipeline([
                ('cyclical_encoder', CyclicalFeatures(
                    variables=self.cyclical_features,
                    drop_original=True
                ))
            ])
            
        # Create a pipeline that drops specified features
        self.drop_pipeline = None
        if self.drop_features:
            self.drop_pipeline = Pipeline([
                ('drop_features', DropFeatures(features_to_drop=self.drop_features))
            ])
            
        # Store pipeline configurations for later
        self.pipelines = []
        
    def _identify_low_variation_features(self, X):
        """
        Identify features with low variation where IQR would be problematic.
        This is especially important for features like precipitation which are often zero.
        """
        self.low_var_features = []
        self.regular_features = []
        
        for col in self.numerical_features:
            if col not in X.columns:
                continue
                
            # Check if column has enough variation for IQR
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            
            # If IQR is very small or zero, it's a low variation feature
            if iqr < 0.001 or (X[col] == 0).mean() > 0.75:
                self.low_var_features.append(col)
            else:
                self.regular_features.append(col)
                
        return self.low_var_features, self.regular_features

    def fit(self, X: pd.DataFrame, y=None) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessing pipeline to the data.
        
        Args:
            X: Input DataFrame
            y: Target variable (not used, included for compatibility)
            
        Returns:
            self
        """
        # Verify types of features to avoid errors
        if self.numerical_features:
            # Ensure numerical features are actually numerical
            non_numeric = [col for col in self.numerical_features 
                           if col in X.columns and not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric:
                self.numerical_features = [col for col in self.numerical_features if col not in non_numeric]
                print(f"Warning: Removed non-numeric columns from numerical features: {non_numeric}")
            
            # Make sure we only work with features present in the dataframe
            self.numerical_features = [col for col in self.numerical_features if col in X.columns]
            
            # Identify low variation features if we have numerical features
            if self.numerical_features:
                self._identify_low_variation_features(X)
                if self.low_var_features:
                    print(f"Low variation features detected (will use special handling): {self.low_var_features}")
            
                # Create pipelines now that we know which features are regular vs low variation
                if self.regular_features:
                    # Create pipeline for regular features
                    numerical_steps = []
                    
                    # Add imputation
                    numerical_steps.append(
                        ('imputer', MeanMedianImputer(
                            imputation_method='median',
                            variables=self.regular_features
                        ))
                    )
                    
                    # Add outlier capping
                    numerical_steps.append(
                        ('outlier_capper', Winsorizer(
                            capping_method='iqr',
                            tail='both',
                            fold=1.5,
                            variables=self.regular_features
                        ))
                    )
                    
                    # Add log transformation for specified features
                    log_vars = [f for f in self.log_transform_features if f in self.regular_features]
                    if log_vars:
                        numerical_steps.append(
                            ('log_transformer', LogTransformer(
                                variables=log_vars
                            ))
                        )
                    
                    # Add Yeo-Johnson transformation
                    if self.use_yeo_johnson:
                        numerical_steps.append(
                            ('yeo_johnson', YeoJohnsonTransformer(
                                variables=self.regular_features
                            ))
                        )
                    
                    # Add mean normalization scaler
                    if self.use_scaling:
                        numerical_steps.append(
                            ('scaler', MeanNormalizationScaler(
                                variables=self.regular_features
                            ))
                        )
                    
                    # Create the pipeline
                    self.numerical_pipeline = Pipeline(steps=numerical_steps)
                    
                    # Fit the pipeline
                    try:
                        self.numerical_pipeline.fit(X[self.regular_features])
                    except Exception as e:
                        print(f"Warning: Error fitting numerical pipeline: {str(e)}")
                        self.numerical_pipeline = None
                else:
                    self.numerical_pipeline = None
                
                # Create pipeline for low variation features
                if self.low_var_features:
                    low_var_steps = []
                    
                    # Add imputation
                    low_var_steps.append(
                        ('imputer', MeanMedianImputer(
                            imputation_method='median',
                            variables=self.low_var_features
                        ))
                    )
                    
                    # Add scaling for low variation features too
                    if self.use_scaling:
                        low_var_steps.append(
                            ('scaler', MeanNormalizationScaler(
                                variables=self.low_var_features
                            ))
                        )
                    
                    # Create the pipeline
                    self.low_var_pipeline = Pipeline(steps=low_var_steps)
                    
                    # Fit the pipeline
                    try:
                        self.low_var_pipeline.fit(X[self.low_var_features])
                    except Exception as e:
                        print(f"Warning: Error fitting low variation pipeline: {str(e)}")
                        self.low_var_pipeline = None
                else:
                    self.low_var_pipeline = None
        
        # Fit categorical pipeline
        if self.categorical_features:
            # Filter to columns in the dataframe
            cats_in_df = [col for col in self.categorical_features if col in X.columns]
            if cats_in_df:
                self.categorical_pipeline = Pipeline([
                    ('encoder', OneHotEncoder(
                        variables=cats_in_df,
                        drop_last=True
                    ))
                ])
                self.categorical_pipeline.fit(X[cats_in_df])
            else:
                self.categorical_pipeline = None
        
        # Fit cyclical pipeline
        if self.cyclical_features:
            # Filter to columns in the dataframe
            cyc_in_df = [col for col in self.cyclical_features if col in X.columns]
            if cyc_in_df:
                self.cyclical_pipeline = Pipeline([
                    ('cyclical_encoder', CyclicalFeatures(
                        variables=cyc_in_df,
                        drop_original=True
                    ))
                ])
                self.cyclical_pipeline.fit(X[cyc_in_df])
            else:
                self.cyclical_pipeline = None
        
        # Update pipelines list
        self.pipelines = []
        if self.numerical_pipeline:
            self.pipelines.append(('numerical', self.numerical_pipeline))
        if self.low_var_pipeline:
            self.pipelines.append(('low_var', self.low_var_pipeline))
        if self.categorical_pipeline:
            self.pipelines.append(('categorical', self.categorical_pipeline))
        if self.cyclical_pipeline:
            self.pipelines.append(('cyclical', self.cyclical_pipeline))
        if self.drop_pipeline:
            self.pipelines.append(('drop', self.drop_pipeline))
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted pipeline.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        X_copy = X.copy()
        
        # Get list of all features that will be transformed
        all_features = set()
        all_features.update(self.regular_features if hasattr(self, 'regular_features') else [])
        all_features.update(self.low_var_features if hasattr(self, 'low_var_features') else [])
        all_features.update(self.categorical_features)
        all_features.update(self.cyclical_features)
        
        # Get columns to preserve (not in any feature list)
        if self.preserve_columns:
            preserve_cols = [col for col in X_copy.columns if col not in all_features]
            preserved_data = X_copy[preserve_cols].copy()
        
        result_dfs = []
        
        # Apply numerical pipeline to regular features
        if self.numerical_pipeline and hasattr(self, 'regular_features') and self.regular_features:
            # Get numerical features that exist in the dataframe
            cols_in_df = [col for col in self.regular_features if col in X_copy.columns]
            
            if cols_in_df:
                try:
                    # Transform the data
                    X_numerical = self.numerical_pipeline.transform(X_copy[cols_in_df])
                    
                    # Convert to DataFrame with original column names
                    if isinstance(X_numerical, np.ndarray):
                        X_numerical_df = pd.DataFrame(X_numerical, 
                                                     index=X_copy.index, 
                                                     columns=cols_in_df)
                    else:
                        X_numerical_df = X_numerical
                    
                    # Add to results
                    result_dfs.append(X_numerical_df)
                    
                except Exception as e:
                    print(f"Warning: Error transforming with numerical pipeline: {str(e)}")
                    # Just keep the original columns
                    result_dfs.append(X_copy[cols_in_df])
        
        # Apply low variation pipeline
        if self.low_var_pipeline and hasattr(self, 'low_var_features') and self.low_var_features:
            # Get features that exist in the dataframe
            cols_in_df = [col for col in self.low_var_features if col in X_copy.columns]
            
            if cols_in_df:
                try:
                    # Transform the data
                    X_low_var = self.low_var_pipeline.transform(X_copy[cols_in_df])
                    
                    # Convert to DataFrame with original column names
                    if isinstance(X_low_var, np.ndarray):
                        X_low_var_df = pd.DataFrame(X_low_var, 
                                                  index=X_copy.index, 
                                                  columns=cols_in_df)
                    else:
                        X_low_var_df = X_low_var
                    
                    # Add to results
                    result_dfs.append(X_low_var_df)
                    
                except Exception as e:
                    print(f"Warning: Error transforming with low variation pipeline: {str(e)}")
                    # Just keep the original columns
                    result_dfs.append(X_copy[cols_in_df])
                
        # Apply categorical pipeline
        if self.categorical_pipeline and self.categorical_features:
            # Get categorical features that exist in the dataframe
            cats_in_df = [col for col in self.categorical_features if col in X_copy.columns]
            
            if cats_in_df:
                try:
                    # Transform the data
                    X_categorical = self.categorical_pipeline.transform(X_copy[cats_in_df])
                    # Add to results
                    result_dfs.append(X_categorical)
                except Exception as e:
                    print(f"Warning: Error transforming with categorical pipeline: {str(e)}")
                    # Keep original columns
                    result_dfs.append(X_copy[cats_in_df])
            
        # Apply cyclical pipeline
        if self.cyclical_pipeline and self.cyclical_features:
            # Get cyclical features that exist in the dataframe
            cyc_in_df = [col for col in self.cyclical_features if col in X_copy.columns]
            
            if cyc_in_df:
                try:
                    # Transform the data
                    X_cyclical = self.cyclical_pipeline.transform(X_copy[cyc_in_df])
                    # Add to results
                    result_dfs.append(X_cyclical)
                except Exception as e:
                    print(f"Warning: Error transforming with cyclical pipeline: {str(e)}")
                    # Keep original columns
                    result_dfs.append(X_copy[cyc_in_df])
        
        # Add preserved columns if needed
        if self.preserve_columns and 'preserved_data' in locals() and not preserved_data.empty:
            result_dfs.append(preserved_data)
            
        # Combine all processed dataframes
        if result_dfs:
            if len(result_dfs) == 1:
                X_result = result_dfs[0]
            else:
                # Check for duplicate columns across dataframes
                all_columns = []
                for df in result_dfs:
                    all_columns.extend(df.columns.tolist())
                
                # If there are duplicates, print a warning
                if len(all_columns) != len(set(all_columns)):
                    duplicates = set([col for col in all_columns if all_columns.count(col) > 1])
                    print(f"Warning: Duplicate columns found after preprocessing: {duplicates}")
                
                # Combine all dataframes
                X_result = pd.concat(result_dfs, axis=1)
        else:
            # If nothing was processed, return the original dataframe
            X_result = X_copy
            
        return X_result
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data.
        
        Args:
            X: Input DataFrame
            y: Target variable (not used, included for compatibility)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)


def create_time_series_processor(
    df: pd.DataFrame,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
    cyclical_features: List[str] = None,
    drop_features: List[str] = None,
    log_transform_features: List[str] = None,
    use_yeo_johnson: bool = True,
    use_scaling: bool = True,
    preserve_columns: bool = True
) -> TimeSeriesPreprocessor:
    """
    Create a preprocessor for time series data.
    
    Args:
        df: Sample DataFrame to automatically detect feature types
        numerical_features: List of numerical features (if None, will be auto-detected)
        categorical_features: List of categorical features (if None, will be auto-detected)
        cyclical_features: List of cyclical features (hour, day, month, etc.)
        drop_features: List of features to drop
        log_transform_features: List of features to apply log transformation
        use_yeo_johnson: Whether to use Yeo-Johnson transformation
        use_scaling: Whether to apply mean normalization scaling
        preserve_columns: Whether to preserve columns not in any feature list
        
    Returns:
        TimeSeriesPreprocessor: Configured preprocessor
    """
    # Auto-detect features if not specified
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    processor = TimeSeriesPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        cyclical_features=cyclical_features,
        drop_features=drop_features,
        log_transform_features=log_transform_features,
        use_yeo_johnson=use_yeo_johnson,
        use_scaling=use_scaling,
        preserve_columns=preserve_columns
    )
    
    return processor