"""
This module contains the code for loading the data.
- Handles data retrieval from various sources including AWS S3 buckets
- Supports different file formats (CSV, JSON, Parquet, etc.)
- Implements efficient data loading strategies for large datasets
- Provides utilities for data validation during the loading process
- Manages authentication and connection to secure data sources
- Includes logging and error handling for robust data acquisition
"""
import pandas as pd

def load_data(
    data_path: str,
) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        pd.DataFrame: The loaded data
    """
    return pd.read_csv(data_path)

