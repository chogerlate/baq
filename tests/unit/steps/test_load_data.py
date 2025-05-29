import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import tempfile
import os

# Import the function to be tested
# This will work because of tests/context.py
from baq.steps.load_data import load_data

@pytest.fixture
def temp_csv_file_factory():
    """Factory fixture to create temporary CSV files for testing."""
    created_files = []

    def _create_temp_csv(data_content, filename_suffix=""):
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix=f'_{filename_suffix}.csv',
            newline='' # Ensure consistent line endings
        )
        temp_file.write(data_content)
        temp_file.close() # Close the file so pandas can open it
        created_files.append(temp_file.name)
        return temp_file.name

    yield _create_temp_csv

    # Teardown: remove all created temporary files
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_load_data_successful(temp_csv_file_factory):
    """Test loading a valid CSV file."""
    csv_content = "col1,col2\n1,A\n2,B\n3,C"
    file_path = temp_csv_file_factory(csv_content, "valid_data")
    
    expected_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['A', 'B', 'C']
    })
    
    loaded_df = load_data(file_path)
    
    assert_frame_equal(loaded_df, expected_df)

def test_load_data_file_not_found():
    """Test loading a non-existent CSV file."""
    non_existent_path = "path/to/non_existent_file.csv"
    
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_path)

def test_load_data_empty_csv(temp_csv_file_factory):
    """Test loading an empty CSV file."""
    # CSV with headers but no data rows
    csv_content_headers_only = "header1,header2\n"
    file_path_headers_only = temp_csv_file_factory(csv_content_headers_only, "empty_headers")
    
    expected_df_headers_only = pd.DataFrame(columns=['header1', 'header2'])
    loaded_df_headers_only = load_data(file_path_headers_only)
    
    assert_frame_equal(loaded_df_headers_only, expected_df_headers_only)

    # Completely empty CSV (no headers, no data)
    # Note: pandas behavior with completely empty files can be tricky.
    # pd.read_csv on a truly empty file might raise an EmptyDataError or return an empty DataFrame
    # depending on the pandas version and parameters.
    # For this example, let's assume it should return an empty DataFrame.
    # If it raises an error, the test should be adjusted to expect that error.
    csv_content_truly_empty = ""
    file_path_truly_empty = temp_csv_file_factory(csv_content_truly_empty, "truly_empty")
    
    try:
        loaded_df_truly_empty = load_data(file_path_truly_empty)
        # Expect an empty DataFrame if no error is raised
        assert loaded_df_truly_empty.empty, "DataFrame should be empty for a truly empty CSV file"
    except pd.errors.EmptyDataError:
        # This is also acceptable behavior for a truly empty file.
        # If this is the expected behavior, the test can be:
        # with pytest.raises(pd.errors.EmptyDataError):
        #     load_data(file_path_truly_empty)
        pass # Test passes if EmptyDataError is raised


def test_load_data_different_types(temp_csv_file_factory):
    """Test loading a CSV with various data types."""
    csv_content = "int_col,float_col,str_col,bool_col\n1,1.1,apple,True\n2,2.2,banana,False\n0,,,"
    file_path = temp_csv_file_factory(csv_content, "typed_data")

    # Note: pandas reads everything as object if there are mixed types or NAs that prevent inference.
    # Or it infers types as best as it can.
    # Be explicit with expected types if your load_data has specific dtype handling,
    # otherwise, test based on default pd.read_csv behavior.
    loaded_df = load_data(file_path)

    assert loaded_df['int_col'].iloc[0] == 1
    assert loaded_df['float_col'].iloc[0] == 1.1
    assert loaded_df['str_col'].iloc[0] == 'apple'
    # Pandas by default might not infer 'True'/'False' strings to booleans without explicit converters.
    # If 'bool_col' is read as string 'True', this test is fine.
    # If it's expected to be boolean True, pd.read_csv might need na_values or converters.
    # The current load_data function is a simple pd.read_csv.
    assert loaded_df['bool_col'].iloc[0] == True # This assumes pandas infers it correctly or it's 'True' string

    # Check for NaN handling (empty string for float_col, str_col, bool_col in the last row)
    assert pd.isna(loaded_df['float_col'].iloc[2])
    assert pd.isna(loaded_df['str_col'].iloc[2]) 
    assert pd.isna(loaded_df['bool_col'].iloc[2])


# Add more tests if load_data is expected to handle other parameters,
# encodings, or specific error conditions.
