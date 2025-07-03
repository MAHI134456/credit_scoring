# tests/test_data_processing.py

import pandas as pd
import pytest

def test_model_ready_shape_and_nulls():
    """
    Sanity check:
    - Data loads
    - Shape matches expected rows and reasonable columns
    - No missing values
    """
    # Path to your saved processed data
    df = pd.read_csv('data/processed/model_ready.csv')

    # Replace these with your expected shape
    expected_rows = 95662
    min_expected_cols = 10  # Be flexible if you change pipeline later
    max_expected_cols = 200  # Should never blow up to 296k again

    assert df.shape[0] == expected_rows, f"Row count mismatch: {df.shape[0]} vs {expected_rows}"
    assert min_expected_cols <= df.shape[1] <= max_expected_cols, f"Unexpected number of columns: {df.shape[1]}"
    assert not df.isnull().values.any(), "Found missing values in model_ready.csv"

    print(f"✅ TEST PASSED: Shape {df.shape}, no NaNs found.")

