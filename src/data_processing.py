# src/data_processing.py

import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    """
    return pd.read_csv(filepath)

def cap_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Caps outliers in a numerical Series using the IQR method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + factor * IQR
    lower_bound = Q1 - factor * IQR

    return np.clip(series, lower_bound, upper_bound)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop unhelpful columns
    - Cap outliers for Amount and Value
    - Log-transform skewed numerical features
    """
    # Drop columns
    drop_cols = ['CountryCode']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Cap outliers
    df['Amount_capped'] = cap_outliers(df['Amount'])
    df['Value_capped'] = cap_outliers(df['Value'])

    # Log-transform
    df['log_Amount'] = np.log1p(df['Amount_capped'])
    df['log_Value'] = np.log1p(df['Value_capped'])

    return df

def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save cleaned data to CSV.
    """
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    # Example run: python src/data_processing.py
    raw_df = load_data("data/raw/data.csv")
    clean_df = preprocess_data(raw_df)
    save_processed_data(clean_df, "data/processed/credit_data_clean.csv")
    print("✅ Data preprocessing complete. Clean data saved.")
