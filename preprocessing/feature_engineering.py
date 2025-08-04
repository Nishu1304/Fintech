# preprocessing/feature_engineering.py

import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds commonly useful derived features.

    Examples:
    ---------
    - If 'Transaction Date' exists, extract day/month/year
    - Create 'Balance to Amount Ratio' if applicable

    Returns:
    --------
    df : pd.DataFrame
    """
    df = df.copy()

    # Example 1: Date extraction
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
        except:
            continue

    # Example 2: Ratio of balance and transaction amount
    if 'balance' in df.columns and 'transaction_amount' in df.columns:
        try:
            df['balance_to_amount_ratio'] = df['balance'] / (df['transaction_amount'] + 1e-5)
        except:
            pass

    return df
