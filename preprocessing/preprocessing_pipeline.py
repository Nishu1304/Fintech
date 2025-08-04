from .missing_value_handler import handle_missing_values
from .enc_stan import standardize_numerical, encode_categoricals
from .deduplication import remove_duplicates
from .feature_engineering import add_derived_features


def run_preprocessing_pipeline(df):
    # Step 1: Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Handle missing values
    df, rejected_columns, rejection_reason = handle_missing_values(df)
    if df is None:
        return None, rejection_reason

    # Step 3: Add derived features
    df = add_derived_features(df)

    # Step 4: Encode categorical columns
    df = encode_categoricals(df)

    # Step 5: Standardize numerical columns
    df = standardize_numerical(df)

    # Step 6: Drop unnecessary identifier columns
    if 'Transaction ID' in df.columns:
        df.drop(columns=['Transaction ID'], inplace=True)

    return df, None
