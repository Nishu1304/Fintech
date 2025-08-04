import pandas as pd
import logging

logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame,
                          row_missing_reject_threshold: float = 0.3,
                          row_drop_threshold: float = 0.05,
                          col_reject_threshold: float = 0.15,
                          impute_threshold_low: float = 0.05,
                          impute_threshold_high: float = 0.30
                          ) -> tuple:
    """
    Handles missing data with rules:
    - Reject entire dataframe if more than row_missing_reject_threshold fraction of rows
      have any missing values.
    - Drop rows with missing fraction > row_drop_threshold.
    - Drop columns with missing fraction > col_reject_threshold.
    - Impute columns with missing fraction between impute_threshold_low and impute_threshold_high,
      adding imputation flag columns.

    Returns:
    --------
    cleaned_df : pd.DataFrame or None
        Cleaned dataframe or None if rejected.
    rejected_columns : list
        List of columns dropped.
    rejected_reason : str or None
        Reason for rejection if dataframe is rejected.
    """
    rows_with_missing = df.isnull().any(axis=1).mean()
    logger.info(f"Rows with any missing values: {rows_with_missing:.2%}")

    if rows_with_missing > row_missing_reject_threshold:
        reason = (f"Dataframe rejected: {rows_with_missing:.2%} rows have missing values "
                  f"exceeding threshold {row_missing_reject_threshold:.2%}")
        logger.warning(reason)
        return None, [], reason

    # Drop rows with too many missing values
    rows_before = len(df)
    df = df[df.isnull().mean(axis=1) <= row_drop_threshold].copy()
    rows_after = len(df)
    logger.info(f"Dropped {rows_before - rows_after} rows with missing fraction > {row_drop_threshold:.2%}")

    # Identify columns to reject
    rejected_columns = [col for col in df.columns if df[col].isnull().mean() > col_reject_threshold]
    if rejected_columns:
        logger.info(f"Dropping columns due to missing fraction > {col_reject_threshold:.2%}: {rejected_columns}")
    df.drop(columns=rejected_columns, inplace=True)

    # Impute columns and add imputation flags
    for col in df.columns:
        missing_frac = df[col].isnull().mean()
        if impute_threshold_low < missing_frac <= impute_threshold_high:
            flag_col = f"{col}_imputed"
            df[flag_col] = 0  # Initialize flag

            missing_indices = df[df[col].isnull()].index
            n_missing = len(missing_indices)

            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df.loc[missing_indices, col] = median_val
            elif df[col].dtype == 'object':
                df.loc[missing_indices, col] = "Unknown"

            df.loc[missing_indices, flag_col] = 1
            logger.info(f"Imputed {n_missing} missing values in column '{col}' with flag '{flag_col}'")

    return df, rejected_columns, None
