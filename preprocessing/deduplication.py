# remove duplicate rows
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate rows from the DataFrame.

    Returns:
    --------
    DataFrame without duplicates (keeping the first occurrence).
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(keep='first').copy()
    removed_count = initial_count - len(df_clean)
    logger.info(f"Removed {removed_count} duplicate rows.")
    return df_clean

