import pandas as pd
from sklearn.preprocessing import StandardScaler


# Encoding the categorical values with one hot encoding 
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical columns in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe with possible categorical features.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with categorical columns one-hot encoded.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        return df

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Avoid dummy variable trap
    return df



# Standardarizing the numerical columns

def standardize_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes numerical columns using StandardScaler.

    Parameters:
    -----------
    df : pd.DataFrame

    Returns:
    --------
    df_scaled : pd.DataFrame
        DataFrame with numerical features standardized.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
