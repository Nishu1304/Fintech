# Additional anomaly detection logic
from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies_with_isolation_forest(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_samples='auto',
    contamination='auto',
    max_features=1,
    bootstrap=False,
    n_jobs=None,
    random_state=42,
    verbose=0,
    warm_start=False
) -> pd.DataFrame:
    """
    Applies Isolation Forest to detect anomalies in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed numerical dataframe ready for anomaly detection.

    Returns:
    --------
    df_with_anomalies : pd.DataFrame
        Original dataframe with an added 'anomaly' column (-1 for anomaly, 1 for normal).
    """
    # Ensure only numerical data is used
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    # Train Isolation Forest model
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start
    )

    # Fit and predict
    model.fit(numerical_df)
    anomaly_labels = model.predict(numerical_df)  # -1 = anomaly, 1 = normal

    # Add anomaly column to original dataframe
    df_with_anomalies = df.copy()
    df_with_anomalies["anomaly"] = anomaly_labels

    return df_with_anomalies, model
