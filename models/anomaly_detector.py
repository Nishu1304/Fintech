from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

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
    warm_start=False,
    top_k_features: int = 3  # For reporting
) -> dict:
    """
    Applies Isolation Forest to detect anomalies and returns structured summary, chart data and anomalies.
    """
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

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

    model.fit(numerical_df)
    anomaly_labels = model.predict(numerical_df)
    anomaly_scores = model.decision_function(numerical_df)  # Higher = more normal

    df_with_anomalies = df.copy()
    df_with_anomalies["anomaly"] = anomaly_labels

    # Summary
    label_counts = {
        "normal": int(np.sum(anomaly_labels == 1)),
        "anomalies": int(np.sum(anomaly_labels == -1))
    }
    summary = {
        "total_records": len(df),
        "num_anomalies": label_counts["anomalies"],
        "anomaly_percentage": round((label_counts["anomalies"] / len(df)) * 100, 2)
    }

    # Histogram (scores)
    hist_counts, hist_bins = np.histogram(anomaly_scores, bins=20)

    # PCA for scatter
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numerical_df)
        pca_data = [
            {"x": float(x), "y": float(y), "label": "anomaly" if lbl == -1 else "normal"}
            for (x, y), lbl in zip(pca_result, anomaly_labels)
        ][:1000]  # Limit to 1000 for performance
    except Exception:
        pca_data = []

    # Top-K anomaly feature extraction
    top_features = numerical_df.columns[:top_k_features]
    anomalies_list = []
    for idx in np.where(anomaly_labels == -1)[0]:
        row = df.iloc[idx]
        feature_dict = {col: row[col] for col in top_features}
        anomalies_list.append({
            "index": int(idx),
            "reason": "IsolationForest flagged this as anomaly",
            "features": feature_dict
        })

    return {
        "summary": summary,
        "anomalies": anomalies_list,
        "histogram": {
            "bins": hist_bins.tolist(),
            "counts": hist_counts.tolist()
        },
        "label_distribution": label_counts,
        "pca_scatter": pca_data
    }
