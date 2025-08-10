# Looks for behavioral patterns
# Will use HDBSCAN for the same purpose 
"""
class sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, 
max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto',
leaf_size=40, n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False, store_centers=None, copy=False)

"""

from sklearn.cluster import HDBSCAN
import pandas as pd

def detect_patterns(
    df: pd.DataFrame,
    use_sample: bool = False,
    sample_size: int = 10000,
    min_cluster_size: int = 5,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int = None,
    metric: str = 'euclidean',
    metric_params: dict = None,
    alpha: float = 1.0,
    algorithm: str = 'auto',
    leaf_size: int = 40,
    n_jobs: int = -1,
    cluster_selection_method: str = 'eom',
    allow_single_cluster: bool = False,
    store_centers: bool = None,
    copy: bool = False
) -> pd.DataFrame:
    """
    Applies scikit-learn's HDBSCAN to detect pattern-based clusters in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed numerical dataframe ready for clustering.

    Returns:
    --------
    df_with_clusters : pd.DataFrame
        Original dataframe with an added 'cluster' column (-1 indicates noise/unclustered).
    clusterer : HDBSCAN
        Fitted sklearn HDBSCAN model.
    """
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    if use_sample and len(df) > sample_size:
        numerical_df = df.select_dtypes(include=['int64', 'float64']).sample(sample_size, random_state=42)
    else:
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        store_centers=store_centers,
        copy=copy
    )

    cluster_labels = clusterer.fit_predict(numerical_df)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    return df_with_clusters, clusterer