from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.config import SESSION_STORE
from app.logger import logger
from sklearn.decomposition import PCA
from models.pattern_detector import detect_patterns  # your detect_patterns function

router = APIRouter()


@router.get("/hdbscan-overview")
async def hdbscan_overview(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid session ID"})

    raw_df = SESSION_STORE[session_id]["raw"]
    processed_df = SESSION_STORE[session_id]["processed"]

    try:
        if "hdbscan_clusters" not in SESSION_STORE[session_id]:
            df_with_clusters, clusterer = detect_patterns(processed_df, use_sample=True)  # or False for real
            SESSION_STORE[session_id]["hdbscan_clusters"] = df_with_clusters
            SESSION_STORE[session_id]["hdbscan_model"] = clusterer
        else:
            df_with_clusters = SESSION_STORE[session_id]["hdbscan_clusters"]


        cluster_labels = df_with_clusters['cluster']
        total_points = len(cluster_labels)
        cluster_counts = cluster_labels.value_counts().to_dict()

        num_clusters = len([label for label in cluster_counts if label != -1])
        noise_count = cluster_counts.get(-1, 0)

        # Sample noise points (max 5)
        noise_indices = df_with_clusters[df_with_clusters['cluster'] == -1].index.tolist()[:5]
        sample_noise_points = []
        for idx in noise_indices:
            original_row = raw_df.iloc[idx].dropna().to_dict()
            sample_noise_points.append({
                "index": idx,
                "features": original_row
            })

        # Summary paragraph
        summary = (
            f"HDBSCAN clustering identified {num_clusters} distinct behavior pattern(s). "
            f"Out of {total_points} records, {total_points - noise_count} were clustered, "
            f"while {noise_count} were considered anomalies (noise points)."
        )

        return {
            "status": "success",
            "summary": summary,
            "cluster_stats": {
                "total_points": total_points,
                "num_clusters": num_clusters,
                "noise_count": noise_count,
                "cluster_distribution": cluster_counts
            },
            "sample_noise_points": sample_noise_points
        }

    except Exception as e:
        logger.error(f"HDBSCAN overview error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "HDBSCAN processing failed."})



@router.get("/hdbscan-details")
async def hdbscan_details(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid session ID"})

    raw_df = SESSION_STORE[session_id]["raw"]
    processed_df = SESSION_STORE[session_id]["processed"]

    try:
        # ✅ Use cached results if available
        if "hdbscan_clusters" in SESSION_STORE[session_id] and "hdbscan_model" in SESSION_STORE[session_id]:
            df_with_clusters = SESSION_STORE[session_id]["hdbscan_clusters"]
            clusterer = SESSION_STORE[session_id]["hdbscan_model"]
        else:
            df_with_clusters, clusterer = detect_patterns(processed_df, use_sample=True)  # or use_sample=False
            SESSION_STORE[session_id]["hdbscan_clusters"] = df_with_clusters
            SESSION_STORE[session_id]["hdbscan_model"] = clusterer

        cluster_labels = df_with_clusters['cluster']
        numerical_df = processed_df.select_dtypes(include=['int64', 'float64'])
        cluster_distribution = cluster_labels.value_counts().to_dict()

        # ✅ PCA for cluster visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numerical_df)

        pca_scatter = []
        for i, (x, y) in enumerate(pca_result):
            pca_scatter.append({
                "x": float(x),
                "y": float(y),
                "cluster": int(cluster_labels.iloc[i])
            })

        # ✅ Cluster summaries with top features
        global_means = numerical_df.mean()
        cluster_summaries = {}

        for cluster_id in sorted(cluster_distribution.keys()):
            cluster_rows = df_with_clusters[df_with_clusters['cluster'] == cluster_id].index
            cluster_data = numerical_df.loc[cluster_rows]

            cluster_means = cluster_data.mean()
            diff = (cluster_means - global_means).abs().sort_values(ascending=False)
            top_features = diff.head(3).index.tolist()

            top_feature_info = []
            for feature in top_features:
                top_feature_info.append({
                    "feature": feature,
                    "mean": round(cluster_means[feature], 3),
                    "global_mean": round(global_means[feature], 3),
                    "diff": round(cluster_means[feature] - global_means[feature], 3)
                })

            cluster_summaries[str(cluster_id)] = {
                "size": int(cluster_distribution[cluster_id]),
                "top_features": top_feature_info
            }

        return {
            "status": "success",
            "pca_scatter": pca_scatter,
            "cluster_distribution": {str(k): int(v) for k, v in cluster_distribution.items()},
            "cluster_summaries": cluster_summaries
        }

    except Exception as e:
        logger.error(f"HDBSCAN details error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "HDBSCAN details generation failed."})