
from models.anomaly_detector import detect_anomalies_with_isolation_forest
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.logger import logger
from app.config import SESSION_STORE


router = APIRouter()


@router.get("/detect-anomalies")
async def detect_anomalies(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid session ID"})

    raw_df = SESSION_STORE[session_id]["raw"]
    processed_df = SESSION_STORE[session_id]["processed"]

    result = detect_anomalies_with_isolation_forest(processed_df)

    # Get original values for anomaly indices
    original_anomalies = []
    for anomaly in result["anomalies"][:5]:  # limit to 5
        idx = anomaly["index"]
        original_row = raw_df.iloc[idx].to_dict()
        original_anomalies.append({
            "index": idx,
            "reason": anomaly["reason"],
            "features": original_row
        })

    return {
        "status": "success",
        "summary": result["summary"],
        "anomalies": original_anomalies
    }


@router.get("/isolation-summary")
async def isolation_summary(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid session ID"})

    raw_df = SESSION_STORE[session_id]["raw"]
    processed_df = SESSION_STORE[session_id]["processed"]

    # Run Isolation Forest
    result = detect_anomalies_with_isolation_forest(processed_df)
    anomaly_indices = [a["index"] for a in result["anomalies"]]

    # Extract anomaly rows from raw data
    anomalies_df = raw_df.iloc[anomaly_indices].copy()

    # Pick top 5 anomalies based on "Amount" column (if exists)
    if "Amount" in anomalies_df.columns:
        top_anomalies = anomalies_df.sort_values(by="Amount", ascending=False).head(5)
    else:
        top_anomalies = anomalies_df.head(5)

    top_anomalies_formatted = [
        {
            "index": idx,
            "features": row.dropna().to_dict()
        }
        for idx, row in top_anomalies.iterrows()
    ]

    return {
        "status": "success",
        "summary": result["summary"],
        "top_anomalies_by_amount": top_anomalies_formatted
    }



@router.get("/isolation-details")
async def get_isolation_details(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid session ID"})

    processed_df = SESSION_STORE[session_id]["processed"]
    result = detect_anomalies_with_isolation_forest(processed_df)

    return {
        "status": "success",
        "histogram": result["histogram"],
        "label_distribution": result["label_distribution"],
        "pca_scatter": result["pca_scatter"]
    }
