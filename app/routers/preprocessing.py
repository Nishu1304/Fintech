# app/routers/preprocessing.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
from uuid import uuid4
from preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
from app.logger import logger
from app.config import SESSION_STORE



router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")
    if not file.filename.endswith('.csv'):
        logger.warning(f"Rejected file (not CSV): {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        logger.info(f"CSV file {file.filename} read successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    processed_df, rejection_reason = run_preprocessing_pipeline(df)

    if processed_df is None:
        logger.info(f"File {file.filename} rejected: {rejection_reason}")
        return {"status": "rejected", "reason": rejection_reason}

    # Generate session ID and store processed DataFrame
    session_id = str(uuid4())
    SESSION_STORE[session_id] = {
    "raw": df.copy(),            # before preprocessing
    "processed": processed_df    # after preprocessing
    }
    logger.info(f"Preprocessing complete. Stored in session: {session_id}")

    return {
        "status": "success",
        "message": "Preprocessing complete",
        "columns": processed_df.columns.tolist(),
        "shape": processed_df.shape,
        "session_id": session_id,
        "head": df.head(5).to_dict(orient="records")
    }

