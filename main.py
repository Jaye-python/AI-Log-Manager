import io
import logging
import os
import pickle
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

logging.getLogger("uvicorn.access").disabled = True
# logging.getLogger("uvicorn.error").disabled = True

from src.train import train_pipeline
from src.inference import LogInferenceEngine
from src.config import DATA_DIR, METRICS_PATH, ARTIFACTS_DIR, PREDICTIONS_DIR, EVALUATIONS_CSV_PATH

engine = LogInferenceEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_artifacts()
    yield

app = FastAPI(
    title="Log Manager AI Pipeline",
    description="Automated system log classification, ingestion optimization, and structural breakdown summary engine.",
    version="1.0.0",
    lifespan=lifespan
)

class SingleLogInput(BaseModel):
    log_id: str
    timestamp: str
    service: str
    log_message: str

class AnalysisSummary(BaseModel):
    issue: str
    predicted_root_cause_id: str
    predicted_root_cause_label: str
    affected_service: str
    severity: str
    recommended_action: str
    confidence: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    log_id: str
    root_cause_label: str
    analysis: AnalysisSummary

def _safe_file_response(path: str, allowed_dir: str, filename: str) -> FileResponse:
    """Resolves the file path and confirms it sits inside the allowed directory
    before serving it — prevents path traversal if a constructed path ever
    contains unexpected sequences."""
    resolved = os.path.realpath(path)
    allowed = os.path.realpath(allowed_dir)
    if not resolved.startswith(allowed + os.sep) and resolved != allowed:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    return FileResponse(resolved, media_type="text/csv", filename=filename)

def _save_predictions_csv(results: list, label: str) -> str:
    """Flattens prediction results into a CSV and saves it to the predictions folder."""
    rows = []
    for r in results:
        row = {"log_id": r["log_id"], "root_cause_label": r["root_cause_label"]}
        row.update(r["analysis"])
        rows.append(row)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(PREDICTIONS_DIR, f"{label}_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@app.get("/evaluations", tags=["Pipeline Administration"])
def get_evaluations():
    """Returns the latest training evaluation metrics as a downloadable CSV."""
    if not os.path.exists(EVALUATIONS_CSV_PATH):
        raise HTTPException(status_code=404, detail="No evaluation metrics found. Run /preprocess-and-train.")
    return _safe_file_response(EVALUATIONS_CSV_PATH, ARTIFACTS_DIR, "evaluations.csv")

@app.post("/preprocess-and-train", tags=["Pipeline Administration"])
def pipeline_train():
    """Runs data preparation and model training on local dataset files."""
    dataset_csv = os.path.join(DATA_DIR, "log_dataset.csv")
    if not os.path.exists(dataset_csv):
        raise HTTPException(status_code=404, detail="Training source csv missing from data/ directory.")
    
    try:
        metrics = train_pipeline(dataset_csv)
        engine.load_artifacts() 
        return {"status": "Success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single", tags=["Inference Engine"])
def predict_single(payload: SingleLogInput):
    """Classifies a singular manually posted runtime log record."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model is uninitialized. Run /preprocess-and-train.")

    df = pd.DataFrame([payload.model_dump()])
    res = engine.predict_and_summarize(df)
    path = _save_predictions_csv(res, "prediction_single")
    return _safe_file_response(path, PREDICTIONS_DIR, os.path.basename(path))

@app.post("/predict/file-ingest", tags=["Inference Engine"])
async def predict_file_batch(file: UploadFile = File(...)):
    """Handles high-volume standard CSV system logs parsing."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model is uninitialized. Run /preprocess-and-train.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid extension format. API accepts raw standard csv configurations.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')

    required_cols = ['log_id', 'service', 'log_message']
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"Missing structural features. Required: {required_cols}")

    res = engine.predict_and_summarize(df)
    path = _save_predictions_csv(res, "prediction_batch")
    return _safe_file_response(path, PREDICTIONS_DIR, os.path.basename(path))

@app.post("/predict/pickle-ingest", tags=["High Volume Processing Optimization"])
async def predict_pickle_batch(file: UploadFile = File(...)):
    """Accepts safe system byte serialized (Pickled) Pandas DataFrames for performance optimization."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model runtime uninitialized.")

    try:
        contents = await file.read()
        df = pickle.loads(contents)
        if not isinstance(df, pd.DataFrame):
            raise HTTPException(status_code=400, detail="Pickled byte payload must resolve to a valid Pandas DataFrame object.")

        required_cols = ['log_id', 'service', 'log_message']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"Missing structural features. Required: {required_cols}")

        res = engine.predict_and_summarize(df)
        path = _save_predictions_csv(res, "pickle_ingest")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to securely de-serialize incoming log payload: {str(e)}")

    return _safe_file_response(path, PREDICTIONS_DIR, os.path.basename(path))