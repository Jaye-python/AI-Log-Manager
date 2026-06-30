import io
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from src.train import train_pipeline
from src.inference import LogInferenceEngine
from src.config import DATA_DIR

app = FastAPI(
    title="Log Manager AI Pipeline",
    description="Automated system log classification, ingestion optimization, and structural breakdown summary engine.",
    version="1.0.0"
)

engine = LogInferenceEngine()

class SingleLogInput(BaseModel):
    log_id: str
    timestamp: str
    service: str
    log_message: str 

class AnalysisSummary(BaseModel):
    severity: str
    summary: str

class BatchPredictionResponse(BaseModel):
    log_id: str
    root_cause_label: str
    analysis: AnalysisSummary

@app.on_event("startup")
def startup_event():
    engine.load_artifacts()

@app.post("/preprocess-and-train", tags=["Pipeline Administration"])
def pipeline_train():
    """Runs data preparation and model training on local dataset files."""
    import os
    dataset_csv = os.path.join(DATA_DIR, "log_dataset.csv")
    if not os.path.exists(dataset_csv):
        raise HTTPException(status_code=404, detail="Training source csv missing from data/ directory.")
    
    try:
        metrics = train_pipeline(dataset_csv)
        engine.load_artifacts() 
        return {"status": "Success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single", response_model=BatchPredictionResponse, tags=["Inference Engine"])
def predict_single(payload: SingleLogInput):
    """Classifies a singular manually posted runtime log record."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model is uninitialized. Run /preprocess-and-train.")
    
    df = pd.DataFrame([payload.model_dump()])
    res = engine.predict_and_summarize(df)
    return res[0]

@app.post("/predict/file-ingest", response_model=List[BatchPredictionResponse], tags=["Inference Engine"])
async def predict_file_batch(file: UploadFile = File(...)):
    """Handles high-volume standard CSV system logs parsing."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model is uninitialized. Run /preprocess-and-train.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid extension format. API accepts raw standard csv configurations.")
        
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    required_cols = ['service', 'log_message'] 
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"Missing structural features. Required: {required_cols}")
        
    return engine.predict_and_summarize(df)

@app.post("/predict/pickle-ingest", response_model=List[BatchPredictionResponse], tags=["High Volume Processing Optimization"])
async def predict_pickle_batch(file: UploadFile = File(...)):
    """Accepts safe system byte serialized (Pickled) Pandas DataFrames for performance optimization."""
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="Model runtime uninitialized.")
        
    try:
        contents = await file.read()
        df = pickle.loads(contents)
        if not isinstance(df, pd.DataFrame):
            raise HTTPException(status_code=400, detail="Pickled byte payload must resolve to a valid Pandas DataFrame object.")
            
        return engine.predict_and_summarize(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to securely de-serialize incoming log payload: {str(e)}")