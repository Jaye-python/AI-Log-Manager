import json
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.preprocess import prepare_data
from src.config import (
    MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH, 
    METRICS_PATH, REPORT_PATH
)

def train_pipeline(data_path: str):
    """Loads dataset, trains LinearSVC classifier, and writes all validation files to disk."""
    df = pd.read_csv(data_path)
    
    X, y, vectorizer, encoder = prepare_data(df, is_training=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LinearSVC(C=1.0, random_state=42, dual=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=encoder.classes_, 
        output_dict=True, 
        zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": report_dict
    }
    report_text = str(classification_report(
        y_test, y_pred, 
        target_names=encoder.classes_, 
        zero_division=0
    ))
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    
    return metrics