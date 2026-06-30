import os
import joblib
import pandas as pd
from src.preprocess import prepare_data
from src.config import MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH, LABELS_PATH

class LogInferenceEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.encoder = None
        self.labels_metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
        
        if os.path.exists(LABELS_PATH):
            labels_df = pd.read_csv(LABELS_PATH)
            self.labels_metadata = labels_df.set_index('id').to_dict(orient='index')

    @property
    def is_ready(self):
        return self.model is not None and len(self.labels_metadata) > 0

    def generate_summary(self, row: pd.Series, predicted_rc: str) -> dict:
        """Generates a structured runtime issue breakdown dictionary using the metadata source of truth."""
        msg = row['log_message']
        svc = row['service']
        
        meta = self.labels_metadata.get(predicted_rc, {})
        true_severity = meta.get('severity', 'UNKNOWN').upper()
        friendly_name = meta.get('label', 'System Variance')

        clean_msg = msg.split(']')[-1].strip() if ']' in msg else msg
        
        return {
            "severity": true_severity,
            "summary": f"Component '{svc}' encountered {friendly_name}. Detail: {clean_msg}"
        }

    def predict_and_summarize(self, df: pd.DataFrame) -> list:
        if not self.is_ready:
            raise ValueError("Model artifacts or root cause label metadata missing.")
        
        X = prepare_data(df, is_training=False, vectorizer=self.vectorizer)
        predictions = self.model.predict(X)
        predicted_labels = self.encoder.inverse_transform(predictions)
        
        results = []
        for idx, row in df.iterrows():
            rc_code = predicted_labels[idx]
            results.append({
                "log_id": str(row.get('log_id', f"UNKNOWN_{idx}")),
                "root_cause_label": rc_code,
                "analysis": self.generate_summary(row, rc_code) 
            })
        return results