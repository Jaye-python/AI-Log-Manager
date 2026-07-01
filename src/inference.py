import os
import joblib
import numpy as np
import pandas as pd
from src.preprocess import transform_data
from src.summarize import generate_summary
from src.config import MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH, LABELS_PATH


class LogInferenceEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.encoder = None
        self.labels_metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads the saved model, word-scoring tool, and label mapper from disk.
        Also reads the root cause reference sheet so the engine can look up
        severity levels and recommended actions after making a prediction."""
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

    def _confidence_scores(self, X) -> list[float]:
        """Estimates how confident the model is in each prediction as a score between
        0 and 1. A score close to 1 means the model is very sure; close to 0 means
        it is uncertain and the result should be treated with caution."""
        scores = self.model.decision_function(X)
        if scores.ndim == 1:
            return (1 / (1 + np.exp(-np.abs(scores)))).tolist()
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs.max(axis=1).tolist()

    def predict_and_summarize(self, df: pd.DataFrame) -> list:
        """Takes a table of log records, figures out the root cause for each one,
        and returns a plain-English breakdown of what went wrong, how severe it is,
        and what the on-call engineer should do about it."""
        if not self.is_ready:
            raise ValueError("Model artifacts or root cause label metadata missing.")

        assert self.vectorizer is not None
        X = transform_data(df, self.vectorizer)
        predictions = self.model.predict(X)
        predicted_labels = self.encoder.inverse_transform(predictions)
        confidences = self._confidence_scores(X)

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            rc_code = predicted_labels[i]
            meta = self.labels_metadata.get(rc_code, {})
            results.append({
                "log_id": str(row.get('log_id', f"UNKNOWN_{i}")),
                "root_cause_label": rc_code,
                "analysis": generate_summary(
                    log_message=str(row.get('log_message', '')),
                    service=str(row.get('service', '')),
                    severity=meta.get('severity', 'UNKNOWN'),
                    predicted_label_id=rc_code,
                    predicted_label_name=meta.get('label', 'System Variance'),
                    confidence=confidences[i],
                    typical_resolution=meta.get('typical_resolution'),
                )
            })
        return results
