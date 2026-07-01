import os

ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
PREDICTIONS_DIR = os.path.join(ARTIFACTS_DIR, "predictions")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")

METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
EVALUATIONS_CSV_PATH = os.path.join(ARTIFACTS_DIR, "evaluations.csv")

LABELS_PATH = os.path.join(DATA_DIR, "root_cause_labels.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)