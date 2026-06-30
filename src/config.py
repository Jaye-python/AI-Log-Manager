import os

ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")

METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "classification_report.txt")

LABELS_PATH = os.path.join(DATA_DIR, "root_cause_labels.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)