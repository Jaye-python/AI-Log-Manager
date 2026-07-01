import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from src.preprocess import fit_transform_data, transform_data
from src.evaluate import evaluate_predictions
from src.config import (
    MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH,
    METRICS_PATH, REPORT_PATH, LABELS_PATH, EVALUATIONS_CSV_PATH
)

def train_pipeline(data_path: str):
    """Runs the full training process from start to finish. Reads the log dataset,
    splits it into training and test sets, trains the classifier, measures how well
    it performed, and saves everything to disk so the API can use it."""
    df = pd.read_csv(data_path)
    labels_df = pd.read_csv(LABELS_PATH)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['root_cause_label'])

    X_train, y_train, vectorizer, encoder = fit_transform_data(df_train, labels_df)

    X_test = transform_data(df_test, vectorizer)
    y_test = encoder.transform(df_test['root_cause_label'])

    model = LinearSVC(C=1.0, random_state=42, dual=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_true_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    metrics = evaluate_predictions(
        y_true_labels, y_pred_labels,
        labels=list(encoder.classes_),
        metrics_path=METRICS_PATH,
        report_path=REPORT_PATH,
        csv_path=EVALUATIONS_CSV_PATH,
    )

    return metrics