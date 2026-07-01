import json
import numpy as np
import pandas as pd
from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: Iterable,
    y_pred: Iterable,
    labels=None,
    metrics_path: str | None = None,
    report_path: str | None = None,
    csv_path: str | None = None,
) -> dict:
    """Compares what the model predicted against the correct answers and produces
    a scorecard. Saves the results to metrics.json and classification_report.txt
    if paths are provided. Returns overall accuracy, per-category scores, and a
    grid showing which categories the model confuses with each other."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    precision_per, recall_per, f1_per, support_per = (
        np.asarray(a) for a in precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    )

    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision_per, recall_per, f1_per, support_per)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=False,
        zero_division=0,
    )
    if not isinstance(report_text, str):
        report_text = json.dumps(report_text, indent=2)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
        "n_samples": len(y_true),
    }

    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if report_path:
        with open(report_path, "w") as f:
            f.write(report_text)

    if csv_path:
        rows = [{"metric": "accuracy", "value": metrics["accuracy"], "class": "overall"}]
        rows += [{"metric": "precision", "value": metrics["precision_macro"], "class": "macro"}]
        rows += [{"metric": "recall", "value": metrics["recall_macro"], "class": "macro"}]
        rows += [{"metric": "f1", "value": metrics["f1_macro"], "class": "macro"}]
        rows += [{"metric": "precision", "value": metrics["precision_weighted"], "class": "weighted"}]
        rows += [{"metric": "recall", "value": metrics["recall_weighted"], "class": "weighted"}]
        rows += [{"metric": "f1", "value": metrics["f1_weighted"], "class": "weighted"}]
        for label, scores in per_class.items():
            for metric, val in scores.items():
                rows.append({"metric": metric, "value": val, "class": label})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    return metrics
