"""Problem 2 polarity and intensity metrics, with undefined Pearson handled explicitly."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def sentiment_metrics(labels: np.ndarray, predictions: np.ndarray, truth: np.ndarray, intensity: np.ndarray) -> dict:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    intensity = np.asarray(intensity).reshape(-1)
    if not (len(labels) == len(predictions) == len(truth) == len(intensity)) or not len(labels):
        raise ValueError("metric arrays must be nonempty and have equal length")
    pearson = None
    if len(labels) > 1 and np.std(truth) > 1e-12 and np.std(intensity) > 1e-12:
        pearson = float(np.corrcoef(truth, intensity)[0, 1])
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "neutral_recall": float(recall_score(labels, predictions, labels=[1], average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(truth - intensity))),
        "pearson": pearson,
    }
