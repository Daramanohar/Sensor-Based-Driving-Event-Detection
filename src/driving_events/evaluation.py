"""Window- and event-level metrics with duplicate predictions counted as false positives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
)
from sklearn.preprocessing import label_binarize

from .data import EVENT_LABELS


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def window_classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    y_probability: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> dict[str, object]:
    """Return JSON-ready classification metrics without hiding absent classes."""

    true = np.asarray(list(y_true), dtype=str)
    predicted = np.asarray(list(y_pred), dtype=str)
    labels = classes or sorted(set(true).union(predicted))
    report = classification_report(
        true,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, object] = {
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }
    if y_probability is not None:
        binary = label_binarize(true, classes=labels)
        if binary.shape[1] == y_probability.shape[1]:
            result["average_precision_ovr"] = {
                label: float(average_precision_score(binary[:, index], y_probability[:, index]))
                if len(np.unique(binary[:, index])) > 1
                else None
                for index, label in enumerate(labels)
            }
    return result


def tune_one_vs_rest_thresholds(
    y_true: Iterable[str],
    probabilities: np.ndarray,
    classes: list[str],
    *,
    beta: float = 2.0,
) -> dict[str, float]:
    """Choose per-class thresholds on validation data only using F-beta."""

    true = np.asarray(list(y_true), dtype=str)
    thresholds: dict[str, float] = {}
    threshold_grid = np.linspace(0.10, 0.90, 81)
    for index, label in enumerate(classes):
        binary_true = true == label
        if binary_true.sum() == 0:
            thresholds[label] = 0.50
            continue
        scores = probabilities[:, index]
        objective = [
            fbeta_score(binary_true, scores >= threshold, beta=beta, zero_division=0)
            for threshold in threshold_grid
        ]
        thresholds[label] = float(threshold_grid[int(np.argmax(objective))])
    return thresholds


def apply_multiclass_thresholds(
    probabilities: np.ndarray,
    classes: list[str],
    thresholds: dict[str, float],
    *,
    fallback_label: str = "Normal Driving",
) -> np.ndarray:
    """Apply class-specific thresholds, choosing the largest threshold-normalized score."""

    output: list[str] = []
    for row in probabilities:
        normalized = np.asarray(
            [
                row[index] / max(thresholds.get(label, 0.5), 1e-6)
                for index, label in enumerate(classes)
            ]
        )
        best = int(np.argmax(normalized))
        output.append(classes[best] if normalized[best] >= 1.0 else fallback_label)
    return np.asarray(output, dtype=str)


@dataclass(frozen=True)
class _Match:
    truth_index: int
    prediction_index: int
    onset_latency_s: float


def _interval_distance(
    truth_start: float,
    truth_end: float,
    pred_start: float,
    pred_end: float,
) -> float:
    if pred_end < truth_start:
        return truth_start - pred_end
    if truth_end < pred_start:
        return pred_start - truth_end
    return 0.0


def _match_events(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance_s: float,
) -> list[_Match]:
    match_options: list[tuple[float, int, int]] = []
    for truth_index, truth_row in truth.iterrows():
        same = predictions.loc[
            predictions["session_id"].eq(truth_row["session_id"])
            & predictions["label"].eq(truth_row["label"])
        ]
        for prediction_index, pred_row in same.iterrows():
            distance = _interval_distance(
                float(truth_row["start_time_s"]),
                float(truth_row["end_time_s"]),
                float(pred_row["start_time_s"]),
                float(pred_row["end_time_s"]),
            )
            if distance <= tolerance_s:
                match_options.append((distance, truth_index, prediction_index))

    matches: list[_Match] = []
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    for _, truth_index, prediction_index in sorted(match_options):
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        latency = float(
            predictions.loc[prediction_index, "start_time_s"]
            - truth.loc[truth_index, "start_time_s"]
        )
        matches.append(_Match(truth_index, prediction_index, latency))
    return matches


def event_level_metrics(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    session_duration_s: float,
    tolerance_s: float = 0.50,
    labels: tuple[str, ...] = EVENT_LABELS,
) -> dict[str, object]:
    """Evaluate unique physical events, not individual threshold crossings."""

    reviewed = truth.loc[
        truth["review_status"].eq("reviewed") & truth["label"].isin(labels)
    ].reset_index(drop=True)
    predicted = predictions.loc[predictions["label"].isin(labels)].reset_index(drop=True)
    matches = _match_events(reviewed, predicted, tolerance_s)
    matched_truth = {match.truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}

    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        truth_indices = set(reviewed.index[reviewed["label"].eq(label)])
        prediction_indices = set(predicted.index[predicted["label"].eq(label)])
        tp = len(truth_indices.intersection(matched_truth))
        fn = len(truth_indices) - tp
        fp = len(prediction_indices) - len(prediction_indices.intersection(matched_predictions))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "true_events": len(truth_indices),
            "predicted_events": len(prediction_indices),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    latencies = [match.onset_latency_s for match in matches]
    total_fp = len(predicted) - len(matched_predictions)
    return {
        "matching_tolerance_s": float(tolerance_s),
        "reviewed_true_events": int(len(reviewed)),
        "predicted_events": int(len(predicted)),
        "matched_events": int(len(matches)),
        "per_class": per_class,
        "macro_f1": float(np.mean([metrics["f1"] for metrics in per_class.values()])),
        "false_positives_per_hour": safe_divide(total_fp, session_duration_s / 3600.0),
        "onset_latency_s": {
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
    }
