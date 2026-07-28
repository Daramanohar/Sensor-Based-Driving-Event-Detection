"""Run the complete, reproducible driving-event detection workflow.

The pipeline produces compact result files for the labeled sample and the unlabeled
30-minute drive. Evaluation claims are intentionally limited to the supplied labeled
sample; full-session rows are detector outputs, not reference labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from driving_events.config import load_config
from driving_events.data import (
    label_samples,
    load_annotations,
    load_sensor_csv,
    profile_sensor_frame,
)
from driving_events.evaluation import event_level_metrics
from driving_events.features import extract_feature_table
from driving_events.labeling import annotation_coverage
from driving_events.models import exploratory_group_cross_validation, fit_final_xgboost
from driving_events.preprocessing import preprocess_sensor_frame
from driving_events.robustness import simulate_regular_gps_gaps
from driving_events.streaming import replay_rule_detector

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
LABELS_PATH = ROOT / "data" / "sample_labels.csv"
RESULTS_DIR = ROOT / "results"


def _json_ready(value: Any) -> Any:
    """Convert nested NumPy/Pandas scalar values into regular Python values."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _comparison_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in report["aggregate"].items():
        classification = metrics["classification_report"]
        rows.append(
            {
                "model": model_name,
                "accuracy": classification["accuracy"],
                "macro_precision": classification["macro avg"]["precision"],
                "macro_recall": classification["macro avg"]["recall"],
                "macro_f1": classification["macro avg"]["f1-score"],
                "weighted_f1": classification["weighted avg"]["f1-score"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def main() -> int:
    """Execute every reproducible stage and write the final artifacts."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG_PATH)
    sample_rate = float(config["data"]["sample_rate_hz"])
    full_rate = float(config["data"]["full_rate_hz"])

    sample = load_sensor_csv(
        ROOT / config["data"]["sample_csv"],
        sample_rate,
        session_id="sample_10hz",
        physical_limits=config["data"]["physical_limits"],
    )
    full = load_sensor_csv(
        ROOT / config["data"]["full_csv"],
        full_rate,
        session_id="full_50hz",
        physical_limits=config["data"]["physical_limits"],
    )
    labels = load_annotations(LABELS_PATH, reviewed_only=True)
    labeled_sample = label_samples(sample, labels, require_complete_coverage=True)

    sample_events = replay_rule_detector(sample, sample_rate, config)
    sample_events.to_csv(RESULTS_DIR / "sample_events.csv", index=False)
    rule_metrics = event_level_metrics(
        labels,
        sample_events,
        session_duration_s=len(sample) / sample_rate,
        tolerance_s=0.60,
    )

    gps_robustness: dict[str, dict[str, float | int]] = {}
    for gap_s in (1.0, 2.0, 4.0, 5.0):
        perturbed = simulate_regular_gps_gaps(sample, sample_rate, gap_s)
        gap_events = replay_rule_detector(perturbed, sample_rate, config)
        gap_metrics = event_level_metrics(
            labels,
            gap_events,
            session_duration_s=len(sample) / sample_rate,
            tolerance_s=0.60,
        )
        gps_robustness[f"{gap_s:g}s"] = {
            "detected_events": int(gap_metrics["predicted_events"]),
            "macro_f1": float(gap_metrics["macro_f1"]),
        }

    processed = preprocess_sensor_frame(labeled_sample, sample_rate, config)
    features = extract_feature_table(processed, sample_rate, config)
    comparison_report = exploratory_group_cross_validation(features, config, n_splits=2)
    comparison = _comparison_table(comparison_report)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    (RESULTS_DIR / "model_evaluation.json").write_text(
        json.dumps(_json_ready(comparison_report), indent=2),
        encoding="utf-8",
    )

    model = fit_final_xgboost(features, config)
    joblib.dump(model, RESULTS_DIR / "xgboost_model.joblib")
    importance = (
        pd.DataFrame(
            {
                "feature": model.feature_columns,
                "importance": model.estimator.feature_importances_,
            }
        )
        .sort_values(
            ["importance", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    full_events = replay_rule_detector(full, full_rate, config)
    full_events.to_csv(RESULTS_DIR / "full_session_events.csv", index=False)
    event_counts = {
        str(label): int(count)
        for label, count in full_events["label"].value_counts().sort_index().items()
    }

    best_model = comparison.iloc[0]
    coverage = annotation_coverage(sample, labeled_sample)
    summary = {
        "project": {
            "author": "Dara Manohar",
            "task": "Sensor-Based Driving Event Detection",
            "random_seed": int(config["project"]["random_seed"]),
        },
        "data": {
            "sample": profile_sensor_frame(sample, sample_rate),
            "full_session": profile_sensor_frame(full, full_rate),
            "sample_label_coverage": float(coverage["coverage"]),
        },
        "rule_detector_on_labeled_sample": rule_metrics,
        "gps_gap_robustness": gps_robustness,
        "machine_learning": {
            "validation": "two-fold event-group cross-validation on one labeled session",
            "best_model": str(best_model["model"]),
            "accuracy": float(best_model["accuracy"]),
            "macro_f1": float(best_model["macro_f1"]),
            "feature_count": len(model.feature_columns),
            "limitations": comparison_report["limitations"],
        },
        "full_session_detection": {
            "detected_events": int(len(full_events)),
            "counts_by_class": event_counts,
            "interpretation": (
                "Detector outputs on an unlabeled session; no accuracy claim is made "
                "for these rows."
            ),
        },
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )

    print("Pipeline completed")
    print(f"  Labeled sample rule macro-F1: {rule_metrics['macro_f1']:.4f}")
    print(
        f"  Best grouped-CV model: {best_model['model']} "
        f"(macro-F1={best_model['macro_f1']:.4f})"
    )
    print(f"  Full-session detected episodes: {len(full_events)}")
    print(f"  Results: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
