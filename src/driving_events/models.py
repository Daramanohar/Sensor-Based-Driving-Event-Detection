"""Leakage-aware baseline comparison with XGBoost as the primary learned model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .evaluation import window_classification_metrics
from .features import model_feature_columns


class InsufficientGroundTruthError(RuntimeError):
    """Raised when a defensible train/validation/test experiment cannot be formed."""


@dataclass
class ModelBundle:
    name: str
    estimator: Any
    label_encoder: LabelEncoder
    feature_columns: list[str]
    thresholds: dict[str, float]
    metadata: dict[str, Any]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame[self.feature_columns])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self.estimator.predict(frame[self.feature_columns])
        return self.label_encoder.inverse_transform(np.asarray(encoded, dtype=int))


def chronological_split(
    features: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    *,
    embargo_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create contiguous splits with an embargo to reduce adjacent-window leakage."""

    if features["session_id"].nunique() != 1:
        raise ValueError("chronological_split expects exactly one session")
    ordered = features.sort_values("elapsed_s").reset_index(drop=True)
    end_time = float(ordered["elapsed_s"].max())
    train_end = end_time * (1.0 - validation_fraction - test_fraction)
    validation_end = end_time * (1.0 - test_fraction)
    train = ordered.loc[ordered["elapsed_s"] <= train_end - embargo_s].copy()
    validation = ordered.loc[
        ordered["elapsed_s"].between(train_end + embargo_s, validation_end - embargo_s)
    ].copy()
    test = ordered.loc[ordered["elapsed_s"] >= validation_end + embargo_s].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise InsufficientGroundTruthError("Chronological split is empty after applying embargo")
    return train, validation, test


def validate_split_class_coverage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    minimum_events_per_class: int = 1,
) -> list[str]:
    """Return limitations that prevent a validated multiclass claim."""

    limitations: list[str] = []
    train_labels = set(train["label"].dropna())
    for split_name, split in (("validation", validation), ("test", test)):
        missing = train_labels.difference(split["label"].dropna())
        if missing:
            limitations.append(f"{split_name} lacks labels present in training: {sorted(missing)}")
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        event_counts = split.loc[
            ~split["label"].isin(["Normal Driving", "Stationary"])
        ].groupby("label")["event_id"].nunique()
        under = event_counts.loc[event_counts < minimum_events_per_class]
        if len(under):
            limitations.append(
                f"{split_name} has fewer than {minimum_events_per_class} events for "
                f"{under.index.tolist()}"
            )
    return limitations


def _estimators(config: dict[str, Any], class_count: int) -> dict[str, Any]:
    seed = int(config["project"]["random_seed"])
    xgb = config["model"]["xgboost"]
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            n_estimators=int(xgb["n_estimators"]),
            learning_rate=float(xgb["learning_rate"]),
            max_depth=int(xgb["max_depth"]),
            min_child_weight=float(xgb["min_child_weight"]),
            subsample=float(xgb["subsample"]),
            colsample_bytree=float(xgb["colsample_bytree"]),
            reg_alpha=float(xgb["reg_alpha"]),
            reg_lambda=float(xgb["reg_lambda"]),
            gamma=float(xgb["gamma"]),
            tree_method=str(xgb["tree_method"]),
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def fit_model_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, ModelBundle], dict[str, object]]:
    """Fit transparent baselines and XGBoost on the exact same feature/label data."""

    feature_columns = model_feature_columns(train)
    encoder = LabelEncoder().fit(train["label"].astype(str))
    unseen = set(validation["label"].astype(str)).difference(encoder.classes_)
    if unseen:
        raise InsufficientGroundTruthError(f"Validation contains unseen labels: {sorted(unseen)}")
    y_train = encoder.transform(train["label"].astype(str))
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    bundles: dict[str, ModelBundle] = {}
    comparison: dict[str, object] = {}
    for name, estimator in _estimators(config, len(encoder.classes_)).items():
        fit_kwargs: dict[str, Any] = {}
        if name == "xgboost":
            fit_kwargs["sample_weight"] = sample_weight
        elif name == "logistic_regression":
            fit_kwargs["model__sample_weight"] = sample_weight
        else:
            fit_kwargs["sample_weight"] = sample_weight
        estimator.fit(train[feature_columns], y_train, **fit_kwargs)
        probability = estimator.predict_proba(validation[feature_columns])
        predicted = encoder.inverse_transform(np.argmax(probability, axis=1))
        metrics = window_classification_metrics(
            validation["label"].astype(str),
            predicted,
            y_probability=probability,
            classes=encoder.classes_.tolist(),
        )
        comparison[name] = metrics
        bundles[name] = ModelBundle(
            name=name,
            estimator=estimator,
            label_encoder=encoder,
            feature_columns=feature_columns,
            thresholds={label: 0.5 for label in encoder.classes_},
            metadata={"validation_rows": len(validation), "training_rows": len(train)},
        )
    return bundles, comparison


def exploratory_group_cross_validation(
    features: pd.DataFrame,
    config: dict[str, Any],
    *,
    n_splits: int = 2,
) -> dict[str, object]:
    """Run leakage-reduced event-group CV for a small, single-session demonstration.

    This is deliberately named *exploratory*: two examples per event class cannot support a
    production generalization claim, even when each physical event remains in only one fold.
    """

    prepared = features.loc[features["label"].notna() & features["event_id"].notna()].copy()
    prepared["label"] = prepared["label"].replace({"Stationary": "Normal Driving"})
    group_labels = prepared.groupby("event_id")["label"].agg(
        lambda values: values.mode().iloc[0]
    )
    rng = np.random.default_rng(int(config["project"]["random_seed"]))
    fold_groups: list[set[str]] = [set() for _ in range(n_splits)]
    for _, label_groups in group_labels.groupby(group_labels):
        groups = label_groups.index.to_numpy(dtype=str)
        if len(groups) < n_splits:
            raise InsufficientGroundTruthError(
                f"Need at least {n_splits} independent intervals for label "
                f"{label_groups.iloc[0]!r}; found {len(groups)}"
            )
        for position, group in enumerate(rng.permutation(groups)):
            fold_groups[position % n_splits].add(str(group))
    model_predictions: dict[str, list[pd.DataFrame]] = {}
    fold_summaries: list[dict[str, object]] = []

    for fold_index, test_groups in enumerate(fold_groups, start=1):
        test_mask = prepared["event_id"].astype(str).isin(test_groups)
        train = prepared.loc[~test_mask].copy()
        test = prepared.loc[test_mask].copy()
        bundles, comparison = fit_model_comparison(train, test, config)
        train_event_ids = sorted(train["event_id"].astype(str).unique().tolist())
        test_event_ids = sorted(test["event_id"].astype(str).unique().tolist())
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_event_groups": int(train["event_id"].nunique()),
                "test_event_groups": int(test["event_id"].nunique()),
                "train_event_ids": train_event_ids,
                "test_event_ids": test_event_ids,
                "event_group_overlap": len(set(train_event_ids).intersection(test_event_ids)),
                "models": comparison,
            }
        )
        for name, bundle in bundles.items():
            predicted = bundle.predict(test)
            model_predictions.setdefault(name, []).append(
                pd.DataFrame(
                    {
                        "row_index": test.index,
                        "truth": test["label"].astype(str).to_numpy(),
                        "prediction": predicted,
                    }
                )
            )

    aggregate = {}
    for name, parts in model_predictions.items():
        out_of_fold = pd.concat(parts).sort_values("row_index")
        aggregate[name] = window_classification_metrics(
            out_of_fold["truth"],
            out_of_fold["prediction"],
        )
    return {
        "status": "exploratory_two_fold_event_group_cv",
        "limitations": [
            "Only one 25.7-second session is reviewed.",
            "Each target class has only two physical events.",
            "This estimates within-session separability, not cross-driver/device generalization.",
            "Thresholds were not tuned on an independent validation set.",
        ],
        "grouping": "All windows ending inside one annotated physical interval stay in one fold.",
        "folds": fold_summaries,
        "aggregate": aggregate,
    }


def fit_final_xgboost(features: pd.DataFrame, config: dict[str, Any]) -> ModelBundle:
    """Fit a rebuildable XGBoost model on all labeled sample rows."""

    prepared = features.loc[features["label"].notna()].copy()
    prepared["label"] = prepared["label"].replace({"Stationary": "Normal Driving"})
    feature_columns = model_feature_columns(prepared)
    encoder = LabelEncoder().fit(prepared["label"].astype(str))
    y = encoder.transform(prepared["label"].astype(str))
    weights = compute_sample_weight(class_weight="balanced", y=y)
    estimator = _estimators(config, len(encoder.classes_))["xgboost"]
    estimator.fit(prepared[feature_columns], y, sample_weight=weights)
    return ModelBundle(
        name="xgboost",
        estimator=estimator,
        label_encoder=encoder,
        feature_columns=feature_columns,
        thresholds={label: 0.5 for label in encoder.classes_},
        metadata={
            "status": "fit_on_all_labeled_sample_rows",
            "training_rows": int(len(prepared)),
            "labeled_sessions": int(prepared["session_id"].nunique()),
            "validation_scope": "single_session_exploratory",
        },
    )
