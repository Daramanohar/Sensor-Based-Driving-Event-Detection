from pathlib import Path

import numpy as np

from driving_events.config import load_config
from driving_events.data import label_samples, load_annotations, load_sensor_csv
from driving_events.features import extract_feature_table, model_feature_columns
from driving_events.preprocessing import preprocess_sensor_frame

ROOT = Path(__file__).resolve().parents[1]


def test_feature_table_is_finite_and_causal_shape_is_stable() -> None:
    config = load_config(ROOT / "config.yaml")
    frame = load_sensor_csv(
        ROOT / "data/sample_sensor_data.csv",
        10.0,
        session_id="sample_10hz",
        physical_limits=config["data"]["physical_limits"],
    )
    annotations = load_annotations(ROOT / "data/sample_labels.csv")
    labeled = label_samples(frame, annotations, require_complete_coverage=True)
    processed = preprocess_sensor_frame(labeled, 10.0, config)
    features = extract_feature_table(processed, 10.0, config)
    predictors = model_feature_columns(features)

    assert len(features) == 258
    assert 80 <= len(predictors) <= 150
    assert np.isfinite(features[predictors].to_numpy()).all()
    assert features["label"].isna().sum() == 0
