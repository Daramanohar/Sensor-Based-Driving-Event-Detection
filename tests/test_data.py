from pathlib import Path

from driving_events.config import load_config
from driving_events.data import label_samples, load_annotations, load_sensor_csv
from driving_events.labeling import annotation_coverage

ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_sample_annotations_cover_every_row() -> None:
    config = load_config(ROOT / "config.yaml")
    frame = load_sensor_csv(
        ROOT / "data/sample_sensor_data.csv",
        10.0,
        session_id="sample_10hz",
        physical_limits=config["data"]["physical_limits"],
    )
    annotations = load_annotations(ROOT / "data/sample_labels.csv")
    labeled = label_samples(frame, annotations, require_complete_coverage=True)
    coverage = annotation_coverage(frame, labeled)

    assert len(frame) == 258
    assert coverage["coverage"] == 1.0
    assert coverage["unlabeled_rows"] == 0
    assert frame["gps_is_fresh"].sum() == 26
    assert labeled.loc[labeled["elapsed_s"].eq(10.8), "label"].item() == "Normal Driving"


def test_load_sensor_csv_rejects_nonpositive_rate() -> None:
    config = load_config(ROOT / "config.yaml")
    try:
        load_sensor_csv(
            ROOT / "data/sample_sensor_data.csv",
            0.0,
            physical_limits=config["data"]["physical_limits"],
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected a nonpositive rate to be rejected")
