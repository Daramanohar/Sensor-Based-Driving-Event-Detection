"""Sensor and annotation I/O with auditable data-quality checks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_SENSOR_COLUMNS = (
    "timestamp",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "gps_speed_mps",
    "gps_timestamp",
)
NUMERIC_SENSOR_COLUMNS = (
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "gps_speed_mps",
)
EVENT_LABELS = (
    "Harsh Braking",
    "Harsh Acceleration",
    "Pothole/Bump",
    "Clutch Release",
)
ALL_LABELS = EVENT_LABELS + ("Normal Driving", "Stationary")
ANNOTATION_COLUMNS = (
    "event_id",
    "session_id",
    "start_time_s",
    "end_time_s",
    "label",
    "severity",
    "review_status",
    "annotation_source",
    "notes",
)


def timestamp_to_seconds(value: str) -> float:
    """Parse `MM:SS.s`, `HH:MM:SS.s`, or a plain number into seconds."""

    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Unsupported timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    return float(sum(number * (60**power) for power, number in enumerate(reversed(numbers))))


def load_sensor_csv(
    path: str | Path,
    sample_rate_hz: float,
    session_id: str | None = None,
    physical_limits: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Load one sensor session and add monotonic time/GPS-freshness fields.

    `elapsed_s` is derived from row order and the declared sample rate. This is intentional:
    the provided 50 Hz file rounds display timestamps to 0.1 s, so five rows can share the same
    timestamp even though the samples are ordered at 50 Hz.
    """

    csv_path = Path(path)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    frame = pd.read_csv(csv_path)
    missing_columns = set(REQUIRED_SENSOR_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{csv_path} contains no sensor rows")

    frame = frame.loc[:, REQUIRED_SENSOR_COLUMNS].copy()
    for column in NUMERIC_SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    frame.insert(1, "session_id", session_id or csv_path.stem)
    frame.insert(2, "elapsed_s", np.arange(len(frame), dtype=float) / float(sample_rate_hz))
    frame["display_time_s"] = frame["timestamp"].map(timestamp_to_seconds)
    frame["gps_display_time_s"] = frame["gps_timestamp"].map(timestamp_to_seconds)

    if (frame["display_time_s"].diff().dropna() < 0).any():
        raise ValueError(f"{csv_path} has decreasing sensor timestamps")
    if (frame["gps_display_time_s"].diff().dropna() < 0).any():
        raise ValueError(f"{csv_path} has decreasing GPS timestamps")

    frame["gps_is_fresh"] = frame["gps_timestamp"].ne(frame["gps_timestamp"].shift())
    fresh_positions = np.where(frame["gps_is_fresh"].to_numpy(), frame["row_id"], -1)
    last_fresh = np.maximum.accumulate(fresh_positions)
    frame["gps_age_s"] = (frame["row_id"].to_numpy() - last_fresh) / float(sample_rate_hz)

    finite = np.isfinite(frame[list(NUMERIC_SENSOR_COLUMNS)].to_numpy()).all(axis=1)
    valid = finite.copy()
    if physical_limits:
        accel_limit = float(physical_limits["acceleration_g"])
        gyro_limit = float(physical_limits["gyro_dps"])
        speed_limit = float(physical_limits["speed_mps"])
        valid &= (
            frame[["accel_x_g", "accel_y_g", "accel_z_g"]].abs().max(axis=1).to_numpy()
            <= accel_limit
        )
        valid &= (
            frame[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].abs().max(axis=1).to_numpy()
            <= gyro_limit
        )
        valid &= frame["gps_speed_mps"].between(0.0, speed_limit).to_numpy()
    frame["is_valid"] = valid
    return frame


def load_annotations(
    path: str | Path,
    *,
    reviewed_only: bool = False,
    allowed_labels: Iterable[str] = ALL_LABELS,
) -> pd.DataFrame:
    """Load and validate the interval annotation contract."""

    annotation_path = Path(path)
    annotations = pd.read_csv(annotation_path)
    missing = set(ANNOTATION_COLUMNS).difference(annotations.columns)
    if missing:
        raise ValueError(f"{annotation_path} is missing annotation columns: {sorted(missing)}")
    annotations = annotations.loc[:, ANNOTATION_COLUMNS].copy()
    annotations["start_time_s"] = pd.to_numeric(annotations["start_time_s"], errors="coerce")
    annotations["end_time_s"] = pd.to_numeric(annotations["end_time_s"], errors="coerce")
    if annotations[["start_time_s", "end_time_s"]].isna().any().any():
        raise ValueError(f"{annotation_path} contains invalid annotation times")
    if (annotations["end_time_s"] < annotations["start_time_s"]).any():
        raise ValueError(f"{annotation_path} contains an interval ending before it starts")
    invalid_labels = set(annotations["label"]).difference(allowed_labels)
    if invalid_labels:
        raise ValueError(f"{annotation_path} contains unsupported labels: {sorted(invalid_labels)}")
    invalid_status = set(annotations["review_status"]).difference(
        {"reviewed", "pending", "rejected"}
    )
    if invalid_status:
        raise ValueError(f"{annotation_path} contains invalid review statuses: {invalid_status}")
    if reviewed_only:
        annotations = annotations.loc[annotations["review_status"].eq("reviewed")].copy()
    return annotations.sort_values(["session_id", "start_time_s", "end_time_s"]).reset_index(
        drop=True
    )


def label_samples(
    frame: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    require_complete_coverage: bool = False,
) -> pd.DataFrame:
    """Attach interval labels to sensor rows and reject overlapping reviewed intervals."""

    result = frame.copy()
    result["label"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["event_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["severity"] = pd.Series(pd.NA, index=result.index, dtype="string")
    session_id = str(result["session_id"].iloc[0])
    relevant = annotations.loc[
        annotations["session_id"].eq(session_id) & annotations["review_status"].eq("reviewed")
    ]
    for row in relevant.itertuples(index=False):
        mask = result["elapsed_s"].between(row.start_time_s, row.end_time_s, inclusive="both")
        if result.loc[mask, "label"].notna().any():
            raise ValueError(f"Reviewed annotations overlap around event {row.event_id}")
        result.loc[mask, ["label", "event_id", "severity"]] = (
            row.label,
            row.event_id,
            row.severity,
        )
    if require_complete_coverage and result["label"].isna().any():
        uncovered = result.loc[result["label"].isna(), "elapsed_s"]
        raise ValueError(
            f"Annotations leave {len(uncovered)} samples uncovered; "
            f"first uncovered time={uncovered.iloc[0]:.3f}s"
        )
    return result


def profile_sensor_frame(frame: pd.DataFrame, sample_rate_hz: float) -> dict[str, object]:
    """Return JSON-serializable data-quality evidence."""

    numeric_ranges = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std()),
        }
        for column in NUMERIC_SENSOR_COLUMNS
    }
    gps_fix_rows = np.flatnonzero(frame["gps_is_fresh"].to_numpy())
    gps_gaps_s = np.diff(gps_fix_rows) / float(sample_rate_hz)
    return {
        "session_id": str(frame["session_id"].iloc[0]),
        "rows": int(len(frame)),
        "columns": list(REQUIRED_SENSOR_COLUMNS),
        "sample_rate_hz": float(sample_rate_hz),
        "duration_s": float(len(frame) / sample_rate_hz),
        "missing_values": {
            column: int(frame[column].isna().sum()) for column in REQUIRED_SENSOR_COLUMNS
        },
        "duplicate_rows": int(frame[list(REQUIRED_SENSOR_COLUMNS)].duplicated().sum()),
        "invalid_rows": int((~frame["is_valid"]).sum()),
        "gps_fixes": int(frame["gps_is_fresh"].sum()),
        "gps_gap_s": {
            "median": float(np.median(gps_gaps_s)) if len(gps_gaps_s) else None,
            "max": float(np.max(gps_gaps_s)) if len(gps_gaps_s) else None,
        },
        "display_timestamp_nonunique_rows": int(frame["timestamp"].duplicated().sum()),
        "numeric_ranges": numeric_ranges,
    }
