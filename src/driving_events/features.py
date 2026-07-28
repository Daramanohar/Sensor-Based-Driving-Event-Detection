"""Causal multi-scale feature extraction for compact tabular models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = (
    "accel_x_filt_g",
    "accel_y_filt_g",
    "accel_z_filt_g",
    "accel_y_delta_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "jerk_z_gps",
    "accel_magnitude_g",
    "gyro_magnitude_dps",
)
ZERO_CROSSING_COLUMNS = ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps", "jerk_z_gps")
SPECTRAL_COLUMNS = ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps")


def _spectral_features(values: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    if len(values) < 4:
        return 0.0, 0.0
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
    if float(power.sum()) <= np.finfo(float).eps:
        return 0.0, 0.0
    dominant_hz = float(frequencies[int(np.argmax(power))])
    high_boundary = min(2.0, sample_rate_hz * 0.20)
    high_ratio = float(power[frequencies >= high_boundary].sum() / power.sum())
    return dominant_hz, high_ratio


def extract_feature_table(
    frame: pd.DataFrame,
    sample_rate_hz: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Extract one row of features per causal hop.

    Every window ends at `elapsed_s`; no centered/future samples are used. Label metadata is
    copied from the same end sample when annotations were attached beforehand.
    """

    settings = config["features"]
    windows_s = tuple(float(value) for value in settings["windows_s"])
    hop = max(1, int(round(float(settings["hop_s"]) * sample_rate_hz)))
    end_indices = np.arange(0, len(frame), hop, dtype=int)
    arrays = {column: frame[column].to_numpy(dtype=float) for column in SIGNAL_COLUMNS}
    speed = frame["gps_speed_mps"].to_numpy(dtype=float)
    gps_age = frame["gps_age_s"].to_numpy(dtype=float)
    valid = frame["is_valid"].to_numpy(dtype=bool)

    output: dict[str, object] = {
        "session_id": frame["session_id"].iloc[end_indices].astype(str).to_numpy(),
        "row_id": frame["row_id"].iloc[end_indices].to_numpy(dtype=int),
        "elapsed_s": frame["elapsed_s"].iloc[end_indices].to_numpy(dtype=float),
        "gps_speed_mps": speed[end_indices],
        "gps_age_s": gps_age[end_indices],
        "gps_is_recent": (
            gps_age[end_indices] <= float(config["data"]["max_gps_age_s"])
        ).astype(float),
    }
    if "label" in frame:
        for column in ("label", "event_id", "severity"):
            output[column] = frame[column].iloc[end_indices].to_numpy()

    for window_s in windows_s:
        samples = max(1, int(round(window_s * sample_rate_hz)))
        suffix = f"w{window_s:g}s"
        starts = np.maximum(0, end_indices - samples + 1)
        actual = end_indices - starts + 1
        output[f"history_fraction_{suffix}"] = actual / samples
        output[f"valid_fraction_{suffix}"] = (
            pd.Series(valid.astype(float))
            .rolling(samples, min_periods=1)
            .mean()
            .to_numpy()[end_indices]
        )

        for column, values in arrays.items():
            series = pd.Series(values)
            rolling = series.rolling(samples, min_periods=1)
            statistics = {
                "mean": rolling.mean().to_numpy(),
                "std": rolling.std(ddof=0).fillna(0.0).to_numpy(),
                "rms": series.pow(2).rolling(samples, min_periods=1).mean().pow(0.5).to_numpy(),
                "min": rolling.min().to_numpy(),
                "max": rolling.max().to_numpy(),
                "absmax": series.abs().rolling(samples, min_periods=1).max().to_numpy(),
                "p10": rolling.quantile(0.10).to_numpy(),
                "p90": rolling.quantile(0.90).to_numpy(),
            }
            statistics["range"] = statistics["max"] - statistics["min"]
            for statistic, values_by_row in statistics.items():
                output[f"{column}_{statistic}_{suffix}"] = values_by_row[end_indices]

            if column in ZERO_CROSSING_COLUMNS:
                crossing = np.zeros(len(values), dtype=float)
                crossing[1:] = np.signbit(values[1:]) != np.signbit(values[:-1])
                output[f"{column}_zcr_{suffix}"] = (
                    pd.Series(crossing)
                    .rolling(samples, min_periods=1)
                    .mean()
                    .to_numpy()[end_indices]
                )

            if column in SPECTRAL_COLUMNS and window_s >= float(
                settings["spectral_min_window_s"]
            ):
                spectral = [
                    _spectral_features(values[start : end + 1], sample_rate_hz)
                    for start, end in zip(starts, end_indices, strict=True)
                ]
                output[f"{column}_dominant_hz_{suffix}"] = np.fromiter(
                    (item[0] for item in spectral), dtype=float
                )
                output[f"{column}_high_band_ratio_{suffix}"] = np.fromiter(
                    (item[1] for item in spectral), dtype=float
                )

        speed_delta = speed[end_indices] - speed[starts]
        duration = np.maximum((actual - 1) / sample_rate_hz, 1.0 / sample_rate_hz)
        output[f"gps_speed_delta_{suffix}"] = speed_delta
        output[f"gps_speed_slope_{suffix}"] = speed_delta / duration
        z_series = pd.Series(arrays["accel_z_filt_g"])
        output[f"positive_az_fraction_{suffix}"] = (
            z_series.gt(0.20)
            .astype(float)
            .rolling(samples, min_periods=1)
            .mean()
            .to_numpy()[end_indices]
        )
        output[f"negative_az_fraction_{suffix}"] = (
            z_series.lt(-0.20)
            .astype(float)
            .rolling(samples, min_periods=1)
            .mean()
            .to_numpy()[end_indices]
        )

    features = pd.DataFrame(output)
    numeric = features.select_dtypes(include=[np.number]).columns
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def model_feature_columns(features: pd.DataFrame) -> list[str]:
    """Select a physics-informed subset to control dimensionality on limited labels.

    The full table remains available for analysis. The model deliberately avoids feeding every
    correlated rolling statistic into a 258-row experiment.
    """

    selected: list[str] = []
    direct = {"gps_speed_mps", "gps_age_s", "gps_is_recent"}
    signal_statistics = {
        "accel_x_filt_g": ("rms", "absmax"),
        "accel_y_delta_g": ("min", "max", "range"),
        "accel_z_filt_g": ("mean", "min", "max", "rms"),
        "gyro_x_dps": ("rms", "absmax", "zcr"),
        "gyro_y_dps": ("mean", "rms", "absmax"),
        "gyro_z_dps": ("rms", "absmax", "zcr"),
        "jerk_z_gps": ("rms", "min", "max", "absmax"),
        "gyro_magnitude_dps": ("rms",),
    }
    numeric_columns = set(features.select_dtypes(include=[np.number]).columns)
    for column in features.columns:
        if column not in numeric_columns:
            continue
        if column in direct:
            selected.append(column)
            continue
        if column.startswith(
            (
                "gps_speed_delta_",
                "gps_speed_slope_",
                "positive_az_fraction_",
                "negative_az_fraction_",
                "valid_fraction_",
                "history_fraction_",
            )
        ):
            selected.append(column)
            continue
        if "high_band_ratio" in column and column.startswith(("gyro_x_dps", "gyro_z_dps")):
            selected.append(column)
            continue
        for signal, statistics in signal_statistics.items():
            if column.startswith(f"{signal}_") and any(
                f"_{statistic}_w" in column for statistic in statistics
            ):
                selected.append(column)
                break
    return selected
