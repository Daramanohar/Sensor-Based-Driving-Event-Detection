"""Causal sensor preprocessing shared by feature extraction and replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

ACCEL_COLUMNS = ("accel_x_g", "accel_y_g", "accel_z_g")
GYRO_COLUMNS = ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps")


def _causal_lowpass(values: np.ndarray, rate_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Apply a Butterworth low-pass filter without future-data leakage."""

    if len(values) == 0:
        return values.copy()
    nyquist = rate_hz / 2.0
    effective_cutoff = min(float(cutoff_hz), nyquist * 0.80)
    if effective_cutoff <= 0:
        raise ValueError("Effective low-pass cutoff must be positive")
    sos = butter(order, effective_cutoff, btype="lowpass", fs=rate_hz, output="sos")
    zi = sosfilt_zi(sos) * float(values[0])
    filtered, _ = sosfilt(sos, values, zi=zi)
    return filtered


def _filled_numeric(series: pd.Series) -> np.ndarray:
    """Fill isolated invalid readings only for filter continuity.

    The original `is_valid` flag is retained, so filled points can be excluded from training and
    cannot silently become ground truth.
    """

    clean = series.replace([np.inf, -np.inf], np.nan)
    clean = clean.interpolate(limit_direction="both").ffill().bfill()
    if clean.isna().any():
        raise ValueError(f"Column {series.name} has no finite values")
    return clean.to_numpy(dtype=float)


def preprocess_sensor_frame(
    frame: pd.DataFrame,
    sample_rate_hz: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calibrate fixed-mount bias, estimate gravity, filter, and derive causal jerk."""

    result = frame.copy()
    settings = config["preprocessing"]
    baseline_rows = max(1, int(round(float(settings["baseline_seconds"]) * sample_rate_hz)))
    baseline_slice = result.iloc[:baseline_rows]
    stationary = baseline_slice.loc[baseline_slice["gps_speed_mps"].lt(0.5)]
    if len(stationary) < max(3, baseline_rows // 4):
        stationary = baseline_slice

    x_bias = float(stationary["accel_x_g"].median())
    y_gravity = float(stationary["accel_y_g"].median())
    z_bias = float(stationary["accel_z_g"].median())
    result["accel_x_cal_g"] = result["accel_x_g"] - x_bias
    result["accel_y_delta_g"] = result["accel_y_g"] - y_gravity
    result["accel_z_cal_g"] = result["accel_z_g"] - z_bias

    tau = float(settings["gravity_time_constant_s"])
    alpha = (1.0 / sample_rate_hz) / (tau + 1.0 / sample_rate_hz)
    result["gravity_y_g"] = result["accel_y_g"].ewm(alpha=alpha, adjust=False).mean()
    result["accel_y_linear_g"] = result["accel_y_g"] - result["gravity_y_g"]

    filter_inputs = {
        "accel_x_filt_g": "accel_x_cal_g",
        "accel_y_filt_g": "accel_y_linear_g",
        "accel_z_filt_g": "accel_z_cal_g",
        "gyro_x_filt_dps": "gyro_x_dps",
        "gyro_y_filt_dps": "gyro_y_dps",
        "gyro_z_filt_dps": "gyro_z_dps",
    }
    for output, source in filter_inputs.items():
        result[output] = _causal_lowpass(
            _filled_numeric(result[source]),
            rate_hz=sample_rate_hz,
            cutoff_hz=float(settings["lowpass_cutoff_hz"]),
            order=int(settings["lowpass_order"]),
        )

    result["jerk_z_gps"] = result["accel_z_filt_g"].diff().fillna(0.0) * sample_rate_hz
    result["accel_magnitude_g"] = np.sqrt(
        result["accel_x_cal_g"] ** 2
        + result["accel_y_linear_g"] ** 2
        + result["accel_z_cal_g"] ** 2
    )
    result["gyro_magnitude_dps"] = np.sqrt(
        result["gyro_x_dps"] ** 2
        + result["gyro_y_dps"] ** 2
        + result["gyro_z_dps"] ** 2
    )

    result.attrs["calibration"] = {
        "accel_x_bias_g": x_bias,
        "accel_y_gravity_g": y_gravity,
        "accel_z_bias_g": z_bias,
        "filter_is_causal": True,
    }
    return result
