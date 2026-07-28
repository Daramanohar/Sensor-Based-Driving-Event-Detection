"""Interpretable, rate-aware scores for real-time driving-event detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import EVENT_LABELS


def _rolling_rms(series: pd.Series, window: int) -> pd.Series:
    return series.pow(2).rolling(window, min_periods=1).mean().pow(0.5)


def _evidence(value: pd.Series, reference: float) -> pd.Series:
    """Map a physical threshold to score 0.55 while retaining graded confidence."""

    return (value.clip(lower=0.0) / reference * 0.55).clip(0.0, 1.0)


def rule_score_frame(
    frame: pd.DataFrame,
    sample_rate_hz: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return one interpretable score per event class for every sample.

    GPS never creates an event. A recent, contradictory GPS observation can reduce confidence;
    stale GPS is neutral and is explicitly represented by `gps_age_s`.
    """

    rules = config["rules"]
    max_gps_age = float(config["data"]["max_gps_age_s"])
    harsh_window = max(1, int(np.ceil(float(rules["harsh_min_duration_s"]) * sample_rate_hz)))
    clutch_window = max(1, int(np.ceil(float(rules["clutch_window_s"]) * sample_rate_hz)))

    az = frame.get("accel_z_filt_g", frame["accel_z_g"])
    ay_delta = frame.get("accel_y_delta_g", frame["accel_y_g"] - 1.0)
    gx = frame["gyro_x_dps"]
    gz = frame["gyro_z_dps"]

    accel_threshold = float(rules["harsh_accel_z_g"])
    brake_threshold = float(rules["harsh_brake_z_g"])
    positive_persistence = (
        az.gt(accel_threshold * 0.80).rolling(harsh_window, min_periods=1).mean()
    )
    negative_persistence = (
        az.lt(-brake_threshold * 0.80).rolling(harsh_window, min_periods=1).mean()
    )
    acceleration = _evidence(az, accel_threshold) * positive_persistence
    braking = _evidence(-az, brake_threshold) * negative_persistence

    speed_delta = frame["gps_speed_mps"].diff(periods=max(1, round(sample_rate_hz))).fillna(0.0)
    gps_recent = frame["gps_age_s"].le(max_gps_age)
    accel_contradiction = gps_recent & speed_delta.lt(-1.0)
    brake_contradiction = gps_recent & speed_delta.gt(1.0)
    acceleration = acceleration.where(~accel_contradiction, acceleration * 0.25)
    braking = braking.where(~brake_contradiction, braking * 0.25)

    vertical = _evidence(ay_delta.abs(), float(rules["pothole_vertical_delta_g"]))
    pitch_roll = _evidence(
        pd.concat([gx.abs(), gz.abs()], axis=1).max(axis=1),
        float(rules["pothole_gyro_dps"]),
    )
    pothole = np.sqrt(vertical * pitch_roll).clip(0.0, 1.0)

    gx_rms = _rolling_rms(gx, clutch_window)
    gz_rms = _rolling_rms(gz, clutch_window)
    broadband = _evidence(
        pd.concat([gx_rms, gz_rms], axis=1).min(axis=1),
        float(rules["clutch_gyro_rms_dps"]),
    )
    clutch_az_limit = max(
        float(rules["clutch_max_abs_longitudinal_g"]), np.finfo(float).eps
    )
    directional_penalty = ((clutch_az_limit - az.abs()) / (0.5 * clutch_az_limit)).clip(
        0.0, 1.0
    )
    recent_high_speed = gps_recent & frame["gps_speed_mps"].ge(
        float(rules["clutch_max_speed_mps"])
    )
    gps_factor = pd.Series(1.0, index=frame.index).where(~recent_high_speed, 0.20)
    pothole_suppression = 1.0 - 0.85 * ((vertical - 0.55) / 0.45).clip(0.0, 1.0)
    clutch = (broadband * directional_penalty * gps_factor * pothole_suppression).clip(0.0, 1.0)

    scores = pd.DataFrame(
        {
            "elapsed_s": frame["elapsed_s"].to_numpy(),
            "Harsh Braking": braking.to_numpy(),
            "Harsh Acceleration": acceleration.to_numpy(),
            "Pothole/Bump": pothole.to_numpy(),
            "Clutch Release": clutch.to_numpy(),
        },
        index=frame.index,
    )
    scores.loc[~frame["is_valid"], list(EVENT_LABELS)] = 0.0
    return scores
