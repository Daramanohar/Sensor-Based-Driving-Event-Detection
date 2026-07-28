"""True one-sample-at-a-time rule baseline for live replay and GPS-gap tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .data import EVENT_LABELS
from .event_state_machine import DetectedEvent, EventStateMachine


def _evidence(value: float, reference: float) -> float:
    return float(np.clip(max(value, 0.0) / reference * 0.55, 0.0, 1.0))


def _rms(values: deque[float]) -> float:
    return sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


class StreamingRuleDetector:
    """Rate-aware rule scorer plus event state machine.

    It intentionally uses only current/past samples and treats stale GPS as unknown rather than
    as a blocking condition.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        config: dict[str, Any],
        *,
        session_id: str,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.config = config
        rules = config["rules"]
        self.harsh_window = max(
            1, int(np.ceil(float(rules["harsh_min_duration_s"]) * self.rate_hz))
        )
        self.clutch_window = max(
            1, int(np.ceil(float(rules["clutch_window_s"]) * self.rate_hz))
        )
        self.az_history: deque[float] = deque(maxlen=self.harsh_window)
        self.gx_history: deque[float] = deque(maxlen=self.clutch_window)
        self.gz_history: deque[float] = deque(maxlen=self.clutch_window)
        self.speed_history: deque[float] = deque(maxlen=max(2, round(self.rate_hz)))
        self.last_gps_timestamp: str | None = None
        self.last_speed = 0.0
        self.gps_age_s = float("inf")
        self.row_index = -1
        self.machine = EventStateMachine(
            config,
            session_id=session_id,
            source="streaming_rule_v2",
        )

    def _valid(self, sample: Mapping[str, Any]) -> bool:
        limits = self.config["data"]["physical_limits"]
        numeric = [
            float(sample[column])
            for column in (
                "accel_x_g",
                "accel_y_g",
                "accel_z_g",
                "gyro_x_dps",
                "gyro_y_dps",
                "gyro_z_dps",
                "gps_speed_mps",
            )
        ]
        if not all(isfinite(value) for value in numeric):
            return False
        ax, ay, az, gx, gy, gz, speed = numeric
        return (
            max(abs(ax), abs(ay), abs(az)) <= float(limits["acceleration_g"])
            and max(abs(gx), abs(gy), abs(gz)) <= float(limits["gyro_dps"])
            and 0.0 <= speed <= float(limits["speed_mps"])
        )

    def _scores(self, sample: Mapping[str, Any]) -> dict[str, float]:
        rules = self.config["rules"]
        az = float(sample["accel_z_g"])
        ay = float(sample["accel_y_g"])
        gx = float(sample["gyro_x_dps"])
        gz = float(sample["gyro_z_dps"])
        self.az_history.append(az)
        self.gx_history.append(gx)
        self.gz_history.append(gz)

        accel_threshold = float(rules["harsh_accel_z_g"])
        brake_threshold = float(rules["harsh_brake_z_g"])
        positive_persistence = np.mean(
            [value > accel_threshold * 0.80 for value in self.az_history]
        )
        negative_persistence = np.mean(
            [value < -brake_threshold * 0.80 for value in self.az_history]
        )
        acceleration = _evidence(az, accel_threshold) * float(positive_persistence)
        braking = _evidence(-az, brake_threshold) * float(negative_persistence)

        self.speed_history.append(self.last_speed)
        speed_delta = self.speed_history[-1] - self.speed_history[0]
        gps_recent = self.gps_age_s <= float(self.config["data"]["max_gps_age_s"])
        if gps_recent and speed_delta < -1.0:
            acceleration *= 0.25
        if gps_recent and speed_delta > 1.0:
            braking *= 0.25

        vertical = _evidence(abs(ay - 1.0), float(rules["pothole_vertical_delta_g"]))
        pitch_roll = _evidence(max(abs(gx), abs(gz)), float(rules["pothole_gyro_dps"]))
        pothole = sqrt(vertical * pitch_roll)

        broadband = _evidence(
            min(_rms(self.gx_history), _rms(self.gz_history)),
            float(rules["clutch_gyro_rms_dps"]),
        )
        clutch_az_limit = float(rules["clutch_max_abs_longitudinal_g"])
        directional_penalty = float(
            np.clip((clutch_az_limit - abs(az)) / (0.5 * clutch_az_limit), 0.0, 1.0)
        )
        speed_factor = (
            0.20
            if gps_recent and self.last_speed >= float(rules["clutch_max_speed_mps"])
            else 1.0
        )
        pothole_suppression = 1.0 - 0.85 * float(
            np.clip((vertical - 0.55) / 0.45, 0.0, 1.0)
        )
        clutch = float(
            np.clip(
                broadband * directional_penalty * speed_factor * pothole_suppression,
                0.0,
                1.0,
            )
        )
        return {
            "Harsh Braking": braking,
            "Harsh Acceleration": acceleration,
            "Pothole/Bump": pothole,
            "Clutch Release": clutch,
        }

    def push(self, sample: Mapping[str, Any]) -> DetectedEvent | None:
        self.row_index += 1
        time_s = self.row_index / self.rate_hz
        gps_timestamp = str(sample.get("gps_timestamp", ""))
        if gps_timestamp != self.last_gps_timestamp:
            self.last_gps_timestamp = gps_timestamp
            self.last_speed = float(sample["gps_speed_mps"])
            self.gps_age_s = 0.0
        else:
            self.gps_age_s += self.dt

        if not self._valid(sample):
            return self.machine.update(time_s, {label: 0.0 for label in EVENT_LABELS})
        return self.machine.update(time_s, self._scores(sample))

    def flush(self) -> DetectedEvent | None:
        return self.machine.flush()

    @property
    def events(self) -> tuple[DetectedEvent, ...]:
        return self.machine.events


def replay_rule_detector(
    frame: pd.DataFrame,
    sample_rate_hz: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    session_id = str(frame["session_id"].iloc[0])
    detector = StreamingRuleDetector(sample_rate_hz, config, session_id=session_id)
    for sample in frame.to_dict(orient="records"):
        detector.push(sample)
    detector.flush()
    columns = list(DetectedEvent.__dataclass_fields__)
    return pd.DataFrame([asdict(event) for event in detector.events], columns=columns)
