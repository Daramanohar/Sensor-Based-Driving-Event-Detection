"""Rate-invariant conversion of per-sample/window scores into unique event episodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import class_postprocessing_config
from .data import EVENT_LABELS


@dataclass(frozen=True)
class DetectedEvent:
    event_id: str
    session_id: str
    start_time_s: float
    end_time_s: float
    label: str
    confidence: float
    peak_time_s: float
    duration_s: float
    severity: str
    source: str


class EventStateMachine:
    """Single-active-event state machine with hysteresis and temporal merging.

    All timing is expressed in seconds, so behavior does not silently change when the sensor rate
    changes. Low-score gaps shorter than `merge_gap_s` remain part of the same physical event.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        session_id: str,
        source: str,
        labels: tuple[str, ...] = EVENT_LABELS,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.source = source
        self.labels = labels
        self.mode = "idle"
        self.label: str | None = None
        self.start_time_s = 0.0
        self.last_above_time_s = 0.0
        self.peak_time_s = 0.0
        self.peak_score = 0.0
        self.blocked_label_until_clear: str | None = None
        self._events: list[DetectedEvent] = []

    @property
    def events(self) -> tuple[DetectedEvent, ...]:
        return tuple(self._events)

    def _settings(self, label: str) -> dict[str, float]:
        return class_postprocessing_config(self.config, label)

    def _best(self, scores: Mapping[str, float]) -> tuple[str, float]:
        label = max(self.labels, key=lambda item: float(scores.get(item, 0.0)))
        return label, float(scores.get(label, 0.0))

    def _begin_confirmation(self, time_s: float, label: str, score: float) -> None:
        self.mode = "confirming"
        self.label = label
        self.start_time_s = time_s
        self.last_above_time_s = time_s
        self.peak_time_s = time_s
        self.peak_score = score
        if self._settings(label).get("min_duration_s", 0.0) <= 0.0:
            self.mode = "active"

    def _reset(self) -> None:
        self.mode = "idle"
        self.label = None
        self.peak_score = 0.0

    def _severity(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "strong"
        if confidence >= 0.70:
            return "moderate"
        return "weak"

    def _finalize(self, *, block_until_clear: bool = False) -> DetectedEvent | None:
        if self.mode != "active" or self.label is None:
            self._reset()
            return None
        end_time = max(self.last_above_time_s, self.start_time_s)
        event = DetectedEvent(
            event_id=f"{self.session_id}_pred_{len(self._events) + 1:04d}",
            session_id=self.session_id,
            start_time_s=round(self.start_time_s, 6),
            end_time_s=round(end_time, 6),
            label=self.label,
            confidence=round(float(self.peak_score), 6),
            peak_time_s=round(self.peak_time_s, 6),
            duration_s=round(max(0.0, end_time - self.start_time_s), 6),
            severity=self._severity(self.peak_score),
            source=self.source,
        )
        self._events.append(event)
        blocked_label = self.label if block_until_clear else None
        self._reset()
        self.blocked_label_until_clear = blocked_label
        return event

    def update(self, time_s: float, scores: Mapping[str, float]) -> DetectedEvent | None:
        """Consume one ordered score vector and optionally close one event."""

        time_s = float(time_s)
        usable_scores = dict(scores)
        if self.blocked_label_until_clear is not None:
            blocked_settings = self._settings(self.blocked_label_until_clear)
            if float(scores.get(self.blocked_label_until_clear, 0.0)) < blocked_settings[
                "offset_threshold"
            ]:
                self.blocked_label_until_clear = None
            else:
                usable_scores[self.blocked_label_until_clear] = 0.0
        best_label, best_score = self._best(usable_scores)

        if self.mode == "idle":
            if best_score >= self._settings(best_label)["onset_threshold"]:
                self._begin_confirmation(time_s, best_label, best_score)
            return None

        assert self.label is not None
        settings = self._settings(self.label)
        active_score = float(scores.get(self.label, 0.0))

        if active_score >= settings["offset_threshold"]:
            self.last_above_time_s = time_s
            if active_score > self.peak_score:
                self.peak_score = active_score
                self.peak_time_s = time_s
            if self.mode == "confirming":
                if time_s - self.start_time_s >= settings.get("min_duration_s", 0.0):
                    self.mode = "active"
            max_duration = settings.get("max_duration_s")
            if self.mode == "active" and max_duration is not None:
                if time_s - self.start_time_s >= max_duration:
                    return self._finalize(block_until_clear=True)
            return None

        if self.mode == "confirming":
            self._reset()
            if best_score >= self._settings(best_label)["onset_threshold"]:
                self._begin_confirmation(time_s, best_label, best_score)
            return None

        if time_s - self.last_above_time_s >= settings["merge_gap_s"]:
            closed = self._finalize()
            if best_score >= self._settings(best_label)["onset_threshold"]:
                self._begin_confirmation(time_s, best_label, best_score)
            return closed
        return None

    def flush(self) -> DetectedEvent | None:
        """Finalize an active event at end-of-stream."""

        return self._finalize()


def scores_to_events(
    scores: pd.DataFrame,
    config: dict[str, Any],
    *,
    session_id: str,
    source: str,
) -> pd.DataFrame:
    """Run the state machine over an ordered score table."""

    machine = EventStateMachine(config, session_id=session_id, source=source)
    ordered_scores = scores.loc[:, EVENT_LABELS].to_numpy(dtype=float)
    for time_s, values in zip(
        scores["elapsed_s"].to_numpy(dtype=float),
        ordered_scores,
        strict=True,
    ):
        mapping = dict(zip(EVENT_LABELS, values, strict=True))
        machine.update(float(time_s), mapping)
    machine.flush()
    columns = list(DetectedEvent.__dataclass_fields__)
    return pd.DataFrame([asdict(event) for event in machine.events], columns=columns)
