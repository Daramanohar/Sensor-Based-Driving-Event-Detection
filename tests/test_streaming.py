from collections import Counter
from pathlib import Path

from driving_events.config import load_config
from driving_events.data import load_sensor_csv
from driving_events.streaming import StreamingRuleDetector, replay_rule_detector

ROOT = Path(__file__).resolve().parents[1]


def _sample(gps_timestamp: str, speed: float, gx: float = 0.0, gz: float = 0.0):
    return {
        "timestamp": gps_timestamp,
        "gps_timestamp": gps_timestamp,
        "gps_speed_mps": speed,
        "accel_x_g": 0.0,
        "accel_y_g": 1.0,
        "accel_z_g": 0.0,
        "gyro_x_dps": gx,
        "gyro_y_dps": 0.0,
        "gyro_z_dps": gz,
    }


def test_reviewed_excerpt_produces_two_unique_events_per_class() -> None:
    config = load_config(ROOT / "config.yaml")
    frame = load_sensor_csv(
        ROOT / "data/sample_sensor_data.csv",
        10.0,
        session_id="sample_10hz",
        physical_limits=config["data"]["physical_limits"],
    )
    events = replay_rule_detector(frame, 10.0, config)
    counts = Counter(events["label"])
    assert counts == {
        "Harsh Braking": 2,
        "Harsh Acceleration": 2,
        "Pothole/Bump": 2,
        "Clutch Release": 2,
    }


def test_stale_high_gps_speed_does_not_block_clutch_burst() -> None:
    config = load_config(ROOT / "config.yaml")
    detector = StreamingRuleDetector(50.0, config, session_id="gps_gap")
    detector.push(_sample("00:00.0", 10.0))
    for _ in range(110):
        stale = _sample("00:00.0", 10.0)
        detector.push(stale)
    for _ in range(10):
        detector.push(_sample("00:00.0", 10.0, gx=10.0, gz=10.0))
    for _ in range(40):
        detector.push(_sample("00:00.0", 10.0))
    detector.flush()
    assert any(event.label == "Clutch Release" for event in detector.events)


def test_recent_high_speed_reduces_clutch_false_positive() -> None:
    config = load_config(ROOT / "config.yaml")
    detector = StreamingRuleDetector(50.0, config, session_id="fresh_gps")
    for index in range(20):
        detector.push(
            _sample(
                f"00:00.{index}",
                10.0,
                gx=10.0,
                gz=10.0,
            )
        )
    detector.flush()
    assert not any(event.label == "Clutch Release" for event in detector.events)
