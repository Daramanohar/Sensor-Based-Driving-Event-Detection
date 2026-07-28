from driving_events.config import load_config
from driving_events.data import EVENT_LABELS
from driving_events.event_state_machine import EventStateMachine


def _scores(label: str | None, value: float = 0.0) -> dict[str, float]:
    result = {item: 0.0 for item in EVENT_LABELS}
    if label:
        result[label] = value
    return result


def _synthetic_event(rate_hz: float):
    config = load_config()
    machine = EventStateMachine(
        config,
        session_id=f"synthetic_{rate_hz:g}",
        source="test",
    )
    duration_s = 3.0
    for index in range(round(duration_s * rate_hz)):
        time_s = index / rate_hz
        active = 1.0 <= time_s <= 1.7
        machine.update(
            time_s,
            _scores("Harsh Braking" if active else None, 0.90 if active else 0.0),
        )
    machine.flush()
    return machine.events


def test_state_machine_is_rate_invariant_and_deduplicates() -> None:
    slow = _synthetic_event(10.0)
    fast = _synthetic_event(50.0)

    assert len(slow) == len(fast) == 1
    assert slow[0].label == fast[0].label == "Harsh Braking"
    assert abs(slow[0].start_time_s - fast[0].start_time_s) <= 0.02
    assert abs(slow[0].end_time_s - fast[0].end_time_s) <= 0.02


def test_overlong_clutch_signal_does_not_repeat_until_signal_clears() -> None:
    config = load_config()
    machine = EventStateMachine(config, session_id="long_clutch", source="test")
    for index in range(150):
        machine.update(index / 50.0, _scores("Clutch Release", 0.90))
    machine.flush()

    assert len(machine.events) == 1
    assert machine.events[0].duration_s <= 1.2
