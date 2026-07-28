import pandas as pd

from driving_events.evaluation import event_level_metrics


def test_duplicate_prediction_is_counted_as_false_positive() -> None:
    truth = pd.DataFrame(
        [
            {
                "event_id": "truth_1",
                "session_id": "s1",
                "start_time_s": 5.0,
                "end_time_s": 5.6,
                "label": "Harsh Braking",
                "severity": "strong",
                "review_status": "reviewed",
                "annotation_source": "review",
                "notes": "",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "start_time_s": 5.1,
                "end_time_s": 5.5,
                "label": "Harsh Braking",
            },
            {
                "session_id": "s1",
                "start_time_s": 5.2,
                "end_time_s": 5.4,
                "label": "Harsh Braking",
            },
        ]
    )
    metrics = event_level_metrics(truth, predictions, session_duration_s=60.0)
    braking = metrics["per_class"]["Harsh Braking"]

    assert braking["tp"] == 1
    assert braking["fp"] == 1
    assert braking["fn"] == 0
    assert braking["precision"] == 0.5
