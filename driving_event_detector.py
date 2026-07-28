"""Command-line interface for the rate-aware streaming event detector."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from driving_events.config import load_config
from driving_events.data import load_sensor_csv
from driving_events.streaming import replay_rule_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Sensor CSV to replay")
    parser.add_argument("--rate", type=float, required=True, help="Declared IMU sample rate in Hz")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional event CSV")
    parser.add_argument("--list", action="store_true", help="Print every de-duplicated event")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    frame = load_sensor_csv(
        args.csv,
        args.rate,
        session_id=args.session_id,
        physical_limits=config["data"]["physical_limits"],
    )
    events = replay_rule_detector(frame, args.rate, config)
    counts = Counter(events["label"]) if len(events) else Counter()
    print(f"{args.csv} (@{args.rate:g} Hz): {len(events)} unique event episodes")
    for label in config["project"]["event_labels"]:
        print(f"  {label:<20} {counts.get(label, 0)}")
    if args.list and len(events):
        print()
        print(
            events[
                ["start_time_s", "end_time_s", "label", "confidence", "severity"]
            ].to_string(index=False)
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
