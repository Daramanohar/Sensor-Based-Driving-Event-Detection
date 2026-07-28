# Sensor-Based Driving Event Detection

**Author: Dara Manohar**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Daramanohar/Sensor-Based-Driving-Event-Detection/blob/main/solution.ipynb)

An end-to-end machine-learning solution for detecting driving events from smartphone
accelerometer, gyroscope, and GPS signals. The implementation combines a transparent
streaming detector with an XGBoost experiment, event-level evaluation, reproducible
configuration, automated tests, and saved results.

## Problem statement

The phone is fixed in portrait orientation and records:

- acceleration and angular velocity at 10-50 Hz;
- GPS speed at a lower, irregular rate;
- normal driving mixed with Harsh Braking, Harsh Acceleration, Pothole/Bump, and
  Clutch Release events.

The goal is to convert noisy, asynchronous sensor samples into one de-duplicated interval
per physical event. A useful solution must work causally, tolerate stale GPS, handle class
imbalance, avoid leakage between overlapping time windows, and report event quality rather
than relying on accuracy alone.

## Solution

The system has two complementary paths:

1. **Streaming detector:** causal filtering, physics-based signals, class scores, and a
   duration-aware state machine with onset/offset hysteresis and gap merging.
2. **XGBoost experiment:** 125 selected features derived from causal 0.25, 0.5, 1, and
   2-second windows, compared with Logistic Regression and Random Forest using event-group
   cross-validation.

GPS is optional evidence. Its value and age are tracked explicitly; stale GPS never blocks
a strong IMU event. All thresholds and model settings are stored in `config.yaml`.

## Reproduce the results

Python 3.11 or 3.12 is supported. Core numerical and ML packages are pinned to the versions used
for the reported results. For review and presentation, use the Colab button above with a standard
CPU runtime. A GPU is not needed for this dataset.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python run_pipeline.py
python -m pytest
```

Open `solution.ipynb` for the complete analysis, plots, equations, and
interpretation. The Colab badge opens it directly; the first notebook code cell clones or locates
the project and installs the package and test dependencies.

### What the executed notebook proves

The notebook does not merely print the final scores. Its saved outputs include:

- SHA-256 hashes for every input and result artifact;
- fail-fast assertions for schema, missing values, physical limits, and 100% label coverage;
- an annotated sensor timeline and the complete interval-label table;
- a causal prefix-invariance test with a maximum feature difference of `0.000e+00`;
- an audit showing that labels, timestamps, and event identifiers are excluded from features;
- one-to-one truth/detection matching with onset and end-time errors;
- exact train/test event IDs for every fold and zero group overlap;
- out-of-fold per-class metrics and an XGBoost confusion matrix;
- feature importance and full-session event visualizations;
- a second clean pipeline run proving byte-identical CSV/JSON results;
- visible Ruff and Pytest output.

Assertions stop notebook execution if any of these checks fail.

## Main results

| Experiment | Result | Correct interpretation |
|---|---:|---|
| Rule detector, labeled 25.7 s sample | 1.000 event macro-F1 | Sanity check on the same small sample used to set rules |
| XGBoost, two-fold event-group CV | 0.9186 accuracy, 0.7596 macro-F1 | Within-session separability estimate |
| Logistic Regression, grouped CV | 0.8372 accuracy, 0.6523 macro-F1 | Linear comparison baseline |
| Random Forest, grouped CV | 0.8915 accuracy, 0.6293 macro-F1 | Tree-ensemble comparison baseline |
| GPS gaps of 1-5 seconds | 1.000 rule macro-F1 | Confirms IMU-first behavior on the labeled sample |
| Unlabeled 30-minute drive | 89 detected episodes | Detection output only; no accuracy claim without labels |

The grouped split keeps every window from the same physical event in one fold. This is much
stricter than a random row split, although one short labeled session is still insufficient
to establish cross-driver or cross-device generalization.

## Repository layout

```text
.
|-- solution.ipynb                       # executable analysis and answers
|-- README.md                            # project overview and instructions
|-- SOLUTION.md                          # methodology and technical decisions
|-- config.yaml                          # all data, rule, feature, and model settings
|-- run_pipeline.py                      # one-command reproducible workflow
|-- driving_event_detector.py            # command-line streaming detector
|-- data/                                # supplied sessions and complete sample labels
|-- results/                             # concise, reproducible outputs
|-- src/driving_events/                  # reusable ML and streaming implementation
`-- tests/                               # data, feature, evaluation, and streaming tests
```

## Result artifacts

- `results/summary.json`: data profile, event metrics, robustness, model summary, limitations
- `results/model_comparison.csv`: identical grouped-CV comparison across three algorithms
- `results/model_evaluation.json`: fold membership, per-class reports, and confusion matrices
- `results/feature_importance.csv`: XGBoost feature importance
- `results/sample_events.csv`: detected episodes for the labeled sample
- `results/full_session_events.csv`: detected episodes for the unlabeled 30-minute session

The generated `results/xgboost_model.joblib` is intentionally ignored by Git because it is
fully reproducible from the committed code, configuration, data, and random seed.

## Honest scope

The project demonstrates a reliable engineering and experimentation pipeline, not a
production-generalization claim. The next decisive improvement is to label multiple complete
trips from different drivers, phones, mounts, vehicles, and road conditions. Those sessions
should be split by driver/trip, followed by probability calibration, threshold selection
against a false-positive-per-hour budget, and on-device latency profiling.

See [SOLUTION.md](SOLUTION.md) for the detailed reasoning and improvement roadmap.
