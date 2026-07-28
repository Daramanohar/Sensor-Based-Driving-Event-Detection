# Technical Solution

**Author: Dara Manohar**

## 1. Problem formulation

This is a multivariate time-series event-detection problem, not ordinary independent-row
classification. The output must identify the event class and its start/end time while
suppressing repeated detections from one physical maneuver.

Phone-axis interpretation for the supplied fixed portrait mount:

- `accel_y`: vertical axis containing the gravity baseline; useful for bumps;
- `accel_z`: longitudinal axis; useful for acceleration and braking;
- `accel_x`: lateral axis; useful for separating turns from longitudinal events;
- gyroscope axes: pitch, yaw, and roll dynamics that add shape and vibration context.

GPS is asynchronous and can be repeated or missing. It is treated as optional context through
speed, speed change, age, and freshness, not as a required gate.

## 2. Data quality and labels

Both CSV files are parsed through one strict loader that:

- validates required columns and numeric values;
- creates elapsed time from the configured sampling rate;
- marks physical-limit violations without silently deleting them;
- tracks GPS fix changes and the age of the latest fix;
- attaches a session identifier for grouped evaluation.

`data/sample_labels.csv` covers all 258 rows of the 25.7-second labeled excerpt exactly once.
The 30-minute drive has no reference labels, so its output is reported only as detected event
episodes.

## 3. Causal preprocessing

The implementation is suitable for replay or online use:

- second-order low-pass filtering suppresses high-frequency sensor noise;
- a rolling gravity/baseline estimate separates orientation drift from dynamic acceleration;
- all rolling features use past and current samples only;
- invalid values are flagged and safely imputed within the processing path;
- calculations are expressed in seconds and converted by sample rate, allowing the same logic
  at 10 Hz and 50 Hz.

For longitudinal acceleration \(a_z\), discrete jerk is:

\[
j[n] = \frac{a_z[n]-a_z[n-1]}{\Delta t}.
\]

For a window containing \(N\) gyroscope values:

\[
RMS[n] = \sqrt{\frac{1}{N}\sum_{k=0}^{N-1}g[k]^2}.
\]

Jerk captures abrupt onset while RMS captures sustained rotational or vibration energy.

## 4. Feature engineering

The model uses 125 selected inputs from causal 0.25, 0.5, 1, and 2-second windows:

- mean, standard deviation, minimum, maximum, range, RMS, and robust quantiles;
- longitudinal/vertical jerk statistics;
- acceleration and gyroscope vector magnitude;
- vibration and spectral-energy ratios where the window is long enough;
- threshold-duration fractions;
- GPS speed, delta, age, and freshness;
- signal-validity indicators.

Multi-scale features separate short impulses such as potholes from longer acceleration/braking
maneuvers. Feature selection explicitly excludes labels, timestamps, event identifiers, and
other metadata.

## 5. Streaming detector

Interpretable scores provide a strong solution when labels are scarce:

- Harsh Acceleration and Harsh Braking use signed longitudinal dynamics and duration;
- Pothole/Bump uses vertical deviation plus rotational/vibration evidence;
- Clutch Release uses short rotational energy, limited longitudinal motion, and recent low-speed
  evidence when GPS is available.

The state machine moves through idle, onset confirmation, active event, and offset confirmation
states. It applies:

- class-specific onset and offset thresholds;
- minimum and optional maximum duration;
- hysteresis to prevent rapid toggling;
- temporal merging to produce one row per physical event;
- confidence, severity, and detection-latency metadata.

This design is directly deployable as sample-by-sample logic and avoids a common error where
every threshold-crossing row is counted as a separate event.

## 6. Model selection and training

XGBoost is the strongest learned approach for the current data because it:

- models nonlinear sensor interactions;
- performs well on tabular rolling-window features;
- accepts class-balanced sample weights;
- provides feature importance;
- trains quickly on CPU and has practical mobile/server deployment options.

It is compared under the same feature data and folds with:

- Logistic Regression as a transparent linear baseline;
- Random Forest as a bagged-tree baseline.

A deep sequence model was deliberately not selected. With only one 25.7-second labeled trip and
two physical examples per target event class, an LSTM or 1-D CNN would add variance and
overfitting risk without credible validation.

## 7. Leakage-aware evaluation

Overlapping windows from the same event are highly correlated. Random row splitting would put
near-duplicate windows in train and validation and inflate performance. The implemented
two-fold event-group split keeps all windows belonging to one annotated physical interval in
exactly one fold.

Primary reporting includes:

- macro precision, recall, and F1 for imbalanced multiclass comparison;
- class-level metrics;
- event-level precision, recall, F1, and matching tolerance;
- duplicate predictions as false positives;
- false positives per hour;
- GPS-gap robustness.

The current grouped-CV result is:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| XGBoost | 0.9186 | 0.7596 |
| Logistic Regression | 0.8372 | 0.6523 |
| Random Forest | 0.8915 | 0.6293 |

XGBoost is selected because macro F1, not accuracy, is the main comparison criterion. These
numbers estimate separability inside the labeled excerpt; they do not establish performance on
new drivers, devices, or vehicles.

## 8. Reliability and reproducibility

- one YAML file controls sampling rates, thresholds, windows, and model parameters;
- one command rebuilds every published result;
- the random seed is fixed;
- model comparisons use identical folds and features;
- the package is installable and the notebook imports production code instead of duplicating it;
- tests cover schema/coverage, causal features, event matching, state-machine behavior, and
  streaming replay;
- CI runs linting, tests, and the reproducible evidence pipeline.

Colab is optional, not required. CPU execution is more appropriate for this dataset and keeps
the workflow easy for reviewers to reproduce locally.

## 9. Strongest path to higher real-world performance

The limiting factor is labeled diversity, not XGBoost compute. Improvements should be made in
this order:

1. **Build representative event-level ground truth.** Label complete trips across drivers,
   phones, mounting angles, vehicles, road surfaces, traffic, weather, and day/night conditions.
   Include hard negatives such as turns, speed breakers, phone handling, and rough roads.
2. **Use outer grouped evaluation.** Hold out complete drivers or trips for the final test.
   Perform feature/model selection only inside training trips using nested grouped validation.
3. **Normalize orientation robustly.** Estimate gravity and vehicle coordinates, detect mount
   changes, and add calibration quality flags. Train with realistic orientation augmentation.
4. **Tune XGBoost inside validation only.** Search depth, child weight, learning rate, estimator
   count, subsampling, column sampling, and regularization using macro F1 plus a false-positive
   budget. Use early stopping on grouped validation trips.
5. **Calibrate class probabilities.** Fit per-class calibration on held-out validation data, then
   choose onset/offset thresholds for the required recall and false positives per driving hour.
6. **Mine failure cases.** Add false positives and missed events back into training with reviewed
   labels. Clutch Release deserves targeted data because it is the weakest and shortest class.
7. **Improve temporal modeling after data grows.** Compare XGBoost with a compact 1-D CNN/TCN or
   gradient-boosted event proposal/reranker. Retain the state machine for stable episode output.
8. **Measure deployment behavior.** Report onset latency, CPU time, memory, energy, model size,
   sensor dropouts, and performance under different sampling rates.

This roadmap improves the credibility of performance, not just the apparent validation score.
