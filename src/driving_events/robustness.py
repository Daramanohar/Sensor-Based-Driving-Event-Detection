"""Deterministic perturbations for detector robustness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_regular_gps_gaps(
    frame: pd.DataFrame,
    sample_rate_hz: float,
    fix_interval_s: float,
) -> pd.DataFrame:
    """Hold GPS timestamp and speed between synthetic fixes at a chosen cadence."""

    if fix_interval_s <= 0:
        raise ValueError("fix_interval_s must be positive")
    result = frame.copy()
    block_size = max(1, int(round(fix_interval_s * sample_rate_hz)))
    source_positions = (np.arange(len(result)) // block_size) * block_size
    result["gps_timestamp"] = result["gps_timestamp"].to_numpy()[source_positions]
    result["gps_speed_mps"] = result["gps_speed_mps"].to_numpy()[source_positions]
    result["gps_is_fresh"] = result["gps_timestamp"].ne(result["gps_timestamp"].shift())
    fresh_positions = np.where(result["gps_is_fresh"].to_numpy(), result["row_id"], -1)
    last_fresh = np.maximum.accumulate(fresh_positions)
    result["gps_age_s"] = (result["row_id"].to_numpy() - last_fresh) / sample_rate_hz
    return result
