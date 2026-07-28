"""Annotation coverage checks."""

from __future__ import annotations

import pandas as pd


def annotation_coverage(frame: pd.DataFrame, labeled_frame: pd.DataFrame) -> dict[str, float | int]:
    uncovered = labeled_frame["label"].isna()
    return {
        "rows": int(len(frame)),
        "labeled_rows": int((~uncovered).sum()),
        "unlabeled_rows": int(uncovered.sum()),
        "coverage": float((~uncovered).mean()),
    }
