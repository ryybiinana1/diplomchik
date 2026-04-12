from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def _psi_for_series(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return np.nan

    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)

    e_perc = np.where(e_counts == 0, 1e-6, e_counts / e_counts.sum())
    a_perc = np.where(a_counts == 0, 1e-6, a_counts / a_counts.sum())

    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi)


def compute_feature_psi(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        try:
            psi = _psi_for_series(train_df[col], test_df[col])
        except Exception:
            psi = np.nan
        rows.append({"feature": col, "psi": psi})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)