from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd


def write_model_card(
    template: str,
    horizon_days: int,
    history_days: int,
    metrics_raw: Dict,
    metrics_cal: Dict,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)

    text = f"""# Model Card

## Overview
- Task: churn / retention scoring
- Template: `{template}`
- Horizon H: `{horizon_days}` days
- History window W: `{history_days}` days

## Intended use
This model is intended for prioritizing customers/accounts/users for retention analysis or campaigns.

## Training/evaluation setup
- Time-aware split
- Walk-forward CV
- Probability calibration enabled

## Metrics (raw model)
- ROC-AUC: {metrics_raw.get("roc_auc")}
- PR-AUC: {metrics_raw.get("pr_auc")}
- Brier: {metrics_raw.get("brier")}

## Metrics (calibrated probabilities)
- ROC-AUC: {metrics_cal.get("roc_auc")}
- PR-AUC: {metrics_cal.get("pr_auc")}
- Brier: {metrics_cal.get("brier")}

## Limitations
- This is not uplift modeling.
- Probability quality depends on calibration and data stability.
- Distribution shift / concept drift may reduce quality over time.
- Recommended to retrain periodically and monitor drift.

## Ethical / business notes
- Scores should support decision-making, not replace domain judgment.
- Avoid using direct PII as model features unless governance explicitly allows it.
"""
    out_path.write_text(text, encoding="utf-8")


def write_datasheet(
    df_raw: pd.DataFrame,
    template: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)

    lines = [
        "# Datasheet",
        "",
        f"## Dataset template",
        f"- Template: `{template}`",
        f"- Rows: `{len(df_raw)}`",
        f"- Columns: `{len(df_raw.columns)}`",
        "",
        "## Columns",
    ]

    for col in df_raw.columns:
        dtype = str(df_raw[col].dtype)
        nulls = int(df_raw[col].isna().sum())
        lines.append(f"- `{col}`: dtype={dtype}, nulls={nulls}")

    lines += [
        "",
        "## Notes",
        "- This datasheet is auto-generated from the uploaded raw file.",
        "- Review missing values, business definitions, and collection process before production use.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")