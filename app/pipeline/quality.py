from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def build_quality_payload(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, p)
    pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(y_true, p)

    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred).tolist()

    return {
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        },
        "pr_curve": {
            "precision": [float(x) for x in pr_curve_precision],
            "recall": [float(x) for x in pr_curve_recall],
        },
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "target_rate": float(np.mean(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def choose_metric(result: Dict[str, Any], metric_name: str) -> float:
    """
    Для brier — меньше лучше, поэтому возвращаем -brier.
    Для остальных — больше лучше.
    """
    if metric_name == "base_max_profit":
        return float(result.get("business_metrics", {}).get("base_max_profit", -1e18))

    tm = result.get("test_metrics_cal", {})
    if metric_name in tm:
        v = float(tm[metric_name])
        return -v if metric_name == "brier" else v

    cv = result.get("cv_metrics_mean", {})
    if metric_name in cv:
        v = float(cv[metric_name])
        return -v if metric_name == "brier" else v

    wf = result.get("walk_forward", {}).get("mean_metrics", {})
    if metric_name in wf:
        v = float(wf[metric_name])
        return -v if metric_name == "brier" else v

    raise ValueError(f"Unknown metric for selection: {metric_name}")
