from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "isotonic"   # "sigmoid" | "isotonic"
    cv: int = 3


def prob_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def calibrate(model, X_train, y_train, X_test, y_test, cfg: CalibrationConfig) -> Dict:
    raw_p = model.predict_proba(X_test)[:, 1]
    raw_metrics = prob_metrics(y_test, raw_p)

    cal = CalibratedClassifierCV(estimator=model, method=cfg.method, cv=cfg.cv)
    cal.fit(X_train, y_train)
    p_cal = cal.predict_proba(X_test)[:, 1]
    cal_metrics = prob_metrics(y_test, p_cal)

    delta = {
        "roc_auc_delta": cal_metrics["roc_auc"] - raw_metrics["roc_auc"],
        "pr_auc_delta": cal_metrics["pr_auc"] - raw_metrics["pr_auc"],
        "brier_delta": cal_metrics["brier"] - raw_metrics["brier"],
    }

    return {
        "calibrator": cal,
        "p_raw": raw_p,
        "p_cal": p_cal,
        "raw_metrics": raw_metrics,
        "metrics": cal_metrics,
        "delta": delta,
        "method": cfg.method,
    }


def save_calibration_plot(y_true: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, path: str, n_bins: int = 10):
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, p_raw, n_bins=n_bins)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, p_cal, n_bins=n_bins)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred_raw, frac_pos_raw, marker="o", label="raw")
    plt.plot(mean_pred_cal, frac_pos_cal, marker="o", label="calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.title("Calibration curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_calibration_summary(cal_res: Dict, path: str):
    payload = {
        "method": cal_res["method"],
        "raw_metrics": cal_res["raw_metrics"],
        "calibrated_metrics": cal_res["metrics"],
        "delta": cal_res["delta"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)