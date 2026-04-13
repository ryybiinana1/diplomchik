# churnlib/report_module.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .economy_module import Scenario, expected_value, profit_curve, best_k


@dataclass(frozen=True)
class ReportConfig:
    out_dir: str
    top_k: int = 500
    value_col: str = "value_proxy"


def write_reports(df_test: pd.DataFrame, p: np.ndarray, scenarios: List[Scenario], cfg: ReportConfig) -> Dict[str, Any]:
    out = Path(cfg.out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = df_test.copy()
    df["p"] = np.asarray(p, dtype=float)
    V = df[cfg.value_col].astype(float).values

    base = next((s for s in scenarios if s.name == "base"), scenarios[0])
    df["EV_base"] = expected_value(df["p"].values, V, base)

    prio = df.sort_values("EV_base", ascending=False).head(cfg.top_k)
    prio_path = out / "tables" / "priority_list_topk.csv"
    prio.to_csv(prio_path, index=False)

    plt.figure(figsize=(8, 5))
    summary_rows = []
    base_best_k = None
    base_max_profit = None

    for sc in scenarios:
        curve = profit_curve(df["p"].values, V, sc)
        k, mx = best_k(curve)
        summary_rows.append({"scenario": sc.name, "best_k": k, "max_profit": mx})
        plt.plot(curve, label=f"{sc.name} (best_k={k})")
        if sc.name == "base":
            base_best_k = k
            base_max_profit = mx

    plt.title("Profit curve (cumulative EV for top-k)")
    plt.xlabel("k")
    plt.ylabel("Expected profit")
    plt.legend()
    plot_path = out / "plots" / "profit_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    summary_path = out / "tables" / "profit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "priority_csv": str(prio_path),
        "profit_plot": str(plot_path),
        "profit_summary": str(summary_path),
        "base_best_k": int(base_best_k) if base_best_k is not None else None,
        "base_max_profit": float(base_max_profit) if base_max_profit is not None else None,
    }


def _add_picture_if_exists(doc, path: Optional[str], width_inch: float = 6.0) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        from docx.shared import Inches
        doc.add_picture(str(p), width=Inches(width_inch))
    except Exception:
        return


def write_docx_report(
    out_path: str,
    template: str,
    params_used: Dict[str, Any],
    suitability: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_paths: Dict[str, Optional[str]],
    extra_feature_audit: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Полноценный .docx отчёт. Требует установленный python-docx. citeturn25search3
    """
    from docx import Document

    out_path = str(out_path)
    doc = Document()

    doc.add_heading("Отчёт по модели churn/retention", level=1)

    doc.add_heading("Что было сделано", level=2)
    doc.add_paragraph(
        f"Мы обучили модель прогнозировать риск оттока на основе исторических данных. "
        f"Шаблон данных: {template}."
    )

    doc.add_heading("Параметры обучения", level=2)
    for k in ["horizon_days", "history_days", "step_days", "model_kind", "calibration"]:
        if k in params_used:
            doc.add_paragraph(f"- {k}: {params_used.get(k)}")

    if params_used.get("extra_feature_cols"):
        doc.add_paragraph(f"- extra_feature_cols: {params_used.get('extra_feature_cols')}")

    doc.add_heading("Вердикт по данным", level=2)
    doc.add_paragraph(f"Итог: {suitability.get('verdict')}")
    for r in suitability.get("reasons", []):
        doc.add_paragraph(f"- {r}")

    doc.add_heading("Качество модели", level=2)
    # таблица метрик
    table = doc.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "Метрика"
    hdr[1].text = "Значение"

    def add_row(name: str, value: Any):
        row = table.add_row().cells
        row[0].text = name
        row[1].text = str(value)

    add_row("target_rate", metrics.get("target_rate"))
    add_row("roc_auc (test, raw)", metrics.get("test_metrics_raw", {}).get("roc_auc"))
    add_row("pr_auc (test, raw)", metrics.get("test_metrics_raw", {}).get("pr_auc"))
    add_row("brier (test, raw)", metrics.get("test_metrics_raw", {}).get("brier"))
    add_row("roc_auc (test, calibrated)", metrics.get("test_metrics_cal", {}).get("roc_auc"))
    add_row("pr_auc (test, calibrated)", metrics.get("test_metrics_cal", {}).get("pr_auc"))
    add_row("brier (test, calibrated)", metrics.get("test_metrics_cal", {}).get("brier"))
    add_row("walk_forward roc_auc (mean)", metrics.get("walk_forward_mean", {}).get("roc_auc"))
    add_row("walk_forward pr_auc (mean)", metrics.get("walk_forward_mean", {}).get("pr_auc"))
    add_row("walk_forward brier (mean)", metrics.get("walk_forward_mean", {}).get("brier"))
    add_row("base_best_k", metrics.get("business_metrics", {}).get("base_best_k"))
    add_row("base_max_profit", metrics.get("business_metrics", {}).get("base_max_profit"))

    doc.add_heading("Графики", level=2)
    doc.add_paragraph("Кривые прибыли (profit curve):")
    _add_picture_if_exists(doc, artifact_paths.get("profit_plot"))

    doc.add_paragraph("Калибровка вероятностей:")
    _add_picture_if_exists(doc, artifact_paths.get("calibration_plot"), width_inch=5.5)

    if artifact_paths.get("shap_beeswarm"):
        doc.add_paragraph("SHAP (глобальные объяснения):")
        _add_picture_if_exists(doc, artifact_paths.get("shap_beeswarm"))
        _add_picture_if_exists(doc, artifact_paths.get("shap_bar"))

    if extra_feature_audit:
        doc.add_heading("Дополнительные колонки и безопасность", level=2)
        for a in extra_feature_audit:
            doc.add_paragraph(f"- {a.get('column')}: {a.get('verdict')} — {a.get('reason')}")

    doc.add_heading("Ограничения и рекомендации", level=2)
    doc.add_paragraph(
        "Рекомендуется сначала использовать прогноз как пилот: "
        "проверить, насколько кампании удержания действительно улучшают метрики бизнеса. "
        "Если данных мало или период короткий — сначала улучшить сбор данных."
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return {"docx_path": out_path}
