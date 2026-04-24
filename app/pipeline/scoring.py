from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.pipeline.bundle import compare_input_to_schema, load_model_bundle
from app.pipeline.features import prepare_feature_matrix
from churnlib.data_module import SnapshotConfig, build_latest_snapshot, canonicalize_types
from churnlib.economy_module import build_scenarios, expected_value, profit_curve_from_ev, scenario_from_params, best_k
from churnlib.validation_module import basic_validate


def _attach_rule_based_reasons(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Простые объяснения "на человеческом языке" для клиента.
    Мы не делаем SHAP на фронте: только доп. колонки reason_1..reason_3.
    """

    def pick_reasons(row: pd.Series) -> List[str]:
        reasons: List[str] = []
        if "recency_days" in row and pd.notna(row["recency_days"]) and float(row["recency_days"]) > 60:
            reasons.append("Давно не было активности/покупок")
        if "frequency_tx" in row and pd.notna(row["frequency_tx"]) and float(row["frequency_tx"]) <= 1:
            reasons.append("Редкая активность (мало покупок в истории)")
        if "monetary" in row and pd.notna(row["monetary"]) and float(row["monetary"]) <= 0:
            reasons.append("Низкая выручка/сумма покупок")
        if not reasons:
            reasons.append("Риск повышен по совокупности поведения")
        return reasons[:3]

    rr = scored.apply(pick_reasons, axis=1)
    scored = scored.copy()
    scored["reason_1"] = rr.apply(lambda x: x[0] if len(x) > 0 else "")
    scored["reason_2"] = rr.apply(lambda x: x[1] if len(x) > 1 else "")
    scored["reason_3"] = rr.apply(lambda x: x[2] if len(x) > 2 else "")
    return scored


def _value_proxy_meta(template: str) -> Dict[str, Any]:
    if template == "transactions":
        return {"label": "Денежная ценность клиента", "unit": "money", "is_monetary": True}
    if template == "subscriptions":
        return {"label": "MRR / платеж клиента", "unit": "money", "is_monetary": True}
    return {"label": "Условная ценность активности", "unit": "relative", "is_monetary": False}


def _scenario_summary_rows(scored: pd.DataFrame, scenarios: List[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        ev_col = f"EV_{sc.name}"
        ev_series = pd.to_numeric(scored.get(ev_col), errors="coerce").fillna(0.0)
        sorted_ev = ev_series.sort_values(ascending=False, kind="mergesort")
        positive_ev = ev_series[ev_series > 0]
        curve = profit_curve_from_ev(ev_series.to_numpy(dtype=float))
        best_k_value, max_profit = best_k(curve) if curve.size else (0, 0.0)
        rows.append(
            {
                "scenario": sc.name,
                "margin": float(sc.margin),
                "cost": float(sc.cost),
                "success": float(sc.success),
                "clients_with_positive_ev": int((ev_series > 0).sum()),
                "best_k": int(best_k_value),
                "max_profit": float(max_profit),
                "total_positive_ev": float(positive_ev.sum()) if not positive_ev.empty else 0.0,
                "mean_ev_top20": float(sorted_ev.head(min(20, len(sorted_ev))).mean()) if not sorted_ev.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _scenario_curve_rows(scored: pd.DataFrame, scenarios: List[Any]) -> List[Dict[str, float]]:
    curves: Dict[str, np.ndarray] = {}
    max_len = 0
    for sc in scenarios:
        curve = profit_curve_from_ev(pd.to_numeric(scored.get(f"EV_{sc.name}"), errors="coerce").fillna(0.0).to_numpy(dtype=float))
        curves[sc.name] = curve
        max_len = max(max_len, int(curve.size))
    rows: List[Dict[str, float]] = []
    for idx in range(max_len):
        row: Dict[str, float] = {"top_k": int(idx + 1)}
        for name, curve in curves.items():
            row[name] = float(curve[idx]) if idx < curve.size else float(curve[-1]) if curve.size else 0.0
        rows.append(row)
    return rows


def _priority_labels_by_ev(ev_series: pd.Series, best_k_value: int) -> pd.Series:
    if ev_series.empty:
        return pd.Series(dtype=str)
    labels = pd.Series("Низкий", index=ev_series.index, dtype="object")
    positive_mask = pd.to_numeric(ev_series, errors="coerce").fillna(0.0) > 0
    labels.loc[positive_mask] = "Средний"
    if best_k_value > 0:
        high_index = ev_series.index[: min(best_k_value, len(ev_series))]
        labels.loc[high_index] = labels.loc[high_index].where(~positive_mask.loc[high_index], "Высокий")
    return labels


def _priority_sort_key(labels: pd.Series) -> pd.Series:
    order = {"Высокий": 0, "Средний": 1, "Низкий": 2}
    return labels.map(order).fillna(3)


def _scenario_priority_frame(scored: pd.DataFrame, scenario_name: str, best_k_value: int) -> pd.DataFrame:
    ev_col = f"EV_{scenario_name}"
    scenario_df = scored.copy()
    scenario_df = scenario_df.sort_values([ev_col, "p_calibrated"], ascending=[False, False], na_position="last").reset_index(
        drop=True
    )
    scenario_df["priority"] = _priority_labels_by_ev(pd.to_numeric(scenario_df.get(ev_col), errors="coerce"), best_k_value)
    scenario_df["recommended_action"] = np.where(
        scenario_df["priority"].eq("Высокий"),
        "Включить в кампанию удержания",
        np.where(scenario_df["priority"].eq("Средний"), "Рассмотреть при наличии ресурса", "Не приоритизировать"),
    )
    scenario_df["priority_scenario"] = str(scenario_name)
    scenario_df["scenario_ev"] = pd.to_numeric(scenario_df.get(ev_col), errors="coerce")
    scenario_df["_priority_order"] = _priority_sort_key(scenario_df["priority"])
    scenario_df = scenario_df.sort_values(
        ["_priority_order", ev_col, "p_calibrated"], ascending=[True, False, False], na_position="last"
    ).drop(columns=["_priority_order"]).reset_index(drop=True)
    return scenario_df


def run_scoring_pipeline(
    input_csv: str,
    bundle_dir: str,
    out_dir: str,
    score_mapping: Dict[str, Any] | None = None,
    scenario_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(bundle_dir)
    mapping = bundle["mapping"]
    template = bundle["config"]["template"]
    params = bundle["config"]["params"]
    scoring_params = {**params, **(scenario_params or {})}
    feature_cols = bundle["feature_cols"]
    training_schema = bundle.get("training_schema") or {}
    score_mapping = {str(k): str(v) for k, v in (score_mapping or {}).items() if k and v}

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    schema_check = compare_input_to_schema(
        input_columns=list(df_raw.columns),
        training_schema={
            **training_schema,
            "input_dtypes": {col: str(df_raw[col].dtype) for col in df_raw.columns},
        },
        mapping=mapping,
        score_mapping=score_mapping,
        require_full_mapping=True,
        reject_extra_columns=False,
    )
    if not schema_check.get("ok"):
        details = []
        if schema_check.get("missing_mapping"):
            details.append(f"не сопоставлены колонки ({', '.join(schema_check['missing_mapping'])})")
        if schema_check.get("missing_required"):
            details.append(f"отсутствуют обязательные колонки ({', '.join(schema_check['missing_required'])})")
        if schema_check.get("extra_columns"):
            details.append(f"есть лишние колонки ({', '.join(schema_check['extra_columns'])})")
        if schema_check.get("invalid_mapping_keys"):
            details.append(f"есть недопустимые ключи сопоставления ({', '.join(schema_check['invalid_mapping_keys'])})")
        detail_text = "; ".join(details) if details else "схема не совпадает с обучающим набором"
        raise ValueError(f"Файл не подходит для этой модели: {detail_text}.")
    effective_mapping = {
        canon: score_mapping.get(src, src)
        for canon, src in mapping.items()
        if src
    }
    df = df_raw.rename(columns={v: k for k, v in effective_mapping.items() if v}).copy()

    if template == "transactions" and "amount" not in df.columns:
        if "unit_price" in df.columns and "quantity" in df.columns:
            up = pd.to_numeric(df["unit_price"], errors="coerce")
            q = pd.to_numeric(df["quantity"], errors="coerce")
            df["amount"] = up * q

    basic_validate(df, template=template)
    df = canonicalize_types(df, template)

    snap_cfg = SnapshotConfig(
        template=str(template),
        horizon_days=int(params.get("horizon_days", 30)),
        history_days=int(params.get("history_days", 180)),
        step_days=int(params.get("step_days", 30)),
        min_events_in_history=int(params.get("min_events_in_history", 1)),
        min_lifetime_days=int(params.get("min_lifetime_days", 0)),
        max_recency_days=params.get("max_recency_days", None),
        tz=params.get("tz", None),
        extra_feature_cols=params.get("extra_feature_cols", []),
        extra_feature_config=params.get("extra_feature_config", {}),
    )

    snaps = build_latest_snapshot(df, snap_cfg)
    if snaps.empty:
        raise ValueError("No snapshot rows for scoring")

    X = prepare_feature_matrix(snaps, feature_cols)
    model = bundle["model"]
    calibrator = bundle["calibrator"]

    p_raw = model.predict_proba(X)[:, 1]
    p_cal = calibrator.predict_proba(X)[:, 1]

    scored = snaps.copy()
    scored["p_raw"] = p_raw
    scored["p_calibrated"] = p_cal

    try:
        scored["risk_segment"] = pd.qcut(
            scored["p_calibrated"], q=4, labels=["low", "medium", "high", "critical"], duplicates="drop"
        ).astype(str)
    except Exception:
        scored["risk_segment"] = "unknown"

    V = scored["value_proxy"].astype(float).values
    scenarios = build_scenarios(scoring_params)
    for sc in scenarios:
        scored[f"EV_{sc.name}"] = expected_value(scored["p_calibrated"].values, V, sc)
    scored["EV"] = scored["EV_base"]
    scored = scored.sort_values(["EV_base", "p_calibrated"], ascending=[False, False], na_position="last").reset_index(drop=True)
    summary_df = _scenario_summary_rows(scored, scenarios)
    scored = _attach_rule_based_reasons(scored)
    scenario_best_k = {
        str(row["scenario"]): int(row["best_k"])
        for row in summary_df.to_dict(orient="records")
        if row.get("scenario") is not None
    }
    base_row = summary_df.loc[summary_df["scenario"] == "base"].head(1)
    best_k_base = int(base_row.iloc[0]["best_k"]) if not base_row.empty else 0
    scored["priority"] = _priority_labels_by_ev(pd.to_numeric(scored.get("EV_base"), errors="coerce"), best_k_base)
    scored["recommended_action"] = np.where(
        scored["priority"].eq("Высокий"),
        "Включить в кампанию удержания",
        np.where(scored["priority"].eq("Средний"), "Рассмотреть при наличии ресурса", "Не приоритизировать"),
    )

    ev_series = pd.to_numeric(scored["EV_base"], errors="coerce")
    positive_ev = ev_series[ev_series > 0]
    scenario_rows = summary_df.to_dict(orient="records")
    value_meta = _value_proxy_meta(template)
    business_summary = {
        "scenario": "base",
        "scenario_params": (
            {
                "margin": float(base_row.iloc[0]["margin"]),
                "cost": float(base_row.iloc[0]["cost"]),
                "success": float(base_row.iloc[0]["success"]),
            }
            if not base_row.empty
            else {}
        ),
        "clients_with_positive_ev": int((ev_series > 0).sum()) if not ev_series.empty else 0,
        "total_positive_ev": float(positive_ev.sum()) if not positive_ev.empty else 0.0,
        "max_ev": float(ev_series.max()) if not ev_series.empty else None,
        "mean_ev_top20": float(ev_series.head(min(20, len(ev_series))).mean()) if not ev_series.empty else None,
        "best_k": best_k_base,
        "scenario_rows": scenario_rows,
        "scenario_curve_rows": _scenario_curve_rows(scored, scenarios),
        "value_proxy_label": value_meta["label"],
        "value_unit": value_meta["unit"],
        "is_monetary": bool(value_meta["is_monetary"]),
    }
    scenario_priority_frames = {
        sc.name: _scenario_priority_frame(scored, sc.name, scenario_best_k.get(sc.name, 0)) for sc in scenarios
    }
    priority_df = scenario_priority_frames.get("base", scored.copy())

    out_csv = out / "scored_clients.csv"
    priority_csv = out / "retention_priority_list.csv"
    summary_csv = out / "scenario_summary.csv"
    scored.to_csv(out_csv, index=False)
    priority_df.to_csv(priority_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    scenario_priority_csvs: Dict[str, str] = {}
    for scenario_name, scenario_df in scenario_priority_frames.items():
        scenario_path = out / f"retention_priority_list_{scenario_name}.csv"
        scenario_df.to_csv(scenario_path, index=False)
        scenario_priority_csvs[scenario_name] = str(scenario_path)

    return {
        "template": template,
        "bundle_dir": bundle_dir,
        "output_csv": str(out_csv),
        "priority_csv": str(priority_csv),
        "scenario_priority_csvs": scenario_priority_csvs,
        "scenario_summary_csv": str(summary_csv),
        "n_scored": int(len(scored)),
        "schema_check": schema_check,
        "score_mapping": score_mapping,
        "scenario_params_used": dict(scenario_params or {}),
        "business_summary": business_summary,
    }
