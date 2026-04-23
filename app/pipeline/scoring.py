from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.pipeline.bundle import compare_input_to_schema, load_model_bundle
from app.pipeline.features import prepare_feature_matrix
from churnlib.data_module import SnapshotConfig, build_latest_snapshot, canonicalize_types
from churnlib.economy_module import DEFAULT_SCENARIOS
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


def run_scoring_pipeline(
    input_csv: str,
    bundle_dir: str,
    out_dir: str,
    score_mapping: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(bundle_dir)
    mapping = bundle["mapping"]
    template = bundle["config"]["template"]
    params = bundle["config"]["params"]
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
        reject_extra_columns=True,
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

    base = next(s for s in DEFAULT_SCENARIOS if s.name == "base")
    V = scored["value_proxy"].astype(float).values
    scored["EV"] = V * scored["p_calibrated"].values * float(base.gain_if_save)

    scored = _attach_rule_based_reasons(scored)

    out_csv = out / "scored_clients.csv"
    scored.to_csv(out_csv, index=False)

    return {
        "template": template,
        "bundle_dir": bundle_dir,
        "output_csv": str(out_csv),
        "n_scored": int(len(scored)),
        "schema_check": schema_check,
        "score_mapping": score_mapping,
    }
