# churnlib/validation_module.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def basic_validate(df: pd.DataFrame, template: str) -> None:
    if template == "transactions":
        required = ["customer_id", "transaction_id", "event_time", "amount"]
    elif template == "subscriptions":
        required = ["account_id", "period_start", "period_end", "mrr", "subscription_status"]
    elif template == "events":
        required = ["subject_id", "event_time", "event_name"]
    else:
        raise ValueError(f"Unknown template: {template}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # базовая проверка на nulls в required
    null_req = [c for c in required if df[c].isna().mean() > 0.5]
    if null_req:
        raise ValueError(f"Too many nulls in required columns: {null_req}")


def profile_dataset(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    df = df.copy()

    out: Dict[str, Any] = {
        "template": template,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_share_overall": float(df.isna().mean().mean()) if df.shape[1] else 0.0,
        "null_share_by_column": {c: float(df[c].isna().mean()) for c in df.columns},
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }

    if template == "transactions" and {"event_time", "customer_id", "transaction_id", "amount"}.issubset(df.columns):
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        df["customer_id"] = df["customer_id"].astype(str)
        df["transaction_id"] = df["transaction_id"].astype(str)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        tmin = df["event_time"].min()
        tmax = df["event_time"].max()
        span_days = (tmax - tmin).days if pd.notna(tmin) and pd.notna(tmax) else None

        out.update(
            {
                "time_min": str(tmin) if pd.notna(tmin) else None,
                "time_max": str(tmax) if pd.notna(tmax) else None,
                "time_span_days": int(span_days) if span_days is not None else None,
                "n_customers": int(df["customer_id"].nunique()),
                "n_transactions": int(df["transaction_id"].nunique()),
            }
        )

        per_cust = df.groupby("customer_id")["transaction_id"].count()
        out["tx_per_customer"] = {
            "mean": float(per_cust.mean()) if len(per_cust) else None,
            "median": float(per_cust.median()) if len(per_cust) else None,
            "p90": float(per_cust.quantile(0.9)) if len(per_cust) else None,
            "share_customers_1_tx": float((per_cust <= 1).mean()) if len(per_cust) else None,
            "share_customers_2plus_tx": float((per_cust >= 2).mean()) if len(per_cust) else None,
        }

    if template == "subscriptions" and {"account_id", "period_end", "mrr", "subscription_status"}.issubset(df.columns):
        df["account_id"] = df["account_id"].astype(str)
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce")

        tmin = df["period_end"].min()
        tmax = df["period_end"].max()
        span_days = (tmax - tmin).days if pd.notna(tmin) and pd.notna(tmax) else None

        out.update(
            {
                "time_min": str(tmin) if pd.notna(tmin) else None,
                "time_max": str(tmax) if pd.notna(tmax) else None,
                "time_span_days": int(span_days) if span_days is not None else None,
                "n_accounts": int(df["account_id"].nunique()),
                "status_counts": df["subscription_status"].value_counts(dropna=False).to_dict(),
            }
        )

    if template == "events" and {"subject_id", "event_time", "event_name"}.issubset(df.columns):
        df["subject_id"] = df["subject_id"].astype(str)
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

        tmin = df["event_time"].min()
        tmax = df["event_time"].max()
        span_days = (tmax - tmin).days if pd.notna(tmin) and pd.notna(tmax) else None

        out.update(
            {
                "time_min": str(tmin) if pd.notna(tmin) else None,
                "time_max": str(tmax) if pd.notna(tmax) else None,
                "time_span_days": int(span_days) if span_days is not None else None,
                "n_subjects": int(df["subject_id"].nunique()),
                "n_events": int(len(df)),
            }
        )

    return out


def assess_suitability(
    df: pd.DataFrame,
    template: str,
    params: Optional[Dict[str, Any]] = None,
    horizons: List[int] | None = None,
) -> Dict[str, Any]:
    """
    Вердикт пригодности данных для churn‑обучения.
    Это НЕ “истина”, а продуктовая диагностика с чёткими причинами.
    """
    params = params or {}
    horizons = horizons or [30, 60, 90]

    history_days = int(params.get("history_days", 180))
    step_days = int(params.get("step_days", 30))

    verdict = "ready"
    reasons: List[str] = []
    horizon_checks: Dict[str, Any] = {}

    if template == "transactions":
        if "event_time" not in df.columns or "customer_id" not in df.columns:
            return {"verdict": "not_recommended", "reasons": ["Не хватает обязательных колонок для транзакционного шаблона."], "horizons": {}}

        d = df.copy()
        d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce", utc=True)
        d["customer_id"] = d["customer_id"].astype(str)

        tmin = d["event_time"].min()
        tmax = d["event_time"].max()
        if pd.isna(tmin) or pd.isna(tmax):
            return {"verdict": "not_recommended", "reasons": ["Не удалось распознать даты операций (event_time)."], "horizons": {}}

        span_days = int((tmax - tmin).days)
        n_customers = int(d["customer_id"].nunique())

        per_cust = d.groupby("customer_id").size()
        share_2plus = float((per_cust >= 2).mean()) if len(per_cust) else 0.0

        if n_customers < 100:
            verdict = "partial"
            reasons.append(f"Слишком мало клиентов для устойчивого обучения: {n_customers} (желательно ≥ 100).")

        if span_days < (history_days + min(horizons)):
            verdict = "partial"
            reasons.append(
                f"Недостаточно истории: всего {span_days} дней, а для базового окна {history_days} + горизонта ≥{min(horizons)} нужно больше."
            )

        if share_2plus < 0.15:
            verdict = "partial"
            reasons.append(
                f"Слишком мало повторных покупок: доля клиентов с ≥2 покупками = {share_2plus:.1%}. Модели сложнее ловить паттерн оттока."
            )

        for h in horizons:
            feasible = span_days >= (history_days + h)
            # "labelable" — у клиента есть активность не позже, чем tmax-h (иначе “будущего окна” нет)
            cutoff = (tmax - pd.Timedelta(days=h))
            labelable = d.loc[d["event_time"] <= cutoff, "customer_id"].nunique()
            labelable_share = float(labelable / max(n_customers, 1))
            horizon_checks[str(h)] = {
                "feasible_by_span": bool(feasible),
                "labelable_customers_share": labelable_share,
                "cutoff_date": str(cutoff),
            }

        if verdict == "partial" and any(not v["feasible_by_span"] for v in horizon_checks.values()):
            reasons.append("Часть горизонтов недоступна из-за короткого периода данных.")

        if not reasons:
            reasons.append("Данные выглядят пригодными для обучения при выбранных окнах/горизонтах.")

        return {
            "verdict": verdict,
            "reasons": reasons,
            "metrics": {
                "time_span_days": span_days,
                "n_customers": n_customers,
                "share_customers_2plus_tx": share_2plus,
                "history_days": history_days,
                "step_days": step_days,
            },
            "horizons": horizon_checks,
        }

    # Для subscriptions и events делаем более мягкий вердикт (пока проще)
    d = df.copy()
    if template == "subscriptions":
        key = "account_id"
        time_col = "period_end"
    elif template == "events":
        key = "subject_id"
        time_col = "event_time"
    else:
        return {"verdict": "not_recommended", "reasons": ["Неизвестный шаблон данных."], "horizons": {}}

    if key not in d.columns or time_col not in d.columns:
        return {"verdict": "not_recommended", "reasons": ["Не хватает обязательных колонок для выбранного шаблона."], "horizons": {}}

    d[key] = d[key].astype(str)
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)

    tmin = d[time_col].min()
    tmax = d[time_col].max()
    if pd.isna(tmin) or pd.isna(tmax):
        return {"verdict": "not_recommended", "reasons": ["Проблема с распознаванием дат."], "horizons": {}}

    span_days = int((tmax - tmin).days)
    n_entities = int(d[key].nunique())

    if n_entities < 100:
        verdict = "partial"
        reasons.append(f"Сущностей слишком мало: {n_entities}.")
    if span_days < (history_days + min(horizons)):
        verdict = "partial"
        reasons.append(f"Период данных {span_days} дней — вероятно маловат для стабильной модели.")

    if not reasons:
        reasons.append("Данные выглядят пригодными для обучения (базовая проверка).")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {"time_span_days": span_days, "n_entities": n_entities, "history_days": history_days, "step_days": step_days},
        "horizons": {},
    }
