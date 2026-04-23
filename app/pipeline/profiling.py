from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def assess_training_readiness(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    verdict = "ready"
    reasons = []

    if template == "transactions":
        date_col = None
        for col in df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                date_col = col
                break

        if date_col is None:
            verdict = "partial"
            reasons.append("Не найдена колонка с датой.")

    if not reasons:
        reasons.append("Явных стоп-факторов не найдено.")

    return {
        "verdict": verdict,
        "reasons": reasons,
    }


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        s = df[col]
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            valid_share = parsed.notna().mean()
            if valid_share > 0.7:
                years = parsed.dropna().dt.year
                if not years.empty and years.between(2000, 2100).mean() > 0.8:
                    out.append(col)
            elif any(x in str(col).lower() for x in ["date", "time", "timestamp", "datetime"]):
                out.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(out))


def _detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()
        if any(
            x in name
            for x in [
                "id",
                "customer",
                "client",
                "account",
                "subject",
                "invoice",
                "order",
                "transaction",
                "no",
            ]
        ):
            out.append(col)
            continue
        try:
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if nunique_ratio > 0.95:
                out.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(out))


def _detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    id_like = set(_detect_id_like_columns(df))
    for col in df.columns:
        if col in id_like:
            continue
        try:
            s = pd.to_numeric(df[col], errors="coerce")
            valid_share = s.notna().mean()
            if valid_share >= 0.7:
                out.append(col)
        except Exception:
            pass
    return out


def profile_dataset(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    nulls = df.isna().sum().to_dict()
    null_share = (df.isna().mean().round(6)).to_dict()

    date_candidates = _detect_date_columns(df)
    id_candidates = _detect_id_like_columns(df)
    num_candidates = _detect_numeric_columns(df)

    date_ranges: Dict[str, Dict[str, str]] = {}
    for col in date_candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            date_ranges[col] = {
                "min": str(parsed.min()),
                "max": str(parsed.max()),
                "non_null_share": f"{parsed.notna().mean():.3f}",
            }

    warnings = []
    if len(df) < 1000:
        warnings.append("Мало строк: для устойчивого обучения желательно больше наблюдений.")
    if df.duplicated().sum() > 0:
        warnings.append("Есть полные дубликаты строк.")
    if max(null_share.values(), default=0) > 0.5:
        warnings.append("Есть колонки с большим количеством пропусков.")

    diagnostics = {
        "date_candidates": date_candidates,
        "id_candidates": id_candidates,
        "numeric_candidates": num_candidates,
    }

    readiness = assess_training_readiness(df, template)

    return {
        "template": template,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "duplicates_full_rows": int(df.duplicated().sum()),
        "nulls_by_column": nulls,
        "null_share_by_column": null_share,
        "date_ranges": date_ranges,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "readiness": readiness,
    }
