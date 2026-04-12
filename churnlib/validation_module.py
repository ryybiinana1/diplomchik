from __future__ import annotations

from typing import Dict, List, Any
import pandas as pd


REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "transactions": ["customer_id", "transaction_id", "event_time", "amount"],
    "subscriptions": ["account_id", "period_start", "period_end", "mrr", "subscription_status"],
    "events": ["subject_id", "event_time", "event_name"],
}


DATE_COLUMNS: Dict[str, List[str]] = {
    "transactions": ["event_time"],
    "subscriptions": ["period_start", "period_end", "churn_date"],
    "events": ["event_time"],
}


def _check_required_columns(df: pd.DataFrame, template: str) -> None:
    required = REQUIRED_COLUMNS.get(template)
    if required is None:
        raise ValueError(f"Unknown template: {template}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for template='{template}': {missing}. "
            f"Present columns: {list(df.columns)}"
        )


def _check_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")


def profile_dataset(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "template": template,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicates_full_rows": int(df.duplicated().sum()),
        "nulls_by_column": {},
        "null_share_by_column": {},
        "date_ranges": {},
        "warnings": [],
    }

    for col in df.columns:
        nulls = int(df[col].isna().sum())
        report["nulls_by_column"][col] = nulls
        report["null_share_by_column"][col] = float(nulls / max(len(df), 1))

    for col in DATE_COLUMNS.get(template, []):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            valid = parsed.dropna()
            report["date_ranges"][col] = {
                "parseable_share": float(valid.shape[0] / max(len(df), 1)),
                "min": str(valid.min()) if not valid.empty else None,
                "max": str(valid.max()) if not valid.empty else None,
                "invalid_count": int(parsed.isna().sum()),
            }

    # template-specific light checks
    if template == "transactions":
        if "customer_id" in df.columns:
            report["unique_customers"] = int(df["customer_id"].nunique(dropna=True))
        if "transaction_id" in df.columns:
            report["unique_transactions"] = int(df["transaction_id"].nunique(dropna=True))
        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce")
            report["amount_stats"] = {
                "min": float(amt.min()) if amt.notna().any() else None,
                "max": float(amt.max()) if amt.notna().any() else None,
                "non_numeric_count": int(amt.isna().sum()),
                "non_positive_count": int((amt <= 0).sum()) if amt.notna().any() else 0,
            }

    elif template == "subscriptions":
        if "account_id" in df.columns:
            report["unique_accounts"] = int(df["account_id"].nunique(dropna=True))
        if "subscription_status" in df.columns:
            report["status_counts"] = df["subscription_status"].astype(str).value_counts(dropna=False).to_dict()

    elif template == "events":
        if "subject_id" in df.columns:
            report["unique_subjects"] = int(df["subject_id"].nunique(dropna=True))
        if "event_name" in df.columns:
            report["event_name_top"] = df["event_name"].astype(str).value_counts(dropna=False).head(20).to_dict()

    # warnings
    if len(df) < 100:
        report["warnings"].append("Очень мало строк: модель может быть нестабильной.")
    if report["duplicates_full_rows"] > 0:
        report["warnings"].append("Есть полные дубликаты строк.")
    for col, share in report["null_share_by_column"].items():
        if share > 0.3:
            report["warnings"].append(f"Колонка '{col}' содержит более 30% пропусков.")

    return report


def _check_transactions(df: pd.DataFrame) -> None:
    if df["customer_id"].isna().any():
        raise ValueError("transactions: customer_id contains nulls")
    if df["transaction_id"].isna().any():
        raise ValueError("transactions: transaction_id contains nulls")
    if df["event_time"].isna().any():
        raise ValueError("transactions: event_time contains nulls")
    if df["amount"].isna().any():
        raise ValueError("transactions: amount contains nulls")

    parsed_time = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    if parsed_time.isna().any():
        bad_n = int(parsed_time.isna().sum())
        raise ValueError(f"transactions: event_time contains {bad_n} unparsable values")

    parsed_amount = pd.to_numeric(df["amount"], errors="coerce")
    if parsed_amount.isna().any():
        bad_n = int(parsed_amount.isna().sum())
        raise ValueError(f"transactions: amount contains {bad_n} non-numeric values")

    if (parsed_amount <= 0).all():
        raise ValueError("transactions: all amount values are <= 0")

    now_utc = pd.Timestamp.now(tz="UTC")
    if parsed_time.max() > now_utc + pd.Timedelta(days=1):
        raise ValueError("transactions: event_time contains future timestamps")


def _check_subscriptions(df: pd.DataFrame) -> None:
    for col in ["period_start", "period_end"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().any():
            bad_n = int(parsed.isna().sum())
            raise ValueError(f"subscriptions: {col} contains {bad_n} unparsable values")

    parsed_mrr = pd.to_numeric(df["mrr"], errors="coerce")
    if parsed_mrr.isna().any():
        bad_n = int(parsed_mrr.isna().sum())
        raise ValueError(f"subscriptions: mrr contains {bad_n} non-numeric values")
    if (parsed_mrr < 0).any():
        raise ValueError("subscriptions: mrr contains negative values")

    allowed = {"active", "canceled", "past_due"}
    bad = set(df["subscription_status"].astype(str).str.lower().unique()) - allowed
    if bad:
        raise ValueError(
            f"subscriptions: subscription_status contains invalid values: {sorted(bad)}. "
            f"Allowed: {sorted(allowed)}"
        )

    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    period_end = pd.to_datetime(df["period_end"], errors="coerce")
    if (period_end < period_start).any():
        raise ValueError("subscriptions: found rows where period_end < period_start")


def _check_events(df: pd.DataFrame) -> None:
    if df["subject_id"].isna().any():
        raise ValueError("events: subject_id contains nulls")
    if df["event_name"].isna().any():
        raise ValueError("events: event_name contains nulls")

    parsed_time = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    if parsed_time.isna().any():
        bad_n = int(parsed_time.isna().sum())
        raise ValueError(f"events: event_time contains {bad_n} unparsable values")

    now_utc = pd.Timestamp.now(tz="UTC")
    if parsed_time.max() > now_utc + pd.Timedelta(days=1):
        raise ValueError("events: event_time contains future timestamps")


def basic_validate(df: pd.DataFrame, template: str) -> None:
    _check_non_empty(df)
    _check_required_columns(df, template)

    if template == "transactions":
        _check_transactions(df)
    elif template == "subscriptions":
        _check_subscriptions(df)
    elif template == "events":
        _check_events(df)
    else:
        raise ValueError(f"Unknown template: {template}")