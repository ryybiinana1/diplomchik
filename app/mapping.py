# app/mapping.py
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional
import re
import pandas as pd


CANONICAL_FIELDS = {
    "transactions": {
        "required": ["customer_id", "transaction_id", "event_time", "amount"],
        "optional": ["item_id", "quantity", "country", "unit_price", "is_cancellation", "promo_code"],
        "aliases": {
            "customer_id": ["customerid", "customer_id", "client_id", "user_id", "buyer_id"],
            "transaction_id": ["invoiceno", "invoice_no", "transaction_id", "order_id", "purchase_id"],
            "event_time": ["invoicedate", "invoice_date", "event_time", "datetime", "date", "timestamp", "order_date"],
            "amount": ["revenue", "amount", "total", "total_amount", "sales", "price_total"],
            "item_id": ["stockcode", "item_id", "product_id", "sku"],
            "quantity": ["quantity", "qty", "count"],
            "country": ["country", "region", "market"],
            "unit_price": ["unitprice", "unit_price", "price"],
            "is_cancellation": ["is_cancellation", "cancel_flag", "is_return", "is_refund"],
            "promo_code": ["promo_code", "coupon", "discount_code"],
        },
    },
    "subscriptions": {
        "required": ["account_id", "period_start", "period_end", "mrr", "subscription_status"],
        "optional": ["churn_date", "plan_name", "seats_purchased", "seats_used"],
        "aliases": {
            "account_id": ["account_id", "customer_id", "client_id", "org_id"],
            "period_start": ["period_start", "start_date", "subscription_start", "billing_start"],
            "period_end": ["period_end", "end_date", "subscription_end", "billing_end"],
            "mrr": ["mrr", "revenue", "monthly_revenue", "amount"],
            "subscription_status": ["subscription_status", "status", "plan_status"],
            "churn_date": ["churn_date", "cancel_date", "termination_date"],
            "plan_name": ["plan_name", "plan", "tariff"],
            "seats_purchased": ["seats_purchased", "licensed_seats", "bought_seats"],
            "seats_used": ["seats_used", "active_seats", "used_seats"],
        },
    },
    "events": {
        "required": ["subject_id", "event_time", "event_name"],
        "optional": ["account_id", "platform"],
        "aliases": {
            "subject_id": ["subject_id", "user_id", "customer_id", "client_id", "member_id"],
            "event_time": ["event_time", "timestamp", "datetime", "event_date", "date"],
            "event_name": ["event_name", "action", "event", "activity_type"],
            "account_id": ["account_id", "org_id", "company_id"],
            "platform": ["platform", "device", "source"],
        },
    },
}


def _normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def detect_mapping(columns: List[str], template: str) -> Dict[str, Optional[str]]:
    """
    Возвращает suggested mapping:
    canonical_col -> source_col | None
    """
    spec = CANONICAL_FIELDS[template]
    aliases = spec["aliases"]
    suggestions: Dict[str, Optional[str]] = {}

    for canon, alias_list in aliases.items():
        best_col = None
        best_score = 0.0

        for col in columns:
            scores = [_similarity(col, alias) for alias in alias_list + [canon]]
            score = max(scores)
            if score > best_score:
                best_score = score
                best_col = col

        suggestions[canon] = best_col if best_score >= 0.62 else None

    return suggestions


def validate_mapping(mapping: Dict[str, str], template: str, df_columns: List[str]) -> None:
    spec = CANONICAL_FIELDS[template]
    required = spec["required"]

    missing_required = [c for c in required if not mapping.get(c)]
    if missing_required:
        raise ValueError(f"Не выбраны обязательные поля mapping: {missing_required}")

    bad_sources = [src for src in mapping.values() if src and src not in df_columns]
    if bad_sources:
        raise ValueError(f"В mapping указаны отсутствующие колонки: {bad_sources}")


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    cols = {}
    for canon, src in mapping.items():
        if not src:
            continue
        if src not in df.columns:
            raise ValueError(f"Missing source column '{src}' for canonical '{canon}'")
        cols[src] = canon

    out = df.rename(columns=cols).copy()
    selected_cols = [canon for canon, src in mapping.items() if src]
    return out[selected_cols]