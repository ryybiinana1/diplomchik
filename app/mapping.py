# app/mapping.py
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

TemplateName = Literal["transactions", "subscriptions", "events"]
AuditVerdict = Literal["allow", "review", "block"]


# 1) Контракты (что обязательно / что опционально)
# Эти поля должны совпадать с churnlib.data_module.canonicalize_types и build_snapshots. citeturn4view0turn3view0turn3view3
CONTRACTS: Dict[TemplateName, Dict[str, Any]] = {
    "transactions": {
        "label": "Покупки / транзакции",
        "required": {
            "customer_id": {
                "title": "ID клиента",
                "type": "string",
                "meaning": "Уникальный идентификатор клиента.",
            },
            "transaction_id": {
                "title": "ID транзакции",
                "type": "string",
                "meaning": "Уникальный идентификатор операции/заказа.",
            },
            "event_time": {
                "title": "Дата/время операции",
                "type": "datetime",
                "meaning": "Когда была совершена покупка/операция.",
            },
        },
        "optional": {
            "amount": {
                "title": "Сумма операции",
                "type": "float",
                "meaning": "Общая выручка по строке. Если в файле только «цена за штуку» и «количество», оставьте пустым и заполните поля «Цена за единицу» и «Количество» ниже — сумма будет посчитана как цена × количество.",
            },
            "item_id": {
                "title": "ID товара",
                "type": "string",
                "meaning": "Идентификатор товара/позиции (если доступно).",
            },
            "quantity": {
                "title": "Количество",
                "type": "integer",
                "meaning": "Количество единиц товара/позиции (если доступно).",
            },
            "country": {
                "title": "Страна",
                "type": "string",
                "meaning": "Страна клиента/операции (если доступно).",
            },
            "unit_price": {
                "title": "Цена за единицу",
                "type": "float",
                "meaning": "Цена за одну штуку. Вместе с «Количеством» заменяет колонку «Сумма операции», если отдельной суммы в файле нет.",
            },
            "is_cancellation": {
                "title": "Флаг отмены",
                "type": "boolean",
                "meaning": "True/False: операция — отмена/возврат (если доступно).",
            },
            "promo_code": {
                "title": "Промо/акция",
                "type": "string",
                "meaning": "Промокод/акция/кампания (если есть).",
            },
        },
        "entity_id_field": "customer_id",
        "time_field": "event_time",
    },
    "subscriptions": {
        "label": "Подписки",
        "required": {
            "account_id": {
                "title": "ID аккаунта",
                "type": "string",
                "meaning": "Идентификатор подписки/аккаунта.",
            },
            "period_start": {
                "title": "Начало периода",
                "type": "date",
                "meaning": "Дата начала биллингового периода.",
            },
            "period_end": {
                "title": "Конец периода",
                "type": "date",
                "meaning": "Дата окончания биллингового периода.",
            },
            "mrr": {
                "title": "MRR / платёж",
                "type": "float",
                "meaning": "Выручка/платёж за период (MRR или аналог).",
            },
            "subscription_status": {
                "title": "Статус подписки",
                "type": "string",
                "meaning": "Например: active/canceled/paused (как в данных).",
            },
        },
        "optional": {
            "churn_date": {
                "title": "Дата оттока",
                "type": "date",
                "meaning": "Если есть — дата отмены/оттока (используем для диагностики, но осторожно).",
            },
        },
        "entity_id_field": "account_id",
        "time_field": "period_end",
    },
    "events": {
        "label": "События пользователей",
        "required": {
            "subject_id": {
                "title": "ID пользователя",
                "type": "string",
                "meaning": "Уникальный идентификатор пользователя.",
            },
            "event_time": {
                "title": "Дата/время события",
                "type": "datetime",
                "meaning": "Когда произошло событие.",
            },
            "event_name": {
                "title": "Тип события",
                "type": "string",
                "meaning": "Название/тип события (login, view, purchase и т.п.).",
            },
        },
        "optional": {
            "account_id": {
                "title": "ID аккаунта",
                "type": "string",
                "meaning": "Если есть в событиях — привязка к аккаунту/организации.",
            },
        },
        "entity_id_field": "subject_id",
        "time_field": "event_time",
    },
}


# 2) Алиасы для авто-маппинга (эвристика)
ALIASES: Dict[TemplateName, Dict[str, List[str]]] = {
    "transactions": {
        "customer_id": ["customer", "client", "user", "buyer", "cust_id", "client_id", "user_id", "uid"],
        "transaction_id": ["transaction", "order", "purchase", "invoice", "check", "tx_id", "order_id"],
        "event_time": ["event_time", "date", "datetime", "timestamp", "created_at", "paid_at", "purchase_date"],
        "amount": ["amount", "price", "revenue", "total", "sum", "value", "gmv", "order_total"],
        "item_id": ["item", "product", "sku", "product_id", "itemid"],
        "quantity": ["qty", "quantity", "count", "units", "pieces", "pcs", "num_items", "items_count", "n_items"],
        "country": ["country", "geo", "region_country", "billing_country"],
        "unit_price": [
            "unit_price",
            "price_per_unit",
            "ppu",
            "item_price",
            "product_price",
            "unit_cost",
            "price_for_item",
            "line_price",
            "cost",
            "goods_price",
            "product_cost",
        ],
        "is_cancellation": ["is_cancellation", "is_refund", "refund", "cancel", "cancellation", "returned"],
        "promo_code": ["promo", "promo_code", "coupon", "discount_code", "campaign"],
    },
    "subscriptions": {
        "account_id": ["account", "tenant", "org", "company", "workspace", "account_id", "org_id"],
        "period_start": ["period_start", "start_date", "from", "begin"],
        "period_end": ["period_end", "end_date", "to", "finish"],
        "mrr": ["mrr", "revenue", "amount", "payment", "charge", "value"],
        "subscription_status": ["subscription_status", "status", "state"],
        "churn_date": ["churn_date", "cancel_date", "cancellation_date", "ended_at", "closed_at"],
    },
    "events": {
        "subject_id": ["subject", "user", "customer", "client", "uid", "user_id", "client_id"],
        "event_time": ["event_time", "timestamp", "datetime", "date", "created_at"],
        "event_name": ["event_name", "event", "event_type", "action", "name"],
        "account_id": ["account_id", "org_id", "workspace_id", "tenant_id"],
    },
}


def get_contract(template: str) -> Dict[str, Any]:
    if template not in CONTRACTS:
        raise ValueError(f"Unknown template: {template}")
    return CONTRACTS[template]  # type: ignore[return-value]


def canonical_fields(template: str) -> List[str]:
    c = get_contract(template)
    return list(c["required"].keys()) + list(c["optional"].keys())


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def detect_mapping(columns: List[str], template: str, threshold: float = 0.62) -> Dict[str, str]:
    """
    Возвращает словарь canonical_field -> выбранная колонка (или "" если не найдено).
    """
    contract = get_contract(template)
    all_fields = list(contract["required"].keys()) + list(contract["optional"].keys())
    cols_norm = {c: _norm(c) for c in columns}

    out: Dict[str, str] = {}
    for field in all_fields:
        best_col = ""
        best_score = 0.0

        field_norm = _norm(field)
        field_aliases = [_norm(x) for x in ALIASES.get(template, {}).get(field, [])]

        for col, cn in cols_norm.items():
            # прямое совпадение по алиасам
            if cn == field_norm or cn in field_aliases:
                best_col = col
                best_score = 1.0
                break

            # fuzzy
            score = max(_sim(cn, field_norm), *( _sim(cn, a) for a in field_aliases )) if field_aliases else _sim(cn, field_norm)
            if score > best_score:
                best_score = score
                best_col = col

        out[field] = best_col if best_score >= threshold else ""

    return out


def transactions_financial_sources_ok(mapping: Dict[str, Any]) -> bool:
    """Достаточно данных для суммы строки: либо колонка amount, либо unit_price и quantity."""
    if mapping.get("amount"):
        return True
    return bool(mapping.get("unit_price")) and bool(mapping.get("quantity"))


def validate_mapping(mapping: Dict[str, str], template: str, available_columns: List[str]) -> None:
    contract = get_contract(template)
    required = list(contract["required"].keys())

    if template == "transactions":
        if not transactions_financial_sources_ok(mapping):
            raise ValueError(
                "Для транзакций укажите колонку «Сумма операции» или обе колонки "
                "«Цена за единицу» и «Количество» (сумма строки будет цена × количество)."
            )
        core = ["customer_id", "transaction_id", "event_time"]
        if mapping.get("amount"):
            core.append("amount")
        else:
            core.extend(["unit_price", "quantity"])
        missing_required = [f for f in core if not mapping.get(f)]
    else:
        missing_required = [f for f in required if not mapping.get(f)]

    if missing_required:
        raise ValueError(f"Не сопоставлены обязательные поля: {missing_required}")

    # проверка: все выбранные колонки существуют
    avail = set(available_columns)
    bad = {k: v for k, v in mapping.items() if v and v not in avail}
    if bad:
        raise ValueError(f"В mapping указаны колонки, которых нет в CSV: {bad}")

    # проверка: один source -> не маппить в два поля
    used = [v for v in mapping.values() if v]
    dup = {x for x in used if used.count(x) > 1}
    if dup:
        raise ValueError(f"Одна и та же колонка сопоставлена в несколько полей: {sorted(dup)}")


def rename_to_canonical(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Renames selected columns to canonical names, keeping all other columns.
    mapping is canonical -> source_name.
    """
    rename_map = {src: canon for canon, src in mapping.items() if src}
    return df.rename(columns=rename_map).copy()


# --- Leakage guard for extra feature columns ---

_LEAK_PATTERNS_BLOCK = [
    r"\bchurn\b",
    r"\bchurned\b",
    r"\btarget\b",
    r"\blabel\b",
    r"\boutcome\b",
    r"\bground[_\s]?truth\b",
    r"\bretained\b",
    r"\breten(tion|tive)\b",
    r"\bcancel(led|lation)?\b",
    r"\bclosed\b",
    r"\blost\b",
    r"\bstatus_after\b",
]

_LEAK_PATTERNS_REVIEW = [
    r"\bnext\b",
    r"\bfuture\b",
    r"\bafter\b",
    r"\buntil\b",
    r"\bdays?_to\b",
    r"\bdays?_until\b",
    r"\bdays?_since\b",
    r"\btime_to\b",
    r"\btime_since\b",
    r"\bfinal\b",
    r"\bend_state\b",
]

_ID_LIKE = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\bhash\b", r"\btoken\b", r"\bkey\b"]


def _matches_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def audit_column_for_leakage(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Возвращает verdict + человеческое объяснение.
    """
    coln = _norm(col)

    non_null = int(df[col].notna().sum())
    nunique = int(df[col].nunique(dropna=True))
    unique_share = float(nunique / max(non_null, 1))
    missing_share = float(1.0 - (non_null / max(len(df), 1)))

    # 0) мусор/константа
    if nunique <= 1:
        return {
            "column": col,
            "verdict": "block",
            "reason": "Почти константная колонка — не даёт сигнал модели.",
            "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
        }

    # 1) прямые маркеры таргета/исхода
    if _matches_any(_LEAK_PATTERNS_BLOCK, coln):
        return {
            "column": col,
            "verdict": "block",
            "reason": "Похоже на прямой маркер оттока/исхода (риск утечки таргета).",
            "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
        }

    # 2) сигналы будущего / пост-фактум
    if _matches_any(_LEAK_PATTERNS_REVIEW, coln):
        return {
            "column": col,
            "verdict": "review",
            "reason": "Название похоже на признак из будущего или посчитанный пост‑фактум. Использовать осторожно.",
            "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
        }

    # 3) почти уникальные — чаще всего технический ID => либо блок, либо осторожно
    if unique_share >= 0.97 and nunique >= 200:
        if _matches_any(_ID_LIKE, coln):
            return {
                "column": col,
                "verdict": "block",
                "reason": "Похоже на технический ID (почти все значения уникальны) — модель может переобучиться.",
                "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
            }
        return {
            "column": col,
            "verdict": "review",
            "reason": "Почти уникальные значения — есть риск переобучения. Лучше проверить смысл колонки.",
            "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
        }

    # 4) много пропусков — предупреждение
    if missing_share >= 0.6:
        return {
            "column": col,
            "verdict": "review",
            "reason": "Слишком много пропусков — может ухудшить качество и устойчивость.",
            "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
        }

    return {
        "column": col,
        "verdict": "allow",
        "reason": "Выглядит безопасно (по базовым эвристикам).",
        "signals": {"nunique": nunique, "unique_share": unique_share, "missing_share": missing_share},
    }


def audit_extra_columns(
    df: pd.DataFrame,
    template: str,
    mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Возвращает аудит всех 'лишних' колонок (не входящих в контракт и не выбранных в mapping).
    """
    contract_cols = set(canonical_fields(template))
    mapped_source_cols = {v for v in mapping.values() if v}
    mapped_canonical_cols = {k for k, v in mapping.items() if v}
    used_cols = contract_cols | mapped_source_cols | mapped_canonical_cols

    extra_cols = [c for c in df.columns if c not in used_cols]
    audits = [audit_column_for_leakage(df, c) for c in extra_cols]
    return sorted(audits, key=lambda x: {"block": 0, "review": 1, "allow": 2}[x["verdict"]])


def sanitize_extra_feature_columns(
    df: pd.DataFrame,
    template: str,
    mapping: Dict[str, str],
    extra_cols: List[str],
    max_cols: int = 20,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Жёстко фильтрует extra cols.
    Возвращает: (final_extra_cols, audit_for_selected)
    """
    if not extra_cols:
        return [], []

    contract_cols = set(canonical_fields(template))
    mapped_source_cols = {v for v in mapping.values() if v}
    mapped_canonical_cols = {k for k, v in mapping.items() if v}
    forbidden = contract_cols | mapped_source_cols | mapped_canonical_cols

    # 1) только существующие
    candidate = [c for c in extra_cols if c in df.columns and c not in forbidden]

    # 2) лимит
    candidate = candidate[:max_cols]

    audits = [audit_column_for_leakage(df, c) for c in candidate]
    allowed = [a["column"] for a in audits if a["verdict"] in ("allow", "review")]
    return allowed, audits
