from __future__ import annotations

from typing import Any

import pandas as pd


LOW_CARDINALITY_LIMIT = 12
MEDIUM_CARDINALITY_LIMIT = 50
MAX_ONEHOT_VALUES = 8


def is_textual(series: pd.Series) -> bool:
    return series.dtype == "object" or str(series.dtype).startswith("string")


def _looks_like_datetime_name(name: str) -> bool:
    low = str(name).lower()
    return any(token in low for token in ["date", "time", "timestamp", "datetime", "period", "дата", "время"])


def _datetime_parse_share(series: pd.Series, col_name: str) -> float:
    """
    Быстрая оценка доли дат:
    - не пытаемся парсить весь столбец каждый раз;
    - парсим только небольшой sample для текстовых колонок.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0

    if pd.api.types.is_numeric_dtype(series):
        return 0.0

    s = series.dropna()
    if s.empty:
        return 0.0

    sample_size = min(200, len(s))
    sample = s.astype(str).head(sample_size)
    if not _looks_like_datetime_name(col_name):
        # Для неочевидных колонок вообще не делаем тяжёлый dateutil parse:
        # это главный источник лагов и предупреждений.
        return 0.0
    dt = pd.to_datetime(sample, errors="coerce")
    return float(dt.notna().mean())


def _sample_values(series: pd.Series, limit: int = 4) -> list[str]:
    values = series.dropna().astype(str).head(limit).tolist()
    return [v[:60] for v in values]


def _is_id_like(name: str, unique_share: float, nunique: int) -> bool:
    low = name.lower()
    return unique_share > 0.97 and nunique > 200 and any(
        token in low for token in ["id", "uuid", "guid", "token", "hash", "key"]
    )


def profile_extra_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    nunique = int(s.nunique(dropna=True))
    non_null = int(s.notna().sum())
    unique_share = float(nunique / max(non_null, 1))
    missing_share = float(s.isna().mean())
    dtype_name = str(s.dtype)
    num = pd.to_numeric(s, errors="coerce")
    num_share = float(num.notna().mean())
    dt_share = _datetime_parse_share(s, col)
    samples = _sample_values(s)

    if pd.api.types.is_numeric_dtype(s) or num_share >= 0.9:
        kind = "numeric"
    elif dt_share >= 0.9:
        kind = "datetime"
    elif _is_id_like(col, unique_share, nunique):
        kind = "id_like"
    elif is_textual(s):
        mean_len = float(s.dropna().astype(str).str.len().mean() or 0.0)
        if nunique <= LOW_CARDINALITY_LIMIT:
            kind = "categorical_low_card"
        elif nunique <= MEDIUM_CARDINALITY_LIMIT:
            kind = "categorical_medium_card"
        elif mean_len >= 40:
            kind = "free_text"
        else:
            kind = "categorical_high_card"
    elif nunique <= LOW_CARDINALITY_LIMIT:
        kind = "categorical_low_card"
    elif nunique <= MEDIUM_CARDINALITY_LIMIT:
        kind = "categorical_medium_card"
    else:
        kind = "categorical_high_card"

    value_counts = s.fillna("NA").astype(str).value_counts(dropna=False)
    top_values = [str(v) for v in value_counts.head(MAX_ONEHOT_VALUES).index.tolist()]

    return {
        "column": col,
        "dtype": dtype_name,
        "kind": kind,
        "nunique": nunique,
        "non_null": non_null,
        "unique_share": unique_share,
        "missing_share": missing_share,
        "num_share": num_share,
        "dt_share": dt_share,
        "samples": samples,
        "top_values": top_values,
    }


def suggest_feature_encoding(profile: dict[str, Any]) -> tuple[str | None, str]:
    kind = str(profile.get("kind") or "")
    nunique = int(profile.get("nunique") or 0)

    if kind in {"numeric", "datetime"}:
        return None, "Колонка агрегируется в числовые признаки автоматически."
    if kind == "id_like":
        return "label", "Похоже на технический идентификатор: лучше исключить из обучения."
    if kind == "free_text":
        return "label", "Свободный текст лучше сжимать в компактные числовые сводки."
    if nunique <= LOW_CARDINALITY_LIMIT:
        return "onehot", f"Низкая кардинальность ({nunique} значений): one-hot будет интерпретируемым."
    if nunique <= MEDIUM_CARDINALITY_LIMIT:
        return "label", f"Кардинальность умеренная ({nunique} значений): компактное кодирование устойчивее."
    return "label", f"Высокая кардинальность ({nunique} значений): one-hot сильно раздует число признаков."


def build_feature_config(df: pd.DataFrame, col: str, *, keep: bool = True) -> dict[str, Any]:
    profile = profile_extra_column(df, col)
    encoding, recommendation = suggest_feature_encoding(profile)
    return {
        "column": col,
        "keep": bool(keep),
        "kind": profile["kind"],
        "dtype": profile["dtype"],
        "nunique": profile["nunique"],
        "missing_share": profile["missing_share"],
        "samples": profile["samples"],
        "encoding": encoding,
        "recommended_encoding": encoding,
        "recommendation": recommendation,
        "top_values": profile["top_values"],
    }


def audit_extra_column(df: pd.DataFrame, col: str) -> tuple[str, str]:
    """
    Оценка пригодности колонки как *дополнительного* признака для модели.
    """
    name = col.lower()
    profile = profile_extra_column(df, col)
    nunique = int(profile["nunique"])
    unique_share = float(profile["unique_share"])
    missing_share = float(profile["missing_share"])
    kind = str(profile["kind"])

    if any(x in name for x in ["target", "label", "churn", "outcome", "future", "next_", "status_after"]):
        return "block", "Похоже на целевую метку или данные из будущего — в модель не рекомендуется."

    if kind == "id_like":
        return "block", "Почти все значения уникальны и колонка похожа на технический идентификатор."

    if unique_share > 0.97 and nunique > 200:
        return "review", "Почти все значения уникальны — похоже на технический идентификатор, а не на общий признак."

    if missing_share > 0.6:
        return "review", "Много пропусков — признак может быть нестабильным."

    if kind == "numeric":
        return "allow", "Числовой признак — модель может использовать агрегаты напрямую."
    if kind == "datetime":
        return "allow", "Дата будет преобразована в интервальные признаки относительно точки прогноза."
    if kind == "categorical_low_card":
        return "review", f"Категориальный признак с {nunique} значениями: подойдёт one-hot."
    if kind in {"categorical_medium_card", "categorical_high_card"}:
        return "review", f"Категориальный признак с {nunique} значениями: нужен компактный режим кодирования."
    if kind == "free_text":
        return "review", "Свободный текст лучше использовать только в компактном виде, без one-hot."

    return "review", "Тип неоднозначен: проверьте колонку вручную перед добавлением в модель."


def find_target_like_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in ["target", "label", "class", "churn", "outcome"]):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 10:
                return col
    return None


def pick_best_date_col(df: pd.DataFrame, diagnostics: dict) -> str | None:
    candidates = diagnostics.get("date_candidates", []) or []
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in ["date", "time", "timestamp", "datetime"]):
            return col
    return None


def pick_best_id_col(df: pd.DataFrame, diagnostics: dict) -> str | None:
    candidates = diagnostics.get("id_candidates", []) or []
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in ["customer", "client", "user", "account", "id"]):
            return col
    return None


def transactions_financial_sources_ok(mapping: dict) -> bool:
    """
    Для шаблона «транзакции»: достаточно колонки суммы или пары цена × количество.
    Логика должна совпадать с app.mapping.validate_mapping.
    """
    if mapping.get("amount"):
        return True
    return bool(mapping.get("unit_price")) and bool(mapping.get("quantity"))
