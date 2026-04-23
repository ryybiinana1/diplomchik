# churnlib/feature_module.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def aggregate_extra_features(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    extra_cols: List[str],
    extra_feature_config: Optional[Dict[str, Dict[str, Any]]],
    anchor_time: pd.Timestamp,
    prefix: str = "extra",
    max_cols: int = 20,
) -> pd.DataFrame:
    """
    Превращает произвольные extra cols в числовые агрегаты по entity.
    Важно: НЕ добавляем строковые колонки напрямую (иначе ML упадёт).
    """
    extra_cols = [c for c in extra_cols if c in df.columns]
    extra_cols = extra_cols[:max_cols]
    if not extra_cols:
        return pd.DataFrame(columns=[id_col]).set_index(id_col)

    # сортировка для last
    df_sorted = df.sort_values(time_col).copy()

    out_parts = []
    extra_feature_config = extra_feature_config or {}

    for col in extra_cols:
        s = df[col]
        col_cfg = extra_feature_config.get(col) or {}

        # 1) пробуем datetime
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        dt_share = float(dt.notna().mean())

        if dt_share >= 0.8:
            # возраст признака в днях (anchor_time - dt)
            age_days = (anchor_time - dt).dt.total_seconds() / 86400.0
            tmp = pd.DataFrame({id_col: df[id_col], "_v": age_days})
            agg = tmp.groupby(id_col)["_v"].agg(["mean", "std", "min", "max"]).rename(
                columns={
                    "mean": f"{prefix}__{col}__age_mean_days",
                    "std": f"{prefix}__{col}__age_std_days",
                    "min": f"{prefix}__{col}__age_min_days",
                    "max": f"{prefix}__{col}__age_max_days",
                }
            )
            out_parts.append(agg)
            continue

        # 2) пробуем numeric
        num = pd.to_numeric(s, errors="coerce")
        num_share = float(num.notna().mean())

        if num_share >= 0.8:
            tmp = pd.DataFrame({id_col: df[id_col], "_v": num})
            agg = tmp.groupby(id_col)["_v"].agg(["mean", "std", "min", "max", "sum"]).rename(
                columns={
                    "mean": f"{prefix}__{col}__mean",
                    "std": f"{prefix}__{col}__std",
                    "min": f"{prefix}__{col}__min",
                    "max": f"{prefix}__{col}__max",
                    "sum": f"{prefix}__{col}__sum",
                }
            )
            # last numeric
            last = (
                pd.DataFrame({id_col: df_sorted[id_col], "_v": pd.to_numeric(df_sorted[col], errors="coerce")})
                .dropna(subset=["_v"])
                .groupby(id_col)["_v"]
                .last()
                .rename(f"{prefix}__{col}__last")
            )
            agg = agg.join(last, how="left")
            out_parts.append(agg)
            continue

        # 3) categorical → либо one-hot по top values, либо компактное кодирование
        cat = s.astype(str)
        cat = cat.replace("nan", np.nan).fillna("NA")

        # nunique
        nunique = df.groupby(id_col)[col].nunique(dropna=True).rename(f"{prefix}__{col}__nunique")

        # top1 share + entropy (через loop, т.к. групповые value_counts)
        ent = {}
        top1 = {}
        for gid, sub in df[[id_col, col]].copy().fillna("NA").groupby(id_col):
            vc = sub[col].astype(str).value_counts()
            ent[gid] = _safe_entropy(vc.values)
            top1[gid] = float(vc.iloc[0] / vc.sum()) if vc.sum() else 0.0

        ent_s = pd.Series(ent, name=f"{prefix}__{col}__entropy")
        top1_s = pd.Series(top1, name=f"{prefix}__{col}__top1_share")

        # missing share per entity
        miss = df[col].isna().groupby(df[id_col]).mean().rename(f"{prefix}__{col}__missing_share")

        agg = pd.concat([nunique, ent_s, top1_s, miss], axis=1)

        encoding = str(col_cfg.get("encoding") or "label")
        top_values = [str(v) for v in (col_cfg.get("top_values") or [])]
        if encoding == "onehot" and top_values:
            safe = df[[id_col, col]].copy().fillna("NA")
            safe[col] = safe[col].astype(str)
            last_values = (
                df_sorted[[id_col, col]]
                .copy()
                .fillna("NA")
                .groupby(id_col)[col]
                .last()
                .astype(str)
            )
            for raw_value in top_values:
                key = str(raw_value)
                suffix = "".join(ch if ch.isalnum() else "_" for ch in key.lower()).strip("_")[:24] or "value"
                share = (
                    (safe[col] == key)
                    .groupby(safe[id_col])
                    .mean()
                    .rename(f"{prefix}__{col}__share_{suffix}")
                )
                last_is = (
                    (last_values == key)
                    .astype(float)
                    .rename(f"{prefix}__{col}__last_is_{suffix}")
                )
                agg = pd.concat([agg, share, last_is], axis=1)
        else:
            vocab = {value: idx for idx, value in enumerate(top_values, start=1)}
            tmp = df[[id_col, col]].copy().fillna("NA")
            tmp[col] = tmp[col].astype(str).map(lambda x: vocab.get(x, 0)).astype(float)
            code_stats = tmp.groupby(id_col)[col].agg(["mean", "max", "last"]).rename(
                columns={
                    "mean": f"{prefix}__{col}__code_mean",
                    "max": f"{prefix}__{col}__code_max",
                    "last": f"{prefix}__{col}__code_last",
                }
            )
            agg = pd.concat([agg, code_stats], axis=1)
        out_parts.append(agg)

    out = pd.concat(out_parts, axis=1)
    out.index.name = id_col
    return out


def build_transaction_features(
    df: pd.DataFrame,
    anchor_time: pd.Timestamp,
    extra_feature_cols: Optional[List[str]] = None,
    extra_feature_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Строит RFM‑подобные признаки по транзакциям.
    Возвращает DataFrame с customer_id + числовые признаки.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "is_cancellation" in df.columns:
        df["is_cancellation"] = df["is_cancellation"].astype(bool)
    else:
        df["is_cancellation"] = False

    # берём только "реальные покупки"
    purchase_df = df.loc[~df["is_cancellation"]].copy()
    if purchase_df.empty:
        return pd.DataFrame()

    # базовые агрегаты
    g = purchase_df.groupby("customer_id", as_index=False)

    freq = g["transaction_id"].count().rename(columns={"transaction_id": "frequency_tx"})
    monetary = g["amount"].sum().rename(columns={"amount": "monetary"})
    amount_mean = g["amount"].mean().rename(columns={"amount": "amount_mean"})
    amount_std = g["amount"].std().rename(columns={"amount": "amount_std"})

    first_time = g["event_time"].min().rename(columns={"event_time": "first_purchase_time"})
    last_time = g["event_time"].max().rename(columns={"event_time": "last_purchase_time"})

    feat = freq.merge(monetary, on="customer_id").merge(amount_mean, on="customer_id").merge(amount_std, on="customer_id")
    feat = feat.merge(first_time, on="customer_id").merge(last_time, on="customer_id")

    feat["recency_days"] = (anchor_time - feat["last_purchase_time"]).dt.total_seconds() / 86400.0
    feat["customer_lifetime_days"] = (anchor_time - feat["first_purchase_time"]).dt.total_seconds() / 86400.0

    # межпокупочный интервал
    purchase_df = purchase_df.sort_values(["customer_id", "event_time"])
    purchase_df["prev_time"] = purchase_df.groupby("customer_id")["event_time"].shift(1)
    purchase_df["delta_days"] = (purchase_df["event_time"] - purchase_df["prev_time"]).dt.total_seconds() / 86400.0
    ip = purchase_df.groupby("customer_id")["delta_days"].agg(["mean", "std"]).rename(
        columns={"mean": "interpurchase_mean_days", "std": "interpurchase_std_days"}
    )
    feat = feat.merge(ip, left_on="customer_id", right_index=True, how="left")

    # optional: quantity
    if "quantity" in purchase_df.columns:
        purchase_df["quantity"] = pd.to_numeric(purchase_df["quantity"], errors="coerce")
        q = purchase_df.groupby("customer_id")["quantity"].agg(["sum", "mean"]).rename(
            columns={"sum": "qty_sum", "mean": "qty_mean"}
        )
        feat = feat.merge(q, left_on="customer_id", right_index=True, how="left")

    # optional: unit_price
    if "unit_price" in purchase_df.columns:
        purchase_df["unit_price"] = pd.to_numeric(purchase_df["unit_price"], errors="coerce")
        up = purchase_df.groupby("customer_id")["unit_price"].agg(["mean", "std"]).rename(
            columns={"mean": "unit_price_mean", "std": "unit_price_std"}
        )
        feat = feat.merge(up, left_on="customer_id", right_index=True, how="left")

    # optional: item diversity
    if "item_id" in purchase_df.columns:
        item_n = purchase_df.groupby("customer_id")["item_id"].nunique(dropna=True).rename("item_nunique")
        feat = feat.merge(item_n, left_on="customer_id", right_index=True, how="left")

    # optional: country summary (как простой proxy)
    if "country" in purchase_df.columns:
        country_nu = purchase_df.groupby("customer_id")["country"].nunique(dropna=True).rename("country_nunique")
        feat = feat.merge(country_nu, left_on="customer_id", right_index=True, how="left")

    # NEW: extra feature cols
    if extra_feature_cols:
        extra_df = aggregate_extra_features(
            purchase_df,
            id_col="customer_id",
            time_col="event_time",
            extra_cols=list(extra_feature_cols),
            extra_feature_config=extra_feature_config,
            anchor_time=anchor_time,
            prefix="extra",
            max_cols=20,
        ).reset_index()
        feat = feat.merge(extra_df, on="customer_id", how="left")

    # финальная чистка
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # убираем служебные datetime (модели их не любят)
    feat = feat.drop(columns=["first_purchase_time", "last_purchase_time"], errors="ignore")

    return feat
