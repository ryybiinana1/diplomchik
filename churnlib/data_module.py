from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

from churnlib.feature_module import build_transaction_features


@dataclass(frozen=True)
class SnapshotConfig:
    template: str  # "transactions" | "subscriptions" | "events"
    horizon_days: int = 30
    history_days: int = 180
    step_days: int = 30
    min_events_in_history: int = 1
    min_lifetime_days: int = 0
    max_recency_days: Optional[int] = None
    tz: Optional[str] = None


def canonicalize_types(df: pd.DataFrame, template: str) -> pd.DataFrame:
    df = df.copy()

    if template == "transactions":
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["customer_id"] = df["customer_id"].astype(str)
        df["transaction_id"] = df["transaction_id"].astype(str)

        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype(str)
        if "quantity" in df.columns:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        if "country" in df.columns:
            df["country"] = df["country"].astype(str)
        if "unit_price" in df.columns:
            df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        if "is_cancellation" in df.columns:
            df["is_cancellation"] = df["is_cancellation"].astype(bool)
        else:
            df["is_cancellation"] = df["transaction_id"].str.startswith("C")

    elif template == "subscriptions":
        df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce").dt.date
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date
        if "churn_date" in df.columns:
            df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce").dt.date
        df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce")
        df["account_id"] = df["account_id"].astype(str)
        df["subscription_status"] = df["subscription_status"].astype(str)

    elif template == "events":
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        df["subject_id"] = df["subject_id"].astype(str)
        df["event_name"] = df["event_name"].astype(str)
        if "account_id" in df.columns:
            df["account_id"] = df["account_id"].astype(str)

    else:
        raise ValueError(f"Unknown template: {template}")

    return df


def build_snapshots(df: pd.DataFrame, cfg: SnapshotConfig) -> pd.DataFrame:
    template = cfg.template
    rows: List = []

    if template == "transactions":
        df = df.dropna(subset=["event_time", "customer_id", "transaction_id", "amount"]).copy()

        tmin = df["event_time"].min().normalize()
        tmax = df["event_time"].max().normalize()

        start = tmin + pd.Timedelta(days=cfg.history_days)
        end = tmax - pd.Timedelta(days=cfg.horizon_days)

        anchors = pd.date_range(start=start, end=end, freq=f"{cfg.step_days}D", tz="UTC")

        for t in anchors:
            past_start = t - pd.Timedelta(days=cfg.history_days)
            future_end = t + pd.Timedelta(days=cfg.horizon_days)

            past = df[(df["event_time"] > past_start) & (df["event_time"] <= t)].copy()
            if past.empty:
                continue

            future = df[(df["event_time"] > t) & (df["event_time"] <= future_end)].copy()
            future_buyers = set(future.loc[~future["is_cancellation"], "customer_id"].unique())

            feat = build_transaction_features(past, t)
            if feat.empty:
                continue

            # policy filters
            if cfg.min_events_in_history > 1 and "frequency_tx" in feat.columns:
                feat = feat[feat["frequency_tx"] >= cfg.min_events_in_history].copy()

            if cfg.min_lifetime_days > 0 and "customer_lifetime_days" in feat.columns:
                feat = feat[feat["customer_lifetime_days"] >= cfg.min_lifetime_days].copy()

            if cfg.max_recency_days is not None and "recency_days" in feat.columns:
                feat = feat[feat["recency_days"] <= cfg.max_recency_days].copy()

            if feat.empty:
                continue

            feat["anchor_time"] = t
            feat["target"] = feat["customer_id"].apply(lambda cid: 0 if cid in future_buyers else 1)
            feat["value_proxy"] = feat["monetary"].fillna(0.0)

            feat = feat.rename(columns={"customer_id": "entity_id"})
            rows.append(feat)

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return out

    elif template == "subscriptions":
        df = df.dropna(subset=["account_id", "period_start", "period_end", "mrr", "subscription_status"]).copy()
        df["period_end_dt"] = pd.to_datetime(df["period_end"])

        tmin = df["period_end_dt"].min().normalize()
        tmax = df["period_end_dt"].max().normalize()

        start = tmin + pd.Timedelta(days=cfg.history_days)
        end = tmax - pd.Timedelta(days=cfg.horizon_days)
        anchors = pd.date_range(start=start, end=end, freq=f"{cfg.step_days}D")

        rows = []
        for t in anchors:
            past_start = t - pd.Timedelta(days=cfg.history_days)
            future_end = t + pd.Timedelta(days=cfg.horizon_days)

            past = df[(df["period_end_dt"] > past_start) & (df["period_end_dt"] <= t)]
            if past.empty:
                continue

            future = df[(df["period_end_dt"] > t) & (df["period_end_dt"] <= future_end)]
            future_active = set(
                future.loc[future["subscription_status"].isin(["active"]), "account_id"].unique()
            )

            last_end = past.groupby("account_id")["period_end_dt"].max()
            mrr_sum = past.groupby("account_id")["mrr"].sum()
            mrr_last = (
                past.sort_values("period_end_dt")
                .groupby("account_id", as_index=True)["mrr"]
                .last()
            )

            accounts = past["account_id"].unique()
            for aid in accounts:
                recency = (t - last_end.get(aid)).days if pd.notna(last_end.get(aid, pd.NaT)) else np.nan

                rows.append({
                    "entity_id": aid,
                    "anchor_time": t,
                    "target": 0 if aid in future_active else 1,
                    "value_proxy": float(mrr_last.get(aid, mrr_sum.get(aid, 0.0))),
                    "recency_days": recency,
                    "mrr_sum": float(mrr_sum.get(aid, 0.0)),
                    "mrr_last": float(mrr_last.get(aid, np.nan)) if aid in mrr_last.index else np.nan,
                })

        out = pd.DataFrame(rows)
        return out

    elif template == "events":
        df = df.dropna(subset=["subject_id", "event_time", "event_name"]).copy()

        tmin = df["event_time"].min().normalize()
        tmax = df["event_time"].max().normalize()

        start = tmin + pd.Timedelta(days=cfg.history_days)
        end = tmax - pd.Timedelta(days=cfg.horizon_days)
        anchors = pd.date_range(start=start, end=end, freq=f"{cfg.step_days}D", tz="UTC")

        rows = []
        for t in anchors:
            past_start = t - pd.Timedelta(days=cfg.history_days)
            future_end = t + pd.Timedelta(days=cfg.horizon_days)

            past = df[(df["event_time"] > past_start) & (df["event_time"] <= t)]
            if past.empty:
                continue

            future = df[(df["event_time"] > t) & (df["event_time"] <= future_end)]
            future_active = set(future["subject_id"].unique())

            last_evt = past.groupby("subject_id")["event_time"].max()
            freq = past.groupby("subject_id")["event_name"].count()
            subjects = past["subject_id"].unique()

            for sid in subjects:
                if int(freq.get(sid, 0)) < cfg.min_events_in_history:
                    continue

                recency = (t - last_evt.get(sid)).days if pd.notna(last_evt.get(sid, pd.NaT)) else np.nan
                rows.append({
                    "entity_id": sid,
                    "anchor_time": t,
                    "target": 0 if sid in future_active else 1,
                    "value_proxy": float(freq.get(sid, 0)),
                    "recency_days": recency,
                    "event_count": int(freq.get(sid, 0)),
                })

        out = pd.DataFrame(rows)
        return out

    else:
        raise ValueError(f"Unknown template: {template}")
    
def assert_no_time_leakage(df_snapshots: pd.DataFrame, time_col: str = "anchor_time") -> None:
    if time_col not in df_snapshots.columns:
        raise ValueError(f"Missing {time_col} in snapshots")
    if df_snapshots[time_col].isna().any():
        raise ValueError("anchor_time contains nulls")
    
def build_latest_snapshot(df: pd.DataFrame, cfg: SnapshotConfig) -> pd.DataFrame:
    template = cfg.template

    if template == "transactions":
        df = df.dropna(subset=["event_time", "customer_id", "transaction_id", "amount"]).copy()
        t = df["event_time"].max().normalize()

        past_start = t - pd.Timedelta(days=cfg.history_days)
        past = df[(df["event_time"] > past_start) & (df["event_time"] <= t)].copy()

        feat = build_transaction_features(past, t)
        if feat.empty:
            return pd.DataFrame()

        if cfg.min_events_in_history > 1 and "frequency_tx" in feat.columns:
            feat = feat[feat["frequency_tx"] >= cfg.min_events_in_history].copy()

        if cfg.min_lifetime_days > 0 and "customer_lifetime_days" in feat.columns:
            feat = feat[feat["customer_lifetime_days"] >= cfg.min_lifetime_days].copy()

        if cfg.max_recency_days is not None and "recency_days" in feat.columns:
            feat = feat[feat["recency_days"] <= cfg.max_recency_days].copy()

        feat["anchor_time"] = t
        feat["value_proxy"] = feat["monetary"].fillna(0.0)
        feat = feat.rename(columns={"customer_id": "entity_id"})
        return feat

    raise NotImplementedError("build_latest_snapshot currently implemented only for transactions")