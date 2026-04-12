from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _slope_from_series(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def build_transaction_features(past: pd.DataFrame, anchor_time: pd.Timestamp) -> pd.DataFrame:
    """
    Build transaction-level customer features from the past window.
    Expected canonical columns:
      customer_id, transaction_id, event_time, amount
    Optional:
      item_id, quantity, country, unit_price, is_cancellation
    """

    df = past.copy()

    # normalize dtypes
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["customer_id"] = df["customer_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    if "country" in df.columns:
        df["country"] = df["country"].astype(str)
    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(str)

    if "is_cancellation" not in df.columns:
        df["is_cancellation"] = df["transaction_id"].astype(str).str.startswith("C")

    purchase_df = df[~df["is_cancellation"]].copy()
    cancel_df = df[df["is_cancellation"]].copy()

    purchase_df = purchase_df.dropna(subset=["customer_id", "transaction_id", "event_time", "amount"]).copy()
    purchase_df = purchase_df[purchase_df["amount"] > 0].copy()

    if purchase_df.empty:
        return pd.DataFrame()

    # invoice-level
    inv = (
        purchase_df.groupby(["customer_id", "transaction_id"], as_index=False)
        .agg(
            inv_time=("event_time", "max"),
            inv_amount=("amount", "sum"),
            purchase_day=("event_time", lambda x: x.max().normalize()),
        )
    )

    # ---------- Base RFM+ ----------
    last_purchase = inv.groupby("customer_id")["inv_time"].max()
    first_purchase = inv.groupby("customer_id")["inv_time"].min()
    frequency_tx = inv.groupby("customer_id")["transaction_id"].nunique()
    monetary = inv.groupby("customer_id")["inv_amount"].sum()

    recency_days = (anchor_time - last_purchase).dt.days
    days_since_first_purchase = (anchor_time - first_purchase).dt.days
    customer_lifetime_days = (last_purchase - first_purchase).dt.days

    avg_basket_value = inv.groupby("customer_id")["inv_amount"].mean()
    median_basket_value = inv.groupby("customer_id")["inv_amount"].median()
    max_basket_value = inv.groupby("customer_id")["inv_amount"].max()
    min_basket_value = inv.groupby("customer_id")["inv_amount"].min()
    basket_std = inv.groupby("customer_id")["inv_amount"].std().fillna(0.0)
    number_of_purchase_days = inv.groupby("customer_id")["purchase_day"].nunique()

    # ---------- Interpurchase ----------
    inv_sorted = inv.sort_values(["customer_id", "inv_time"]).copy()
    gap_rows = []
    for cid, sub in inv_sorted.groupby("customer_id"):
        dts = sub["inv_time"].values
        if len(dts) < 2:
            gaps = np.array([], dtype=float)
        else:
            gaps = np.diff(dts).astype("timedelta64[D]").astype(float)

        if len(gaps) == 0:
            gap_mean = np.nan
            gap_median = np.nan
            gap_std = np.nan
            gap_cv = np.nan
        else:
            gap_mean = float(np.mean(gaps))
            gap_median = float(np.median(gaps))
            gap_std = float(np.std(gaps))
            gap_cv = float(gap_std / (gap_mean + EPS))

        gap_rows.append({
            "customer_id": cid,
            "interpurchase_mean_days": gap_mean,
            "interpurchase_median_days": gap_median,
            "interpurchase_std_days": gap_std,
            "interpurchase_cv_days": gap_cv,
        })
    gap_df = pd.DataFrame(gap_rows).set_index("customer_id")

    # ---------- Rolling windows ----------
    def line_amount_sum(days: int):
        mask = (purchase_df["event_time"] > anchor_time - pd.Timedelta(days=days)) & (purchase_df["event_time"] <= anchor_time)
        return purchase_df.loc[mask].groupby("customer_id")["amount"].sum()

    def tx_count(days: int):
        mask = (inv["inv_time"] > anchor_time - pd.Timedelta(days=days)) & (inv["inv_time"] <= anchor_time)
        return inv.loc[mask].groupby("customer_id")["transaction_id"].nunique()

    spend_last7 = line_amount_sum(7)
    spend_last30 = line_amount_sum(30)
    spend_last60 = line_amount_sum(60)
    spend_last90 = line_amount_sum(90)

    tx_last7 = tx_count(7)
    tx_last30 = tx_count(30)
    tx_last60 = tx_count(60)
    tx_last90 = tx_count(90)

    prev30_mask = (purchase_df["event_time"] > anchor_time - pd.Timedelta(days=60)) & (
        purchase_df["event_time"] <= anchor_time - pd.Timedelta(days=30)
    )
    spend_prev30 = purchase_df.loc[prev30_mask].groupby("customer_id")["amount"].sum()

    prev60_mask = (purchase_df["event_time"] > anchor_time - pd.Timedelta(days=120)) & (
        purchase_df["event_time"] <= anchor_time - pd.Timedelta(days=60)
    )
    spend_prev60 = purchase_df.loc[prev60_mask].groupby("customer_id")["amount"].sum()

    ratio_last30_prev30 = (spend_last30 / (spend_prev30 + EPS)).replace([np.inf, -np.inf], np.nan)
    ratio_last60_prev60 = (spend_last60 / (spend_prev60 + EPS)).replace([np.inf, -np.inf], np.nan)

    # ---------- Trend slopes ----------
    trend_rows = []
    for cid, sub in purchase_df.groupby("customer_id"):
        amounts = []
        txs = []
        for i in range(6, 0, -1):  # last 6 windows of 30d
            start = anchor_time - pd.Timedelta(days=i * 30)
            end = anchor_time - pd.Timedelta(days=(i - 1) * 30)
            win = sub[(sub["event_time"] > start) & (sub["event_time"] <= end)]
            amounts.append(float(win["amount"].sum()))
            txs.append(float(win["transaction_id"].nunique()))
        trend_rows.append({
            "customer_id": cid,
            "spend_slope_6x30d": _slope_from_series(np.array(amounts)),
            "tx_slope_6x30d": _slope_from_series(np.array(txs)),
        })
    trend_df = pd.DataFrame(trend_rows).set_index("customer_id")

    # ---------- Stability / regularity ----------
    inv["week_key"] = inv["inv_time"].dt.isocalendar().year.astype(int) * 100 + inv["inv_time"].dt.isocalendar().week.astype(int)
    inv["month_key"] = inv["inv_time"].dt.year.astype(int) * 100 + inv["inv_time"].dt.month.astype(int)

    active_weeks = inv.groupby("customer_id")["week_key"].nunique()
    active_months = inv.groupby("customer_id")["month_key"].nunique()

    history_days = max(1, int((anchor_time - purchase_df["event_time"].min()).days))
    active_weeks_ratio = active_weeks / max(1, int(np.ceil(history_days / 7)))
    active_months_ratio = active_months / max(1, int(np.ceil(history_days / 30)))

    repeat_purchase_rate = frequency_tx / (number_of_purchase_days + EPS)
    regularity_score = 1.0 / (gap_df["interpurchase_cv_days"] + 1.0)

    # ---------- Diversity ----------
    diversity_df = None
    if "item_id" in purchase_df.columns:
        div_rows = []
        for cid, sub in purchase_df.groupby("customer_id"):
            item_counts = sub["item_id"].astype(str).value_counts()
            unique_items_count = int(item_counts.shape[0])
            unique_item_ratio = float(unique_items_count / (len(sub) + EPS))
            item_entropy = _entropy_from_counts(item_counts.values)
            top1_item_concentration = float(item_counts.iloc[0] / item_counts.sum()) if len(item_counts) else 0.0
            top3_item_concentration = float(item_counts.iloc[:3].sum() / item_counts.sum()) if len(item_counts) else 0.0
            repeated_items_share = float(item_counts[item_counts > 1].sum() / (item_counts.sum() + EPS))
            new_items_share = float(item_counts[item_counts == 1].sum() / (item_counts.sum() + EPS))
            div_rows.append({
                "customer_id": cid,
                "unique_items_count": unique_items_count,
                "unique_item_ratio": unique_item_ratio,
                "item_entropy": item_entropy,
                "top1_item_concentration": top1_item_concentration,
                "top3_item_concentration": top3_item_concentration,
                "repeated_items_share": repeated_items_share,
                "new_items_share": new_items_share,
            })
        diversity_df = pd.DataFrame(div_rows).set_index("customer_id")

    # ---------- Price / value ----------
    price_df = None
    if "unit_price" in purchase_df.columns:
        q75_price = purchase_df["unit_price"].quantile(0.75)
        q25_price = purchase_df["unit_price"].quantile(0.25)

        price_rows = []
        for cid, sub in purchase_df.groupby("customer_id"):
            unit_price = pd.to_numeric(sub["unit_price"], errors="coerce").dropna()
            avg_item_price = float(unit_price.mean()) if len(unit_price) else np.nan
            median_item_price = float(unit_price.median()) if len(unit_price) else np.nan
            value_volatility = float(unit_price.std()) if len(unit_price) > 1 else 0.0
            expensive_purchase_ratio = float((unit_price > q75_price).mean()) if len(unit_price) else 0.0
            low_price_share = float((unit_price <= q25_price).mean()) if len(unit_price) else 0.0

            price_rows.append({
                "customer_id": cid,
                "avg_item_price": avg_item_price,
                "median_item_price": median_item_price,
                "value_volatility": value_volatility,
                "expensive_purchase_ratio": expensive_purchase_ratio,
                "low_price_share": low_price_share,
            })
        price_df = pd.DataFrame(price_rows).set_index("customer_id")

    # ---------- Country ----------
    country_df = None
    if "country" in purchase_df.columns:
        country_mode = (
            purchase_df.groupby("customer_id")["country"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "UNK")
        )
        country_counts = purchase_df["country"].astype(str).value_counts(normalize=True)

        country_df = pd.DataFrame({
            "is_uk": (country_mode == "United Kingdom").astype(int),
            "country_freq": country_mode.map(country_counts).fillna(0.0),
        })

    # ---------- Seasonality ----------
    seasonality_df = pd.DataFrame({
        "last_purchase_month": last_purchase.dt.month.astype(int),
        "last_purchase_quarter": last_purchase.dt.quarter.astype(int),
        "last_purchase_dayofweek": last_purchase.dt.dayofweek.astype(int),
        "holiday_season_flag": last_purchase.dt.month.isin([11, 12]).astype(int),
    })

    christmas_current_year = pd.to_datetime(last_purchase.dt.year.astype(str) + "-12-25", utc=True)
    seasonality_df["days_to_christmas"] = (christmas_current_year - last_purchase).dt.days

    # ---------- Returns / cancellations ----------
    if not cancel_df.empty:
        cancel_df["abs_amount"] = pd.to_numeric(cancel_df["amount"], errors="coerce").abs()

        cancel_count = cancel_df.groupby("customer_id")["transaction_id"].nunique()
        cancel_amount = cancel_df.groupby("customer_id")["abs_amount"].sum()

        purchase_count = purchase_df.groupby("customer_id")["transaction_id"].nunique()
        purchase_amount = purchase_df.groupby("customer_id")["amount"].sum()

        return_df = pd.DataFrame({
            "had_return": (cancel_count > 0).astype(int),
            "return_count": cancel_count,
            "return_ratio": cancel_count / (purchase_count + EPS),
            "canceled_amount_ratio": cancel_amount / (purchase_amount + EPS),
        }).fillna(0.0)
    else:
        return_df = pd.DataFrame(index=purchase_df["customer_id"].unique())
        return_df["had_return"] = 0
        return_df["return_count"] = 0.0
        return_df["return_ratio"] = 0.0
        return_df["canceled_amount_ratio"] = 0.0

    # ---------- Final merge ----------
    base = pd.DataFrame({
        "recency_days": recency_days,
        "frequency_tx": frequency_tx,
        "monetary": monetary,
        "avg_basket_value": avg_basket_value,
        "median_basket_value": median_basket_value,
        "max_basket_value": max_basket_value,
        "min_basket_value": min_basket_value,
        "basket_std": basket_std,
        "number_of_purchase_days": number_of_purchase_days,
        "days_since_first_purchase": days_since_first_purchase,
        "customer_lifetime_days": customer_lifetime_days,
        "spend_last7": spend_last7,
        "spend_last30": spend_last30,
        "spend_last60": spend_last60,
        "spend_last90": spend_last90,
        "tx_last7": tx_last7,
        "tx_last30": tx_last30,
        "tx_last60": tx_last60,
        "tx_last90": tx_last90,
        "spend_prev30": spend_prev30,
        "spend_prev60": spend_prev60,
        "ratio_last30_prev30": ratio_last30_prev30,
        "ratio_last60_prev60": ratio_last60_prev60,
        "active_weeks": active_weeks,
        "active_months": active_months,
        "active_weeks_ratio": active_weeks_ratio,
        "active_months_ratio": active_months_ratio,
        "repeat_purchase_rate": repeat_purchase_rate,
        "regularity_score": regularity_score,
    })

    base = base.join(gap_df, how="left")
    base = base.join(trend_df, how="left")
    base = base.join(seasonality_df, how="left")
    base = base.join(return_df, how="left")

    if diversity_df is not None:
        base = base.join(diversity_df, how="left")
    if price_df is not None:
        base = base.join(price_df, how="left")
    if country_df is not None:
        base = base.join(country_df, how="left")

    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base = base.reset_index().rename(columns={"index": "customer_id"})
    return base