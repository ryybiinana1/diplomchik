# ui/components/charts.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def monthly_counts(df: pd.DataFrame, time_col: str, value_col: str | None = None, title: str = "") -> None:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[time_col])
    d["month"] = d[time_col].dt.to_period("M").dt.to_timestamp()

    if value_col is None:
        agg = d.groupby("month").size().rename("count").reset_index()
        st.subheader(title)
        st.bar_chart(agg.set_index("month")["count"])
    else:
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        agg = d.groupby("month")[value_col].sum(min_count=1).rename("sum").reset_index()
        st.subheader(title)
        st.bar_chart(agg.set_index("month")["sum"])


def hist_per_entity(df: pd.DataFrame, entity_col: str, title: str = "", max_bins: int = 50) -> None:
    per = df.groupby(entity_col).size()
    st.subheader(title)
    st.bar_chart(per.value_counts().sort_index().head(max_bins))
