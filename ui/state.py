from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import pandas as pd
import streamlit as st


@dataclass
class AppState:
    file_name: str | None = None
    file_bytes: bytes | None = None
    template: str = "transactions"

    inspect: dict | None = None
    contracts: dict | None = None

    mapping: dict = field(default_factory=dict)
    extra_feature_cols: list[str] = field(default_factory=list)

    job_id: str | None = None
    job_status: dict | None = None
    job_result: dict | None = None

    models_cache: dict | None = None
    score_result: dict | None = None


def init_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()


def get_state() -> AppState:
    init_state()
    return st.session_state["app_state"]


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))