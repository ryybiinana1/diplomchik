from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st


@dataclass
class AppState:
    file_name: str | None = None
    file_bytes: bytes | None = None
    working_file_bytes: bytes | None = None
    did_dedupe: bool = False
    did_dropna: bool = False
    clean_notice_dedupe: str | None = None
    clean_notice_dropna: str | None = None
    clean_notice_restore: str | None = None

    template: str = "transactions"
    inspect: dict | None = None
    contracts: dict | None = None

    mapping: dict[str, Any] = field(default_factory=dict)
    extra_feature_cols: list[str] = field(default_factory=list)
    extra_feature_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    job_id: str | None = None
    job_status: dict | None = None
    job_result: dict | None = None

    models_cache: dict | None = None

    score_file_name: str | None = None
    score_file_bytes: bytes | None = None
    score_column_mapping: dict[str, Any] = field(default_factory=dict)
    score_schema_check: dict | None = None
    score_id: str | None = None
    score_status: dict | None = None
    score_result: dict | None = None


def init_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()


def get_state() -> AppState:
    init_state()
    return st.session_state["app_state"]


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    # utf-8-sig: как в Excel UTF-8 — убирает BOM из имени первой колонки (иначе API и UI расходятся).
    return pd.read_csv(BytesIO(file_bytes), low_memory=False, encoding="utf-8-sig")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")