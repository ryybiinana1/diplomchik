"""Периодический rerun страницы Streamlit (опционально через streamlit-autorefresh)."""

from __future__ import annotations

import streamlit as st


def schedule_autorefresh(interval_ms: int, key: str) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        st.caption(
            "Автообновление отключено: в окружении нет пакета `streamlit-autorefresh`. "
            "Установите зависимости UI или пересоберите образ: `docker compose build --no-cache ui`."
        )
        return
    st_autorefresh(interval=interval_ms, limit=None, key=key)
