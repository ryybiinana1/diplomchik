"""Периодический rerun страницы Streamlit (опционально через streamlit-autorefresh)."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def schedule_autorefresh(interval_ms: int, key: str) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        seconds = max(int(interval_ms / 1000), 1)
        components.html(
            f"""
            <script>
              setTimeout(function () {{
                window.parent.location.reload();
              }}, {int(interval_ms)});
            </script>
            """,
            height=0,
            width=0,
        )
        st.caption(f"Страница обновится автоматически примерно через {seconds} сек.")
        return
    st_autorefresh(interval=interval_ms, limit=None, key=key)
