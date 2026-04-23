from __future__ import annotations

import streamlit as st


def set_current_step(step: str, *, scroll: bool = True) -> None:
    """
    Переключение активного шага.
    Увеличивает «поколение» сайдбар-радио, чтобы виджет перемонтировался: иначе Streamlit
    может подставить устаревший выбор с клиента и откатить шаг после st.rerun().
    """
    st.session_state["current_step"] = step
    st.session_state["_nav_widget_gen"] = int(st.session_state.get("_nav_widget_gen", 0)) + 1
    if scroll:
        st.session_state["_scroll_top_after_rerun"] = True
