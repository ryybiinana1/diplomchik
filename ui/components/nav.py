from __future__ import annotations

import streamlit as st


def page_nav(prev_step: str | None, next_step: str | None) -> None:
    st.markdown("---")
    left, middle, right = st.columns([1, 2, 1])

    with left:
        if prev_step:
            if st.button(f"← {prev_step}", use_container_width=True):
                st.session_state["current_step"] = prev_step
                st.rerun()

    with right:
        if next_step:
            if st.button(f"{next_step} →", use_container_width=True):
                st.session_state["current_step"] = next_step
                st.rerun()