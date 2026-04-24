from __future__ import annotations

import streamlit as st

from ui.navigation import set_current_step
from ui.steps import STEP_NAV_SHORT


def page_nav(
    prev_step: str | None,
    next_step: str | None,
    *,
    next_disabled_reason: str | None = None,
) -> None:
    """Единый блок перехода между шагами."""
    st.markdown("---")
    with st.container(border=True):
        st.caption("Переход между разделами")

        left, right = st.columns(2, gap="medium")

        with left:
            if prev_step:
                label = STEP_NAV_SHORT.get(prev_step, prev_step)
                if st.button(
                    f"← {label}",
                    key=f"nav_prev_{prev_step}_{next_step}",
                    type="secondary",
                    use_container_width=True,
                ):
                    set_current_step(prev_step)
                    st.rerun()
            else:
                st.caption("Это первый шаг сценария.")

        with right:
            if next_step:
                label = STEP_NAV_SHORT.get(next_step, next_step)
                disabled = bool(next_disabled_reason)
                if st.button(
                    f"{label} →",
                    key=f"nav_next_{prev_step}_{next_step}",
                    type="secondary",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    set_current_step(next_step)
                    st.rerun()
                if disabled:
                    st.caption(next_disabled_reason)
            else:
                st.caption("Это финальный шаг сценария.")
