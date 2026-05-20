from __future__ import annotations

import streamlit as st

from ui.components.scroll import scroll_to_top
from ui.components.theme import inject_global_theme
from ui.components.progress_header import render_progress_header
from ui.state import init_state
from ui.state import get_state
from ui.steps import (
    STEP_1_DATA,
    STEP_2_ANALYTICS,
    STEP_3_TRAIN,
    STEP_4_QUALITY,
    STEP_5_FORECAST,
    STEP_LABELS,
    STEPS_ORDER,
)
from ui.views.model_quality import page as model_quality_page
from ui.views.scoring import page as scoring_page
from ui.views.step1_data_mapping import page as step1_page
from ui.views.step2_analytics import page as step2_page
from ui.views.training import page as training_page


_LEGACY_STEPS = {
    "Данные": STEP_1_DATA,
    "Сопоставление": STEP_2_ANALYTICS,
    "Обучение": STEP_3_TRAIN,
    "Качество модели": STEP_4_QUALITY,
    "Прогноз": STEP_5_FORECAST,
}
st.set_page_config(
    page_title="Churn / Retention Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_theme()
init_state()

st.session_state.pop("nav_radio", None)

if "current_step" not in st.session_state:
    st.session_state["current_step"] = STEP_1_DATA

legacy = st.session_state.get("current_step")
if legacy in _LEGACY_STEPS:
    st.session_state["current_step"] = _LEGACY_STEPS[legacy]
elif legacy not in STEPS_ORDER:
    st.session_state["current_step"] = STEP_1_DATA

_should_scroll_top = st.session_state.pop("_scroll_top_after_rerun", False)

st.sidebar.markdown("## Churn / Retention Studio")
st.sidebar.caption("Разделы")

_cur = st.session_state["current_step"]
if _cur not in STEPS_ORDER:
    _cur = STEP_1_DATA
    st.session_state["current_step"] = _cur

_nav_gen = int(st.session_state.get("_nav_widget_gen", 0))
_nav_choice = st.sidebar.radio(
    "Раздел приложения",
    options=STEPS_ORDER,
    index=STEPS_ORDER.index(_cur),
    format_func=lambda s: STEP_LABELS[s],
    key=f"sidebar_nav_{_nav_gen}",
    label_visibility="collapsed",
)

if _nav_choice != _cur:
    st.session_state["current_step"] = _nav_choice
    st.session_state["_scroll_top_after_rerun"] = True
    st.rerun()

selected = st.session_state["current_step"]
if st.session_state.get("_last_rendered_step") != selected:
    _should_scroll_top = True
    st.session_state["_last_rendered_step"] = selected

render_progress_header(selected, get_state())

if selected == STEPS_ORDER[0]:
    step1_page()
elif selected == STEPS_ORDER[1]:
    step2_page()
elif selected == STEPS_ORDER[2]:
    training_page()
elif selected == STEPS_ORDER[3]:
    model_quality_page()
else:
    scoring_page()

if _should_scroll_top:
    scroll_to_top()