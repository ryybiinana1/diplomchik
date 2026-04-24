from __future__ import annotations

import streamlit as st

from ui.steps import (
    STEP_1_DATA,
    STEP_2_ANALYTICS,
    STEP_3_TRAIN,
    STEP_4_QUALITY,
    STEP_5_FORECAST,
    STEP_NAV_SHORT,
)


def render_progress_header(current_step: str, state) -> None:
    statuses = {
        STEP_1_DATA: bool(state.working_file_bytes),
        STEP_2_ANALYTICS: bool(state.inspect),
        STEP_3_TRAIN: bool(state.job_id),
        STEP_4_QUALITY: bool(state.job_result),
        STEP_5_FORECAST: bool(state.score_result) or bool(state.score_id),
    }
    labels = []
    for step in [STEP_1_DATA, STEP_2_ANALYTICS, STEP_3_TRAIN, STEP_4_QUALITY, STEP_5_FORECAST]:
        label = STEP_NAV_SHORT.get(step, step)
        if step == current_step:
            labels.append(f"**{label}**")
        elif statuses.get(step):
            labels.append(f"{label} · готово")
        else:
            labels.append(f"{label} · далее")

    st.caption("Этапы работы: " + "  |  ".join(labels))
