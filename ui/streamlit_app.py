from __future__ import annotations

import streamlit as st

from ui.state import init_state
from ui.views.data_readiness import page as readiness_page
from ui.views.schema_mapping import page as mapping_page
from ui.views.training import page as training_page
from ui.views.model_quality import page as model_quality_page
from ui.views.scoring import page as scoring_page

st.set_page_config(
    page_title="Churn/Retention Self-Serve",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 96vw !important;
            padding-top: 1.2rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            padding-bottom: 2rem !important;
        }

        html, body, [class*="css"] {
            font-size: 18px !important;
        }

        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.25rem !important; }

        div[data-testid="stMetric"] {
            background: #f7f7f8;
            border-radius: 14px;
            padding: 14px 16px;
            border: 1px solid #ececec;
        }

        .stDataFrame, .stTable {
            font-size: 16px !important;
        }

        button[kind="secondary"],
        button[kind="primary"] {
            border-radius: 12px !important;
            min-height: 44px !important;
            font-size: 16px !important;
        }

        .top-step-caption {
            color: #666;
            font-size: 15px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

if "current_step" not in st.session_state:
    st.session_state["current_step"] = "Данные"

st.title("Churn / Retention Self-Serve")
st.markdown(
    "<div class='top-step-caption'>Пошаговый сценарий: проверить данные → сопоставить колонки → обучить модель → оценить качество → получить прогноз</div>",
    unsafe_allow_html=True,
)

steps = ["Данные", "Сопоставление", "Обучение", "Качество модели", "Прогноз"]

step_labels = {
    "Данные": "1. Данные",
    "Сопоставление": "2. Сопоставление",
    "Обучение": "3. Обучение",
    "Качество модели": "4. Качество модели",
    "Прогноз": "5. Прогноз",
}

selected = st.radio(
    "Навигация",
    options=steps,
    format_func=lambda x: step_labels[x],
    index=steps.index(st.session_state["current_step"]),
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state["current_step"] = selected
st.divider()

if selected == "Данные":
    readiness_page()
elif selected == "Сопоставление":
    mapping_page()
elif selected == "Обучение":
    training_page()
elif selected == "Качество модели":
    model_quality_page()
else:
    scoring_page()