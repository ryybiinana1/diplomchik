from __future__ import annotations

import streamlit as st

from ui.components.scroll import scroll_to_top
from ui.state import init_state
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

st.markdown(
    """
    <style>
        /* Корень rem: немного крупнее для комфортного чтения с ПК */
        html {
            font-size: 21px;
        }

        /* Вся ширина окна под приложение */
        .stApp {
            max-width: 100vw !important;
        }

        [data-testid="stAppViewContainer"] {
            max-width: none !important;
            width: 100% !important;
        }

        /* Сайдбар: фиксированная комфортная ширина (не сжимать на ноутбуке) */
        section[data-testid="stSidebar"] {
            width: 20.5rem !important;
            min-width: 19rem !important;
            max-width: 22rem !important;
            flex-shrink: 0 !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 100% !important;
        }

        /* У Streamlit у «широкого» режима всё ещё есть внутренний max-width — убираем по цепочке */
        [data-testid="stMain"] {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            max-width: none !important;
            width: 100% !important;
        }

        [data-testid="stMain"] > div,
        [data-testid="stMain"] section.main,
        [data-testid="stMain"] section.main > div {
            max-width: none !important;
            width: 100% !important;
        }

        .main .block-container,
        section.main .block-container,
        [data-testid="stMain"] .block-container {
            max-width: none !important;
            width: 100% !important;
            padding-top: 1.45rem !important;
            padding-bottom: 2.8rem !important;
            padding-left: clamp(1.1rem, 2.8vw, 2.8rem) !important;
            padding-right: clamp(1.1rem, 2.8vw, 2.8rem) !important;
        }

        /* Основная колонка: базовый кегль и межстрочный интервал */
        section[data-testid="stMain"] {
            font-size: 1.18rem !important;
            line-height: 1.68 !important;
        }

        /* Обычный текст: абзацы, списки, ячейки в markdown-таблицах */
        section[data-testid="stMain"] .stMarkdown,
        section[data-testid="stMain"] .stMarkdown p,
        section[data-testid="stMain"] .stMarkdown li,
        section[data-testid="stMain"] .stMarkdown td,
        section[data-testid="stMain"] .stMarkdown th,
        section[data-testid="stMain"] [data-testid="stText"] {
            font-size: 1.16rem !important;
            line-height: 1.68 !important;
            max-width: none !important;
        }

        section[data-testid="stMain"] .stMarkdown p,
        section[data-testid="stMain"] .stMarkdown li {
            max-width: 72ch !important;
            margin-bottom: 0.55em;
        }

        section[data-testid="stMain"] .stMarkdown ul,
        section[data-testid="stMain"] .stMarkdown ol,
        section[data-testid="stMain"] [data-testid="stText"] {
            max-width: 76ch !important;
        }

        /* st.write / success / info / warning / error */
        section[data-testid="stMain"] [data-testid="stAlert"] p,
        section[data-testid="stMain"] [data-testid="stAlert"] span {
            max-width: 76ch !important;
            font-size: 1.12rem !important;
            line-height: 1.56 !important;
        }

        /* Раскрывающиеся блоки: заголовок и содержимое */
        section[data-testid="stMain"] [data-testid="stExpander"] details summary {
            font-size: 1.14rem !important;
            font-weight: 600 !important;
        }
        section[data-testid="stMain"] [data-testid="stExpander"] .stMarkdown p {
            font-size: 1.12rem !important;
        }

        /* Поля ввода, выпадающие списки, мультиселект (Base Web) */
        section[data-testid="stMain"] [data-baseweb="input"] input,
        section[data-testid="stMain"] [data-baseweb="textarea"] textarea,
        section[data-testid="stMain"] [data-baseweb="select"] > div,
        section[data-testid="stMain"] [data-baseweb="popover"] {
            font-size: 1.1rem !important;
        }

        /* Горизонтальные radio / чекбоксы в основной области */
        section[data-testid="stMain"] div[role="radiogroup"] label,
        section[data-testid="stMain"] div[role="group"] label {
            font-size: 1.1rem !important;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 0.9rem !important;
            font-size: 1.1rem !important;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            font-size: 1.08rem !important;
            line-height: 1.5 !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 1.52rem !important;
            font-weight: 700 !important;
            line-height: 1.28 !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            margin-bottom: 0.6rem !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="radio"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            gap: 0.35rem !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.35rem !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            font-size: 1.12rem !important;
            font-weight: 500 !important;
            padding: 0.55rem 0.5rem !important;
            border: none !important;
            border-radius: 10px !important;
            margin: 0 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {
            background: transparent !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
            font-weight: 700 !important;
            background: rgba(49, 51, 63, 0.07) !important;
        }

        section[data-testid="stMain"] h1 {
            font-size: clamp(2.25rem, 3vw, 3rem) !important;
            font-weight: 700 !important;
            text-align: left !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em;
            line-height: 1.12 !important;
            max-width: 24ch !important;
        }

        section[data-testid="stMain"] [data-testid="stCaptionContainer"] {
            font-size: 1.2rem !important;
            line-height: 1.58 !important;
            color: rgba(49, 51, 63, 0.82) !important;
            max-width: 76ch !important;
        }

        section[data-testid="stMain"] h2 {
            font-size: 1.78rem !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.45rem !important;
            line-height: 1.25 !important;
            max-width: 28ch !important;
        }

        section[data-testid="stMain"] h3 {
            font-size: 1.48rem !important;
            font-weight: 600 !important;
            margin-top: 1.15rem !important;
            margin-bottom: 0.35rem !important;
            line-height: 1.3 !important;
            max-width: 34ch !important;
        }

        section[data-testid="stMain"] h4 {
            font-size: 1.24rem !important;
            line-height: 1.4 !important;
            margin-top: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }

        section[data-testid="stMain"] [data-testid="stTabs"] button {
            font-size: 1.04rem !important;
            padding-top: 0.55rem !important;
            padding-bottom: 0.55rem !important;
        }

        section[data-testid="stMain"] .stButton > button {
            min-height: 54px !important;
            font-size: 1.14rem !important;
            border-radius: 12px !important;
        }

        section[data-testid="stMain"] .stMetric {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 18px 18px;
            background: rgba(255, 255, 255, 0.95);
        }

        section[data-testid="stMain"] div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            font-size: 1.08rem !important;
        }

        section[data-testid="stMain"] div[data-testid="metric-container"] label {
            font-size: 1.1rem !important;
        }

        section[data-testid="stMain"] div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
        }

        /* Подписи к виджетам (selectbox, file_uploader, …) */
        section[data-testid="stMain"] label[data-testid="stWidgetLabel"] p,
        section[data-testid="stMain"] label[data-testid="stWidgetLabel"] span {
            font-size: 1.12rem !important;
            line-height: 1.45 !important;
        }

        /* Код и моноширинный текст (пути и т.п.) */
        section[data-testid="stMain"] pre,
        section[data-testid="stMain"] code {
            font-size: 1.03rem !important;
            line-height: 1.45 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

# Старые сессии: ключ nav_radio больше не используется (конфликт с клиентским состоянием виджета).
st.session_state.pop("nav_radio", None)

if "current_step" not in st.session_state:
    st.session_state["current_step"] = STEP_1_DATA

legacy = st.session_state.get("current_step")
if legacy in _LEGACY_STEPS:
    st.session_state["current_step"] = _LEGACY_STEPS[legacy]
elif legacy not in STEPS_ORDER:
    st.session_state["current_step"] = STEP_1_DATA

if st.session_state.pop("_scroll_top_after_rerun", False):
    scroll_to_top()

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
