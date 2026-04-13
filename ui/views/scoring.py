# ui/pages/scoring.py
from __future__ import annotations

import streamlit as st

from ui.components.nav import page_nav
from ui.api_client import ApiClient
from ui.state import get_state


def page():
    st.title("Модель и прогноз")
    st.caption("Шаг 4: выбираем обученную модель, загружаем новые данные и получаем список клиентов с риском.")

    api = ApiClient.from_env()
    state = get_state()

    st.subheader("Выбор модели")
    try:
        models = api.list_models().get("models", [])
    except Exception as e:
        st.error(f"Не удалось получить список моделей: {e}")
        return

    if not models:
        st.info("Пока нет обученных моделей. Сначала обучите модель на странице «Обучение».")
        return

    model_titles = [f"{m.get('job_id')} | {m.get('template')} | {m.get('test_metrics_cal', {})}" for m in models]
    idx = st.selectbox("Модель (bundle)", range(len(models)), format_func=lambda i: model_titles[i])
    chosen = models[idx]
    bundle_dir = chosen["bundle_dir"]

    st.code(bundle_dir)

    st.subheader("Загрузка данных для прогноза")
    up = st.file_uploader("CSV с новыми данными (формат как при обучении)", type=["csv"], key="score_uploader")
    if up is not None:
        state.score_file_name = up.name
        state.score_file_bytes = up.getvalue()

    if not state.score_file_bytes:
        st.info("Загрузите CSV для скоринга.")
        return

    if st.button("Сделать прогноз", type="primary"):
        with st.spinner("Считаем риск..."):
            try:
                res = api.score(state.score_file_bytes, bundle_dir=bundle_dir)
                state.score_result = res
                st.success(f"Скоринг готов: {res.get('n_scored')} клиентов")
            except Exception as e:
                st.error(f"Ошибка скоринга: {e}")
                return

    if not state.score_result:
        return

    st.subheader("Результаты")
    res = state.score_result
    st.write(f"Скорили: {res.get('n_scored')}")
    st.write("Скачать CSV результата:")
    st.write(f"{api.base_url}{res.get('download_url')}")

    preview = res.get("preview", [])
    st.dataframe(preview, use_container_width=True)

    st.info(
        "Как интерпретировать:\n"
        "- p_calibrated ближе к 1 → выше риск\n"
        "- risk_segment: low/medium/high/critical\n"
        "- reason_1..3 — простые причины (rule-based), чтобы клиент понял смысл"
    )
    page_nav("Обучение", None)
