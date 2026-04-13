from __future__ import annotations

import streamlit as st

from ui.components.nav import page_nav
from ui.api_client import ApiClient
from ui.state import get_state


def page():
    st.title("Обучение модели")
    st.caption("Шаг 3: выбираем режим и обучаем. Всё объясняется простым языком.")

    api = ApiClient.from_env()
    state = get_state()

    if not state.file_bytes or not state.mapping:
        st.warning("Сначала пройдите шаги «Готовность данных» и «Сопоставление колонок».")
        page_nav("Сопоставление", "Качество модели")
        return

    mode = st.radio(
        "Режим обучения",
        ["quick", "deep"],
        format_func=lambda x: "Быстрый запуск" if x == "quick" else "Максимальное качество",
        horizontal=True,
    )

    st.subheader("Настройки")

    if mode == "quick":
        horizon = st.selectbox("Горизонт прогноза (дней)", [30, 60, 90], index=0)
        history = st.selectbox("Окно истории (дней)", [90, 180, 365], index=1)
        model_kind = st.selectbox("Модель", ["lightgbm", "logreg", "random_forest"], index=0)
        enable_shap = st.checkbox("Добавить объяснения (SHAP) в отчёт", value=False)

        params = {
            "horizon_days": int(horizon),
            "history_days": int(history),
            "step_days": 30,
            "min_events_in_history": 1,
            "model_kind": model_kind,
            "calibration": "sigmoid",
            "enable_shap": bool(enable_shap),
            "extra_feature_cols": state.extra_feature_cols or [],
        }

        st.caption("Что будет сделано: 1 конфигурация, быстрое обучение, короткий отчёт.")

    else:
        horizons = st.multiselect("Горизонты (дней)", [30, 60, 90], default=[30, 60, 90])
        histories = st.multiselect("Окна истории (дней)", [90, 180, 365], default=[180, 365])
        steps = st.multiselect("Шаг анкеров (дней)", [7, 14, 30], default=[30])
        models = st.multiselect(
            "Модели",
            ["lightgbm", "logreg", "random_forest", "sklearn_gbdt", "catboost"],
            default=["lightgbm", "logreg"],
        )
        selection_metric = st.selectbox(
            "Метрика выбора лучшей конфигурации",
            ["pr_auc", "roc_auc", "brier", "base_max_profit"],
            index=0,
        )
        enable_shap = st.checkbox("Добавить объяснения (SHAP) в отчёт", value=True)

        params = {
            "horizon_days_grid": horizons,
            "history_days_grid": histories,
            "step_days_grid": steps,
            "model_kind_grid": models,
            "selection_metric": selection_metric,
            "calibration_grid": ["sigmoid", "isotonic"],
            "min_events_in_history": 1,
            "enable_shap": bool(enable_shap),
            "extra_feature_cols": state.extra_feature_cols or [],
        }

        st.caption("Что будет сделано: перебор параметров, сравнение, выбор лучшего bundle, полный отчёт.")

    if st.button("Запустить обучение", type="primary"):
        with st.spinner("Создаём job и запускаем обучение..."):
            try:
                resp = api.create_job(
                    file_bytes=state.file_bytes,
                    template=state.template,
                    mapping=state.mapping,
                    params=params,
                )
                state.job_id = resp["job_id"]
                state.job_status = None
                state.job_result = None
                st.success(f"Job создан: {state.job_id}")
            except Exception as e:
                st.error(f"Не удалось создать job: {e}")
                page_nav("Сопоставление", "Качество модели")
                return

    if not state.job_id:
        st.info("Запустите обучение, чтобы увидеть статус.")
        page_nav("Сопоставление", "Качество модели")
        return

    st.subheader("Статус обучения")

    if st.button("Обновить статус"):
        try:
            state.job_status = api.job_status(state.job_id)
        except Exception as e:
            st.error(f"Не удалось получить статус: {e}")
            page_nav("Сопоставление", "Качество модели")
            return

    status = state.job_status
    if not status:
        status = api.job_status(state.job_id)
        state.job_status = status

    st.json(status)

    if status.get("status") == "done":
        st.success("Обучение завершено.")

        try:
            state.job_result = api.job_result(state.job_id)
        except Exception as e:
            st.error(f"Не удалось получить результат job: {e}")
            page_nav("Сопоставление", "Качество модели")
            return

        st.write("Скачать артефакты (zip):")
        st.write(f"{api.base_url}/jobs/{state.job_id}/download")

        st.write("Результат (кратко):")
        st.json(
            {
                k: state.job_result.get(k)
                for k in [
                    "mode",
                    "template",
                    "test_metrics_cal",
                    "business_metrics",
                    "suitability",
                ]
                if k in state.job_result
            }
        )

        if st.button("Перейти к качеству модели", use_container_width=True):
            st.session_state["current_step"] = "Качество модели"
            st.rerun()

    elif status.get("status") == "failed":
        st.error(f"Ошибка обучения: {status.get('error')}")
    else:
        st.warning("Обучение ещё идёт (или в очереди).")

    page_nav("Сопоставление", "Качество модели")