from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.layout import render_page_header, section_card
from ui.poll_rerun import schedule_autorefresh
from ui.components.nav import page_nav
from ui.state import get_state, read_csv_cached
from ui.steps import (
    METRIC_LABELS,
    MODEL_KIND_LABELS,
    STEP_2_ANALYTICS,
    STEP_4_QUALITY,
    TRAINING_STAGE_LABELS,
    format_model_kind,
)


def _stage_label(stage: str) -> str:
    return TRAINING_STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize())


def _row_count_from_state(state) -> int:
    if state.inspect:
        quality = state.inspect.get("quality_report", {}) or {}
        n_rows = quality.get("n_rows")
        if n_rows is not None:
            return int(n_rows)
    if state.working_file_bytes:
        try:
            return int(len(read_csv_cached(state.working_file_bytes)))
        except Exception:
            return 0
    return 0


def _recommended_mode(rows: int) -> str:
    if rows >= 50000:
        return "compare"
    if rows >= 5000:
        return "balanced"
    return "fast"


def _estimated_runtime_hint(experiments: int) -> str:
    if experiments <= 2:
        return "быстро"
    if experiments <= 8:
        return "умеренно"
    return "долго"


@st.cache_data(show_spinner=False)
def _download_job_report_cached(base_url: str, job_id: str) -> bytes:
    return ApiClient(base_url=base_url).download_job_report(job_id)


def page():
    if st.session_state.pop("_job_just_started", False):
        st.success("Обучение запущено. Ниже можно смотреть прогресс.")

    render_page_header(
        "Обучение модели",
        "Задайте бизнес-смысл прогноза: горизонт и глубину подбора. "
        "Технические ML-настройки спрятаны в расширенный блок.",
        eyebrow="Шаг 3",
    )

    api = ApiClient.from_env()
    state = get_state()

    if not state.working_file_bytes or not state.mapping:
        st.warning("Сначала выполните шаги «Данные и колонки» и «Анализ».")
        page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY)
        return

    row_count = _row_count_from_state(state)
    recommended_mode = _recommended_mode(row_count)

    preset = st.radio(
        "Режим запуска",
        options=["fast", "balanced", "compare"],
        format_func=lambda x: {
            "fast": "Быстро",
            "balanced": "Сбалансировано",
            "compare": "Тщательное сравнение",
        }.get(x, x),
        horizontal=True,
    )

    if recommended_mode != preset:
        st.info(
            {
                "fast": "По объёму данных системе больше подходит быстрый baseline.",
                "balanced": "По объёму данных системе больше подходит сбалансированный режим.",
                "compare": "По объёму данных можно запускать тщательное сравнение.",
            }[recommended_mode]
        )

    model_name = st.text_input(
        "Название модели",
        value="Моя модель",
        placeholder="Например: baseline_v1",
        help="Отображается в шаге «Прогноз» при выборе модели.",
    )

    st.markdown("### Настройка модели")
    enable_shap = True

    st.info(
        {
            "fast": "Один устойчивый baseline без лишних настроек.",
            "balanced": "Сбалансированный запуск для сравнения моделей.",
            "compare": "Полный режим для лидерборда и выбора лучшего варианта.",
        }[preset]
    )

    metric_key = "pr_auc"
    calibration_grid = ["sigmoid"]

    if preset == "fast":
        metric_key = st.selectbox(
            "Метрика выбора лучшей модели",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS.get(k, k),
            index=0,
        )
        horizons_grid = st.multiselect("Горизонты прогноза (дней)", [30, 60, 90], default=[30]) or [30]
        default_history = {30: 180, 60: 180, 90: 365}[int(horizons_grid[0])]
        history_grid = [default_history]
        steps_grid = [30]
        selected_models_display = [format_model_kind("lightgbm")]
        st.caption("Быстрый пресет: одна модель с перебором выбранных горизонтов/окон/шагов.")
        with st.expander("Тонкая настройка", expanded=False):
            history_grid = st.multiselect("Окна истории", [90, 180, 365], default=history_grid) or history_grid
            steps_grid = st.multiselect("Шаги точки отсчёта", [7, 14, 30], default=steps_grid) or steps_grid
        params = {
            "mode": "compare",
            "model_name": model_name.strip(),
            "horizon_days_grid": [int(x) for x in horizons_grid],
            "history_days_grid": [int(x) for x in history_grid],
            "step_days_grid": [int(x) for x in steps_grid],
            "min_events_in_history": 1,
            "model_kind_grid": ["lightgbm"],
            "selection_metric": metric_key,
            "calibration_grid": ["sigmoid"],
            "enable_shap": bool(enable_shap),
            "extra_feature_cols": state.extra_feature_cols or [],
            "extra_feature_config": state.extra_feature_config or {},
        }
        experiments = (
            len(params["horizon_days_grid"])
            * len(params["history_days_grid"])
            * len(params["step_days_grid"])
            * len(params["model_kind_grid"])
            * len(params["calibration_grid"])
        )
    elif preset == "balanced":
        horizons_grid = st.multiselect("Горизонты прогноза (дней)", [30, 60, 90], default=[30]) or [30]
        default_history = {30: 180, 60: 180, 90: 365}[int(horizons_grid[0])]
        metric_key = st.selectbox(
            "Метрика выбора лучшей модели",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS.get(k, k),
            index=0,
        )
        balanced_models = ["lightgbm", "catboost", "logreg", "random_forest", "sklearn_gbdt"]
        selected_models_display = [format_model_kind(m) for m in balanced_models]
        history_grid = [int(default_history)]
        steps_grid = [30]
        with st.expander("Тонкая настройка", expanded=False):
            history_grid = st.multiselect("Окна истории", [90, 180, 365], default=history_grid) or history_grid
            steps_grid = st.multiselect("Шаги точки отсчёта", [14, 30], default=steps_grid) or steps_grid
            calibration_grid = st.multiselect("Калибровка", ["sigmoid", "isotonic"], default=["sigmoid"]) or ["sigmoid"]
        params = {
            "mode": "compare",
            "model_name": model_name.strip(),
            "horizon_days_grid": [int(x) for x in horizons_grid],
            "history_days_grid": [int(x) for x in history_grid],
            "step_days_grid": [int(x) for x in steps_grid],
            "model_kind_grid": balanced_models,
            "selection_metric": metric_key,
            "calibration_grid": calibration_grid,
            "min_events_in_history": 1,
            "enable_shap": bool(enable_shap),
            "extra_feature_cols": state.extra_feature_cols or [],
            "extra_feature_config": state.extra_feature_config or {},
        }
        experiments = (
            len(params["horizon_days_grid"])
            * len(params["history_days_grid"])
            * len(params["step_days_grid"])
            * len(params["model_kind_grid"])
            * len(params["calibration_grid"])
        )
    else:
        st.caption("Выберите пространство экспериментов. Только этот режим показывает полный конструктор сравнения.")
        horizons_grid = st.multiselect("Горизонты прогноза (дней)", [30, 60, 90], default=[30, 60]) or [30]
        histories_grid = st.multiselect("Окна истории (дней)", [90, 180, 365], default=[180, 365]) or [180]
        steps_grid = st.multiselect("Шаг точки отсчёта (дней)", [7, 14, 30], default=[30]) or [30]
        metric_key = st.selectbox(
            "Метрика выбора лучшей модели",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS.get(k, k),
            index=0,
        )
        include_experimental = st.checkbox("Показывать экспериментальные модели", value=False)
        model_options = [k for k in MODEL_KIND_LABELS.keys() if include_experimental or k != "mlp"]
        model_defaults = ["lightgbm", "catboost", "logreg"]
        selected_models = st.multiselect(
            "Алгоритмы для сравнения",
            options=model_options,
            default=[m for m in model_defaults if m in model_options],
            format_func=format_model_kind,
        ) or [m for m in model_defaults if m in model_options]
        with st.expander("Тонкая настройка", expanded=False):
            calibration_grid = st.multiselect("Калибровка", ["sigmoid", "isotonic"], default=["sigmoid"]) or ["sigmoid"]

        if "mlp" in selected_models:
            st.warning(
                "Нейронную сеть стоит использовать только на достаточно больших датасетах. "
                "Для табличных данных бустинги часто оказываются сильнее и стабильнее."
            )
            if row_count and row_count < 10000:
                st.warning("В вашем датасете строк пока немного для нейронной сети. Лучше оставить её как эксперимент.")

        selected_models_display = [format_model_kind(m) for m in selected_models]
        params = {
            "mode": "compare",
            "model_name": model_name.strip(),
            "horizon_days_grid": [int(x) for x in horizons_grid],
            "history_days_grid": [int(x) for x in histories_grid],
            "step_days_grid": [int(x) for x in steps_grid],
            "model_kind_grid": [str(x) for x in selected_models],
            "selection_metric": metric_key,
            "calibration_grid": calibration_grid,
            "min_events_in_history": 1,
            "enable_shap": bool(enable_shap),
            "extra_feature_cols": state.extra_feature_cols or [],
            "extra_feature_config": state.extra_feature_config or {},
        }
        experiments = (
            len(params["horizon_days_grid"])
            * len(params["history_days_grid"])
            * len(params["step_days_grid"])
            * len(params["model_kind_grid"])
            * len(params["calibration_grid"])
        )

    with section_card(
        "Что система сделает",
        "Перед запуском можно быстро проверить, какой набор вариантов будет обучаться.",
    ):
        st.write(f"**Модели:** {', '.join(selected_models_display)}")
        st.write(f"**Метрика отбора:** {METRIC_LABELS.get(metric_key, metric_key)}")
        st.write(f"**Оценка сложности запуска:** {_estimated_runtime_hint(experiments)}")
        st.caption(f"Планируется обучить примерно {experiments} вариант(ов).")

        if st.button("Запустить обучение", type="primary", use_container_width=True):
            try:
                resp = api.create_job(
                    file_bytes=state.working_file_bytes,
                    template=state.template,
                    mapping=state.mapping,
                    params=params,
                )
                state.job_id = resp["job_id"]
                state.job_status = None
                state.job_result = None
                st.session_state["_job_just_started"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось запустить обучение: {e}")
                return

    if not state.job_id:
        st.info("После запуска здесь появится прогресс.")
        page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY, next_disabled_reason="Сначала запустите обучение.")
        return

    try:
        state.job_status = api.job_status(state.job_id)
    except Exception as e:
        st.error(f"Не удалось получить статус: {e}")
        page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY, next_disabled_reason="Статус обучения пока недоступен.")
        return

    status = state.job_status or {}
    if status.get("status") not in ("done", "failed"):
        schedule_autorefresh(2500, key="training_job_poll")

    with section_card("Прогресс", "Страница обновляется автоматически, пока задание находится в работе."):
        progress = int(status.get("progress") or 0)
        raw_stage = status.get("stage", "ожидание")
        stage = _stage_label(str(raw_stage))
        st.progress(min(max(progress, 0), 100), text=f"{progress}% — {stage}")

        extra = status.get("extra")
        if extra and isinstance(extra, dict) and extra.get("experiment"):
            st.caption(f"Сейчас: вариант «{extra.get('experiment')}»")

    if status.get("status") == "done":
        st.success("Обучение завершено. Перейдите к шагу «Качество и сравнение».")
        if state.job_result is None or st.session_state.get("_loaded_job_result_id") != state.job_id:
            try:
                state.job_result = api.job_result(state.job_id)
                st.session_state["_loaded_job_result_id"] = state.job_id
            except Exception as e:
                st.error(f"Не удалось загрузить итог: {e}")
                page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY, next_disabled_reason="Сначала дождитесь корректного завершения обучения.")
                return

        result = state.job_result or {}
        saved_params = result.get("params_used") or {}
        display_name = saved_params.get("model_name") or model_name.strip()

        with section_card("Итог", "После успешного завершения здесь появляется краткая сводка по лучшей модели."):
            st.write(f"**Название:** {display_name}")
            st.write(f"**Тип данных:** {result.get('template', state.template)}")
            algo = saved_params.get("model_kind")
            if algo:
                st.write(f"**Алгоритм:** {format_model_kind(str(algo))}")
            hz = saved_params.get("horizon_days")
            if hz is not None:
                st.write(f"**Горизонт (дней):** {hz}")

            if result.get("mode") == "grid_search":
                nexp = result.get("n_experiments")
                if nexp is not None:
                    st.write(f"**Сравнено вариантов:** {nexp}")
                sm = result.get("selection_metric")
                if sm:
                    st.write(f"**Метрика выбора лучшего:** {METRIC_LABELS.get(str(sm), sm)}")
            try:
                report_bytes = _download_job_report_cached(api.base_url, state.job_id)
                st.download_button(
                    "Скачать отчет по обучению zip",
                    data=report_bytes,
                    file_name=f"report_{state.job_id[:8]}.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Не удалось подготовить архив отчётов: {e}")

    elif status.get("status") == "failed":
        st.error(f"Обучение остановилось с ошибкой: {status.get('error')}")
        page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY, next_disabled_reason="Исправьте ошибку и перезапустите обучение.")
        return

    page_nav(STEP_2_ANALYTICS, STEP_4_QUALITY)
